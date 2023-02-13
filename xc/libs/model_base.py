from xclib.evaluation.xc_metrics import precision, ndcg, format, recall, compute_inv_propesity, psprecision
from .utils import pbar, aggregate, xc_set_ddp, xc_unset_ddp
from xc.libs.dataloader import DataLoader as dl
from .model_utils import FilterShortlist
import scipy.sparse as sp
import numpy as np
import torch
import time
import os


class ModelBase:
    def __init__(self, params, network, optimizer):
        self.params = params
        self.filter = FilterShortlist(self.params.keep_all)
        self.accumulate = self.params.accumulate
        self.scaler = torch.cuda.amp.GradScaler()
        self.net, self.optim = xc_set_ddp(network, optimizer, params.bucket)

    def aggregate_mats(self, score, mappings):
        score = score.tocsc()
        rows, cols = mappings.nonzero()
        num_lbls = score.shape[1]
        num_docs = mappings.shape[0]
        h_map = np.zeros(cols.size, np.int32)
        h_map[cols] = rows
        index, indptr, data, end_ptr = [], [], [], []
        lbls = np.arange(0, num_lbls, self.params.batch_size)
        for start in pbar(lbls, desc="aggr"):
            end = min(start+self.params.batch_size, num_lbls)
            _score = score[:, start:end].T.tocsr()
            ind, ptr, dat = h_map[_score.indices], _score.indptr, _score.data
            ind, ptr, dat = aggregate(ind, np.int32(ptr), np.float32(dat))
            ptr += len(index)
            indptr.extend(ptr[:-1])
            end_ptr = ptr[-1]
            index.extend(ind)
            data.extend(dat)
            del ind, ptr, dat, _score
        del score
        indptr.append(end_ptr)
        score = sp.csr_matrix((data, index, indptr),
                              shape=(num_lbls, num_docs))
        return score.T.tocsr()

    def _bucket(self, mode):
        bucket = 1 if mode != "train" else self.params.bucket
        return bucket

    def dataloader(self, doc_dset, mode, collate_fn, shuffle=False,
                   batch_size=1024, num_workers=6):
        return dl(data=doc_dset, batch_size=batch_size, drop_last=False,
                  shuffle=shuffle, params=self.params, collate_fn=collate_fn,
                  num_workers=num_workers, num_process=self._bucket(mode))

    def construct(self, dl, warmup_steps=1000):
        self.optim.construct(self.net, lr=self.params.lr, accumulate=self.accumulate,
                             trn_dl=dl, warmup_steps=warmup_steps,
                             num_epochs=self.params.num_epochs)
    
    def step(self, dataloader, n_epoch):
        self.net = self.net.to()
        self.net = self.net.train()
        self.net.callback()
        self.params.curr_epoch = n_epoch
        _loss, iter_batch = {}, 0
        deno = len(dataloader)
        for batch in pbar(dataloader, write_final=True,
                          desc="(Epoch:%03d)" % (n_epoch)):
            iter_batch += 1
            if batch is None:
                continue
            with torch.cuda.amp.autocast():
                output = self.net(batch)
                loss = 0
                for _k, _l in output.items():
                    loss += _l
                    _loss[_k] = _loss.get(_k, 0) + _l.item()/deno
            self.scaler.scale(loss).backward()
            if iter_batch % self.accumulate == 0 or \
                    iter_batch % len(dataloader) == 0:
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.adjust_lr()
                self.optim.zero_grad()
                self.net.callback()
            del batch
        self.net = self.net.eval()
        self.net = self.net.cpu()
        print(f"Avg. loss {_loss}")
        return _loss

    def predict(self, data_dir, tst_img, tst_txt, tst_lbl, **kwargs):
        pass

    def callback(self, trn_dset):
        pass

    def train(self, trn_dset, tst_dset):
        pass

    def fit(self, data_dir, trn_img, trn_txt, trn_lbl,
            tst_img, tst_txt, tst_lbl, **kwargs):
        pass

    def build_sparse_matrix(self, data, index, shape, pad_index=None):
        rows = np.arange(shape[0]).reshape(-1, 1)
        rows = np.ravel(np.repeat(rows, index.shape[1], axis=1))
        index = np.ravel(index)
        data = np.ravel(data)
        if pad_index is not None:
            rows = rows[index != pad_index]
            data = data[index != pad_index]
            index = index[index != pad_index]

        score_mat = sp.lil_matrix(shape)
        score_mat[rows, index] = data
        return score_mat.tocsr()

    def _filter(self, pred_mat):
        return self.filter(pred_mat)

    def _evaluate(self, pred_mat, true_mat):
        pred_mat = self._filter(pred_mat)
        _prec = precision(pred_mat, true_mat, k=5)
        _ndcg = ndcg(pred_mat, true_mat, k=5)
        _recall = recall(pred_mat, true_mat, k=self.params.top_k)
        inv_prop = compute_inv_propesity(true_mat, self.params.A, self.params.B)
        _psp = psprecision(pred_mat, true_mat, inv_prop, k=5)
        print(f"recall@{self.params.top_k}-> {_recall[-1]*100}")
        print(f"prec-> {format(_prec[::2])}")
        print(f"ndcg-> {format(_ndcg[::2])}")
        print(f"psp-> {format(_psp[::2])}")

    def evaluate(self, pred_mat, true_mat):
        if isinstance(pred_mat, dict):
            for key, mat in pred_mat.items():
                print(key)
                self._evaluate(mat, true_mat)
        else:
            self._evaluate(pred_mat, true_mat)

    def retrain(self, data_dir, trn_img, trn_txt,
                trn_lbl, lbl_img, lbl_txt):
        pass

    def save(self, model_dir, fname):
        print("Saving network..")
        self.filter.save(os.path.join(model_dir, f"filter_{fname}"))
        net = xc_unset_ddp(self.net)
        torch.save(net.state_dict(), os.path.join(model_dir, fname))

    def load(self, model_dir, fname):
        print("Loading model..")
        self.filter.load(os.path.join(model_dir, f"filter_{fname}"))
        data = torch.load(os.path.join(model_dir, fname))
        self.net.load_state_dict(data)
        # except RuntimeError as e:
        #     self.net.module.load_state_dict(data)

    def extract_encoder(self):
        self.load(self.params.model_dir, self.params.model_out_name)
        return self.net.save_encoder()
