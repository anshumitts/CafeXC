from .dataset import (SiameseData, CrossAttention,
                      RankerPredictDataset, OnlyData,
                      GroupFts, FtsData)
from xc.libs.model_base import (ModelBase, torch, np, sp, os)
import xc.libs.utils as ut
from xc.libs.custom_dtypes import (padded_inputs, FeaturesAccumulator)
from xc.models.models_fusion import Fusion
from xc.libs.anns import ANNSBox
from collections import OrderedDict


class MufinModelBase(ModelBase):
    def __init__(self, params, network, optimizer):
        super().__init__(params, network, optimizer)

    def half_dataset(self, data_dir, doc_img, doc_txt, mode="docs"):
        rand_k = int(os.environ["KEEP_TOP_K"])
        feat = GroupFts(data_dir, doc_img, doc_txt, _type=mode, rand_k=rand_k,
                        max_worker_thread=self.params.max_worker_thread,
                        img_db=self.params.img_db)
        txt_model = self.params.txt_model
        img_model, f_name = None, None
        if "PreTrained" in self.params.model_fname:
            img_model = self.params.img_model
        if doc_txt is not None:
            f_name = doc_txt.split("/")[-1].split(".")[0]
        elif doc_img is not None:
            f_name = doc_img.split("/")[-1].split(".")[0]
        feat.build_pre_trained(txt_model, img_model, f_name, self.params)
        return feat

    def load_ground_truth(self, data_dir, lbl_file, _type="lbl"):
        return FtsData(data_dir, lbl_file, _type=_type)

    def load(self, model_dir, fname):
        print("Loading model..")
        self.filter.load(os.path.join(model_dir, f"filter_{fname}"))
        data = torch.load(os.path.join(model_dir, fname))
        self.net.load_state_dict(data)

    def extract_item(self, tst_dset, desc="docs", mode="test"):
        content = FeaturesAccumulator(f"{desc} content")
        if len(tst_dset) == 0:
            return content
        bz = int(self.params.batch_size*self.params.bucket)
        tst_dl = super().dataloader(
            tst_dset, mode, tst_dset.collate_fn,
            batch_size=bz, num_workers=self.params.num_workers)
        self.net.eval()
        torch.cuda.empty_cache()
        self.net.to()
        with torch.no_grad():
            for batch in ut.pbar(tst_dl, desc=desc, write_final=True):
                with torch.cuda.amp.autocast():
                    vect, mask = self.net(batch["docs"], encode=True)
                    content.transform(vect, mask)
        self.net.cpu()
        content.compile()
        return content

    def extract_modal(self, tst_dset):
        bz = int(self.params.batch_size*self.params.bucket)
        tst_dl = super().dataloader(tst_dset, "test", tst_dset.collate_fn,
                                    batch_size=bz,
                                    num_workers=self.params.num_workers)
        _type = "memmap" if os.environ['RESTRICTMEM'] == '1' else "npy"
        img_content = FeaturesAccumulator("Image", _type)
        txt_content = FeaturesAccumulator("Text", _type)
        self.net = self.net.eval()
        self.net.to()
        with torch.no_grad():
            for batch in ut.pbar(tst_dl, desc="extracting", write_final=True):
                with torch.cuda.amp.autocast():
                    data = self.net(batch["docs"], encode=True)
                    if "img_vect" in data:
                        img_content.transform(
                            data["img_vect"], data["img_mask"])
                    if "txt_vect" in data:
                        txt_content.transform(
                            data["txt_vect"], data["txt_mask"])
        self.net = self.net.cpu()
        img_content.compile()
        txt_content.compile()
        return {"img": img_content, "txt": txt_content}

    def extract(self, data_dir, tst_img, tst_txt):
        self.load(self.params.model_dir, "model.pkl")
        tst_dset = self.half_dataset(data_dir, tst_img, tst_txt)
        return self.extract_modal(tst_dset)


class Mufin(MufinModelBase):
    def __init__(self, params, network, optimizer):
        super(Mufin, self).__init__(
            params, network, optimizer)
        self.anns = ANNSBox(num_labels=params.num_labels, top_k=params.top_k)
        self.mz = self.params.min_leaf_sz

    def adjust_leaf_sz(self, bs, ce, ne):
        self.params.min_leaf_sz = int(self.mz + (bs - self.mz)*ce/ne)

    def _predict(self, tst_dset):
        bz = int(self.params.batch_size*self.params.bucket)
        self.net.eval()
        torch.cuda.empty_cache()
        self.net.to()
        tst_dl = super().dataloader(tst_dset, "test", tst_dset.collate_fn,
                                    batch_size=bz,
                                    num_workers=self.params.num_workers)
        def _out_emb(tst_dl):
            with torch.no_grad():
                for batch in ut.pbar(tst_dl, desc="test", write_final=True):
                    doc, _ = self.net(batch["docs"], encode=True)
                    yield doc.squeeze().cpu().numpy()
        scores = []
        for doc in _out_emb(tst_dl):
            scores.append(self.anns.hnsw_query(doc, bz))
            del doc
        return sp.vstack(scores, format='csr')

    def predict(self, data_dir, tst_img, tst_txt, tst_lbl, lbl_img, lbl_txt):
        tst_dset = self.half_dataset(data_dir, tst_img, tst_txt)
        self.anns.use_hnsw = True
        self.load(self.params.model_dir, "model.pkl")
        self.load_anns(self.params.model_dir)
        docs, lbls = ut.load_overlap(self.params.data_dir,
                                     self.params.filter_labels)
        self.filter.set_docs_lbls(docs, lbls)
        return self._predict(tst_dset)

    def _predict_shorty(self, X_tst, shorty, lbls):
        docs = self.extract_item(X_tst, "docs").mean_pooled.cpu().numpy()
        lbls = ut.normalize(lbls)
        num_inst, num_lbls = shorty.shape
        shorty = padded_inputs(shorty)
        index = shorty['index'].cpu().numpy()
        mask = shorty['mask'].cpu().numpy()
        b_size = self.params.batch_size
        idx = np.array_split(np.arange(num_inst), b_size)
        scores = []
        for _idx in ut.pbar(idx):
            _docs = docs[_idx][:, None, :]
            _lbls = lbls[index[_idx]]
            _mask = mask[_idx]
            _scrs = np.sum(_docs*_lbls, axis=-1)
            scores.append(_scrs*_mask)
        scores = np.vstack(scores)
        score_mat = sp.lil_matrix((num_inst, num_lbls))
        idx = np.arange(num_inst).reshape(-1, 1)
        score_mat[idx, index] = scores
        return score_mat.tocsr()

    def predict_shorty(self, data_dir, tst_img, tst_txt, tst_lbl,
                       tst_shorty, lbl_img, lbl_txt):
        X_tst = self.half_dataset(data_dir, tst_img, tst_txt)
        Y_tst = self.load_ground_truth(data_dir, tst_lbl).data
        self.load(self.params.model_dir, "model.pkl")
        self.anns.use_hnsw = False
        self.load_anns(self.params.model_dir)
        docs, lbls = ut.load_overlap(self.params.data_dir,
                                     self.params.filter_labels)
        self.filter.set_docs_lbls(docs, lbls)
        shorty = self.load_ground_truth(data_dir, tst_shorty, "shorty")
        L_tst = self.half_dataset(data_dir, lbl_img, lbl_txt)
        lbls = self.extract_item(L_tst, "lbls").mean_pooled.cpu().numpy()

        score_mat = self._predict_shorty(X_tst, shorty.data, lbls)
        self.evaluate(score_mat, Y_tst)
        return score_mat

    def get_embs(self, dset, get_docs=False):
        lbls = self.extract_item(dset.L, "lbls", "test")
        lbls = lbls.mean_pooled.cpu().numpy()
        docs = None
        if get_docs:
            docs = self.extract_item(dset.X, "docs", "test")
            docs = docs.mean_pooled.cpu().numpy()
        return docs, lbls

    def callback(self, docs, lbls, ymat, epoch=None):
        self.anns.fit(docs, lbls, ymat)
        self.save_anns(self.params.model_dir)

    def dataloader(self, doc_dset, mode):
        return super().dataloader(doc_dset, mode, doc_dset.collate_fn,
                                  shuffle=mode == "train",
                                  batch_size=self.params.batch_size,
                                  num_workers=self.params.num_workers)

    def retrain(self, data_dir, trn_img, trn_txt, trn_lbl, lbl_img, lbl_txt):
        self.load(self.params.model_dir, "model.pkl")
        if self.params.encoder_init is not None:
            init = os.path.join(self.params.result_dir,
                                self.params.encoder_init)
            self.net.init_encoder(init)
        self.anns.use_hnsw = True
        X = self.half_dataset(data_dir, trn_img, trn_txt)
        L = self.half_dataset(data_dir, lbl_img, lbl_txt)
        Y = self.load_ground_truth(data_dir, trn_lbl)

        docs, lbls = ut.load_overlap(self.params.data_dir,
                                     self.params.filter_labels)
        ymat = Y.data
        if self.params.keep_all:
            ymat = Y.data.shape[1]
        self.filter.fit(docs, lbls, ymat)
        print("Extracting items")
        data = self.extract_item(L, "lbls")
        docs = self.extract_item(X, "docs")
        docs.remap(Y.data.T)
        data = data.hstack(docs)
        self.net.item_encoder.module.module = 3
        str_embs = super().extract_modal(L)
        for key in str_embs.keys():
            if str_embs[key].data is None:
                continue
            data = data.hstack(str_embs[key])
        self.anns.fit_anns(data.data, data.smat.T)
        self.save_anns(self.params.model_dir)
        self.save(self.params.model_dir, "model.pkl")

    def train(self, trn_dset, tst_dset):
        self.iter_batch = 0
        trn_dl = self.dataloader(trn_dset, "train")
        self.construct(trn_dl, self.params.surrogate_warm)
        ws, ne = self.params.warm_start, self.params.num_epochs
        print(f"Warmimg up the model from {0} to {ws} epochs")
        for epoch in np.arange(0, ws):
            _ = self.step(trn_dl, epoch)
            if (epoch) % 10 == 0 and tst_dset is not None:
                docs, lbls = self.get_embs(trn_dset, self.params.hard_pos)
                self.callback(docs, lbls, trn_dset.Y.data, epoch)
                if self.params.not_use_module2:
                    score_mat = self._predict_shorty(
                        tst_dset.X, tst_dset.shorty, lbls)
                else:
                    score_mat = self._predict(tst_dset.X)
                self.evaluate(score_mat, tst_dset.gt)
            self.save(self.params.model_dir, "model.pkl")
        if ne - ws > 0:
            docs, lbls = self.get_embs(trn_dset, self.params.hard_pos)
            trn_dl.dataset.callback_(lbls, docs, self.params)
            trn_dl.b_size = self.params.batch_size
            print(f"Rocking up the model from {ws} to {ne} epochs")
        for epoch in np.arange(ws, ne):
            _ = self.step(trn_dl, epoch)
            if (epoch) % self.params.cl_update == 0:
                docs, lbls = self.get_embs(trn_dset, self.params.hard_pos)
                self.callback(docs, lbls, trn_dset.Y.data, epoch)
                trn_dl.dataset.callback_(lbls, docs, self.params)
            if (epoch) % self.params.validate_after == 0:
                if tst_dset is None:
                    pass
                if self.params.not_use_module2:
                    score_mat = self._predict_shorty(
                        tst_dset.X, tst_dset.shorty, lbls)
                else:
                    score_mat = self._predict(tst_dset.X)
                self.evaluate(score_mat, tst_dset.gt)
            self.save(self.params.model_dir, "model.pkl")

    def fit(self, data_dir, trn_img, trn_txt, trn_lbl,
            tst_img, tst_txt, tst_lbl, lbl_img, lbl_txt):
        if self.params.preload:
            self.load(self.params.model_dir, "model.pkl")
        multi_pos = self.params.multi_pos
        X_trn = self.half_dataset(data_dir, trn_img, trn_txt)
        Y_trn = self.load_ground_truth(data_dir, trn_lbl)
        L = self.half_dataset(data_dir, lbl_img, lbl_txt)
        trn_dset = SiameseData(X_trn, L, Y_trn, multi_pos,
                               "train", self.params.doc_first)
        if self.params.not_use_module2:
            trn_shorty = self.load_ground_truth(
                data_dir, self.params.trn_shorty, "shorty")
            trn_dset.shorty = trn_shorty.data
        tst_dset = None

        if self.params.validate:
            X_tst = self.half_dataset(data_dir, tst_img, tst_txt)
            Y_tst = self.load_ground_truth(data_dir, tst_lbl)
            tst_dset = OnlyData(X_tst, Y_tst)
            if self.params.not_use_module2:
                tst_shorty = self.load_ground_truth(
                    data_dir, self.params.tst_shorty, "shorty")
                tst_dset.shorty = tst_shorty.data

        docs, lbls = ut.load_overlap(self.params.data_dir,
                                     self.params.filter_labels)

        ymat = trn_dset.gt.shape[1]
        if not self.params.keep_all:
            ymat = trn_dset.gt
        self.filter.fit(docs, lbls, ymat)
        self.train(trn_dset, tst_dset)

    def save_anns(self, model_dir):
        print("Saving anns..")
        self.anns.save(model_dir)

    def load_anns(self, model_dir):
        print("loading anns..")
        self.anns.load(model_dir)


class MufinRanker(MufinModelBase):
    def __init__(self, params, network, optimizer):
        super(MufinRanker, self).__init__(params, network, optimizer)
        self.anns = ANNSBox(num_labels=params.num_labels, top_k=params.top_k)
        self.fusion = Fusion(use_psp=True, A=params.A, B=params.B)

    def dataloader(self, doc_dset, mode):
        batch_size = self.params.batch_size
        if mode == "predict":
            batch_size = batch_size//2
        return super().dataloader(doc_dset, mode, doc_dset.collate_fn,
                                  mode == "train", batch_size,
                                  self.params.num_workers)

    def retrain(self, data_dir, trn_img, trn_txt, trn_lbl, lbl_img, lbl_txt):
        anns_dir = os.path.join(self.params.model_dir,
                                f"fusion_{self.params.model_out_name}")
        self.load(self.params.model_dir, self.params.model_out_name)
        self.load_anns(anns_dir)
        shorty_dir = os.path.join(self.params.result_dir, "module2")
        root = self.params.result_dir
        X = self.half_dataset(root, trn_img, trn_txt)
        L = self.half_dataset(root, lbl_img, lbl_txt)
        trn_dset = self.dataset(data_dir, trn_lbl, shorty_dir,
                                "train.npz", X, L, "test")
        trn_dset = next(trn_dset.split_dataset(self.fusion.val_indices))
        self.train_fusion(trn_dset)

    def predict(self, data_dir, tst_img, tst_txt, tst_lbl, lbl_img, lbl_txt):
        data_path = self.params.result_dir
        shorty_dir = os.path.join(data_path, "module2")
        if "pp" in self.params.ranker:
            data_path = data_dir
        X = self.half_dataset(data_path, tst_img, tst_txt)
        L = self.half_dataset(data_path, lbl_img, lbl_txt)
        S = self.load_ground_truth(shorty_dir, "test.npz", "shorty")
        Y = self.load_ground_truth(data_dir, None)
        tst_dset = CrossAttention(X, L, Y, S, 0, 0)
        self.load(self.params.model_dir, self.params.model_out_name)
        return self._predict(tst_dset)

    def _predict(self, tst_dset, X=None, L=None):
        self.net.cpu()
        torch.cuda.empty_cache()
        num_lbls, cached = self.params.num_labels, True
        if os.environ['RESTRICTMEM'] == '1':
            temp_dset, cached = tst_dset, False
        else:
            if X is None:
                X = self.extract_item(tst_dset.X, "docs", "get_docs")
            if L is None:
                L = self.extract_item(tst_dset.L, "lbls", "get_docs")
            shorty = tst_dset.module2
            temp_dset = RankerPredictDataset(X, L, shorty)

        temp_dset = self.dataloader(temp_dset, "predict")
        self.net.preset_weights(L.mean_pooled.cpu())
        self.net.to()
        self.net.eval()
        smats = []
        with torch.no_grad():
            for batch in ut.pbar(temp_dset, desc="Predicting",
                                 write_final=True):
                with torch.cuda.amp.autocast():
                    _scores = self.net(batch, overwrite=True, cached=cached)
                mat = self.build_sparse_matrix(
                    _scores.cpu().numpy(), batch['index'].numpy(),
                    (_scores.size(0), num_lbls), pad_index=num_lbls)
                smats.append(mat)
                del batch, _scores
        del temp_dset
        self.net.cpu()
        smats = sp.vstack(smats, 'csr')
        score = {"module4/m4": smats, "module4/m2": tst_dset.module2}
        return score

    def train_fusion(self, val_dset, L=None):
        score_mat = self._predict(val_dset, L=L)
        self.fusion.fit(val_dset.module2, score_mat["module4/m4"], val_dset.gt)
        anns_dir = os.path.join(self.params.model_dir,
                                f"fusion_{self.params.model_out_name}")
        self.save_anns(anns_dir)

    def callback(self, trn_dset, L=None):
        trn_dset.mode = "test"
        if self.params.sampling and self.params.boosting:
            X = self.extract_item(trn_dset.X, "docs", mode="get_docs")
            score_mat = self._predict(trn_dset, X=X, L=L)["module4/m4"]
            self.evaluate({"trn_m4": score_mat}, trn_dset.gt)
            trn_dset.setup_proba(score_mat)
        trn_dset.mode = "train"
        return trn_dset

    def train(self, trn_dset, tst_dset):
        if self.params.boosting:
            self.fusion.build_psp(trn_dset.gt)
            _, val = self.fusion.split(trn_dset)
            val_dset = trn_dset.split_dataset(val)
        trn_dl = self.dataloader(trn_dset, "train")
        self.iter_batch = 0
        self.construct(trn_dl, self.params.ranker_warm)
        docs, lbls = ut.load_overlap(self.params.data_dir,
                                     self.params.filter_labels)
        ymat = trn_dset.module2
        if self.params.keep_all:
            ymat = trn_dset.gt.shape[1]
        self.filter.fit(docs, lbls, ymat)
        L = None
        for epoch in np.arange(0, self.params.num_epochs):
            _ = self.step(trn_dl, epoch)
            if (epoch) % 5 == 0:
                L = None
                if self.params.validate:
                    if os.environ['RESTRICTMEM'] == '0':
                        L = self.extract_item(
                            trn_dset.L, "lbls", mode="get_docs")
                    score_mat = self._predict(tst_dset, L=L)
                    self.evaluate(score_mat, tst_dset.gt)

                if self.params.boosting:
                    if L is None:
                        L = self.extract_item(
                            trn_dset.L, "lbls", mode="get_docs")
                    self.train_fusion(val_dset, L=L)
                    _ = self.callback(trn_dl.dataset, L=L)

            self.save(self.params.model_dir, self.params.model_out_name)
        self.save(self.params.model_dir, self.params.model_out_name)
        if (epoch) % 5 != 0 and self.params.validate:
            if os.environ['RESTRICTMEM'] == '0':
                L = self.extract_item(trn_dset.L, "lbls", mode="get_docs")
            score_mat = self._predict(tst_dset, L=L)
            self.evaluate(score_mat, tst_dset.gt)
        if self.params.boosting:
            self.train_fusion(val_dset)

    def fit(self, data_dir, trn_img, trn_txt, trn_lbl,
            tst_img, tst_txt, tst_lbl, lbl_img, lbl_txt):

        data_path = self.params.result_dir
        shorty_dir = os.path.join(data_path, "module2")
        if "pp" in self.params.ranker:
            data_path = data_dir
        X_trn = self.half_dataset(data_path, trn_img, trn_txt)
        Y_trn = self.load_ground_truth(data_dir, trn_lbl)
        S_trn = self.load_ground_truth(shorty_dir, "train.npz", "shorty")
        L = self.half_dataset(data_path, lbl_img, lbl_txt)
        num_pos = self.params.sample_pos
        num_neg = self.params.sample_neg
           
        trn_dset = CrossAttention(X_trn, L, Y_trn, S_trn,
                                  num_pos, num_neg, mode="train")

        tst_dset = None
        if self.params.validate:
            X_tst = self.half_dataset(data_path, tst_img, tst_txt)
            Y_tst = self.load_ground_truth(data_dir, tst_lbl)
            S_tst = self.load_ground_truth(shorty_dir, "test.npz", "shorty")
            tst_dset = CrossAttention(X_tst, L, Y_tst, S_tst,
                                      num_pos, num_neg)

        if self.params.encoder_init is not None:
            print("Loading encoder")
            init = os.path.join(self.params.result_dir,
                                self.params.encoder_init)
            self.net.init_encoder(init)
        self.train(trn_dset, tst_dset)

    def save_anns(self, model_dir):
        print("Saving anns..")
        self.fusion.save(model_dir)

    def load_anns(self, model_dir):
        print("loading anns..")
        self.fusion.load(model_dir)