from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.hub import load_state_dict_from_url
from torchvision.models.detection import FasterRCNN
from torchvision.ops import boxes as box_ops
from sklearn.preprocessing import normalize
from xclib.data import data_utils as du
import torch.nn.functional as F
import scipy.sparse as sp
from tqdm import tqdm
import numpy as np
import numba as nb
import pickle
import torch
import json
import os
import re


def xc_set_ddp(net, optimizer, num_process):
    #TODO Add DDP support
    return net, optimizer


def xc_unset_ddp(net):
    #TODO Add DDP support
    return net


@nb.njit(cache=True)
def bin_index(array, item):  # Binary search
    first, last = 0, len(array) - 1

    while first <= last:
        mid = (first + last) // 2
        if array[mid] == item:
            return mid

        if item < array[mid]:
            last = mid - 1
        else:
            first = mid + 1

    return -1


def fasterTxtRead(file, chunk=100000, encoding="latin1"):
    with open(file, "r", encoding=encoding) as file:
        data = []
        while True:
            lines = file.readlines(chunk)
            if not lines:
                break
            data.extend(lines)
    return data


@nb.jit(nb.types.Tuple(
    (nb.int64[:], nb.float32[:]))(nb.int64[:], nb.float32[:], nb.int64))
def map_one(indices_labels, similarity, pad_ind):
    unique_point_labels = np.unique(indices_labels)
    unique_point_labels = unique_point_labels[unique_point_labels != pad_ind]
    point_label_similarity = np.ones(
        (len(unique_point_labels), ), dtype=np.float32)
    k = 0
    for lbl in indices_labels:
        if(lbl != pad_ind):
            _ind = bin_index(unique_point_labels, lbl)
            point_label_similarity[_ind] = min(similarity[k],
                                               point_label_similarity[_ind])
        k += 1
    return unique_point_labels, point_label_similarity


@nb.njit(nb.types.Tuple(
    (nb.int32[:], nb.int32[:], nb.float32[:]))(nb.int32[:], nb.int32[:], nb.float32[:]))
def aggregate(ind, ptr, dat):
    _ind = np.zeros(ind.size, dtype=nb.int32)
    _ptr = np.zeros(ptr.size, dtype=nb.int32)
    _dat = -1*np.ones(dat.size, dtype=nb.float32)
    num_rows, d_iter = _ptr.size - 1, 0
    for idx in np.arange(0, num_rows):
        s, e = ptr[idx], ptr[idx+1]
        r_ind = ind[s:e]
        r_dat = dat[s:e]
        ulpt = np.unique(r_ind)
        _ptr[idx] = d_iter
        k = 0
        for lbl in r_ind:
            if(lbl != num_rows):
                t_ind = bin_index(ulpt, lbl)
                _ind[d_iter + t_ind] = lbl
                _dat[d_iter + t_ind] = max(r_dat[k], _dat[d_iter + t_ind])
            k += 1
        d_iter += k
    _ptr[num_rows] = d_iter
    _ind = _ind[:d_iter]
    _dat = _dat[:d_iter]
    return _ind, _ptr, _dat


def ScoreEdges(graph, lbl_emb=None, doc_emb=None, batch_size=5000):
    lbl_emb = normalize(np.nan_to_num(lbl_emb))
    doc_emb = normalize(np.nan_to_num(doc_emb))
    g_lbls, g_docs = graph.nonzero()
    nnz = graph.nnz
    data = np.zeros(nnz)
    for start in tqdm(np.arange(0, nnz, batch_size), desc="prunning"):
        end = min(nnz, start+batch_size)
        _lbl = lbl_emb[g_lbls[start:end]]
        _doc = doc_emb[g_docs[start:end]]
        _dist = np.ravel(np.sum(_lbl*_doc, axis=1))
        data[start:end] = 1.01 - _dist

    far_pos_cone = sp.csr_matrix((data, (g_lbls, g_docs)), shape=graph.shape)
    return far_pos_cone


def load_file(path):
    if path.endswith(".txt"):
        return du.read_sparse_file(path)
    elif path.endswith(".npz"):
        return sp.load_npz(path)
    elif path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".pt"):
        return torch.load(path)
    elif path.endswith(".pkl"):
        return pickle.load(open(path, "rb"))
    elif path.endswith(".memmap"):
        with open(path+".meta", "r") as f:
            elements = f.readline().strip().split(",")
        if elements[0] not in ["int32", "int64", "float32", "float64"]:
            dtype = "float32"
            shape = tuple(map(int, elements))
        else:
            dtype = elements[0]
            shape = tuple(map(int, elements[1:]))
        return np.memmap(path+".dat", dtype=dtype, mode="r", shape=shape)
    else:
        raise TypeError(f"{path} is not supported!!")


def load_overlap(data_dir, filter_label_file='filter_labels'):
    docs = np.asarray([])
    lbls = np.asarray([])
    if os.path.exists(os.path.join(data_dir, filter_label_file)):
        print("UTILS:FILTER:Loading from pre-build file")
        filter_lbs = np.loadtxt(os.path.join(
            data_dir, filter_label_file), dtype=np.int32)
        if filter_lbs.size > 0:
            docs = filter_lbs[:, 0]
            lbls = filter_lbs[:, 1]
    print(f"UTILS:FILTER:Overlap is:{docs.size}")
    return docs, lbls


def save_predictions(preds, result_dir, prefix='predictions'):
    sp.save_npz(os.path.join(result_dir, '{}.npz'.format(prefix)), preds)


def resolve_schema_args(jfile, ARGS):
    """
        Reads JSON and complete the parameters from ARGS
    """
    arguments = re.findall(r"#ARGS\.(.+?);", jfile)
    for arg in arguments:
        replace = '#ARGS.%s;' % (arg)
        to = str(ARGS.__dict__[arg])
        if jfile.find('\"#ARGS.%s;\"' % (arg)) != -1:
            replace = '\"#ARGS.%s;\"' % (arg)
            if isinstance(ARGS.__dict__[arg], str):
                to = str("\""+ARGS.__dict__[arg]+"\"")
        jfile = jfile.replace(replace, to)
    return jfile


def fetch_json(file, ARGS):
    with open(file, encoding='utf-8') as f:
        file = ''.join(f.readlines())
        schema = resolve_schema_args(file, ARGS)
    return json.loads(schema)


class pbar(tqdm):
    def __init__(self, iterable=None, desc=None, total=None, leave=True, file=None,
                 ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
                 ascii=None, disable=False, unit='it', unit_scale=False,
                 dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0,
                 position=None, postfix=None, unit_divisor=1000, write_bytes=None,
                 lock_args=None, nrows=None, colour=None, delay=0, gui=False,
                 write_final=False, **kwargs):
        self.write_final = write_final
        self.disable = False
        super().__init__(iterable, desc, total, leave, file,
                         ncols, mininterval, maxinterval, miniters,
                         ascii, disable, unit, unit_scale,
                         dynamic_ncols, smoothing, bar_format, initial,
                         position, postfix, unit_divisor, write_bytes,
                         lock_args, nrows, colour, delay, gui, **kwargs)

    def close(self):
        if self.disable:
            return
        self.disable = True
        pos = abs(self.pos)
        self._decr_instances(self)

        if self.last_print_t < self.start_t + self.delay:
            return

        if getattr(self, 'sp', None) is None:
            return

        def fp_write(s):
            self.fp.write(str(s))

        try:
            fp_write('')
        except ValueError as e:
            if 'closed' in str(e):
                return
            raise  # pragma: no cover

        leave = pos == 0 if self.leave is None else self.leave

        with self._lock:
            if leave and not self.write_final:
                self._ema_dt = lambda: None
                self.display(pos=0)
                fp_write('\n')
            else:
                if self.display(msg='', pos=pos) and not pos:
                    fp_write('\r')
                if self.write_final:
                    self.write(self.__str__())


def mean_pooling(token_embeddings, attention_mask):  # BxSqxTxD, Bx1xT
    # type: (Tensor, Tensor)->Tensor
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()  # BxSqxTxD
    doc_vects = torch.sum(token_embeddings * input_mask_expanded, -2)  # BxSqxD
    return doc_vects / torch.clamp(input_mask_expanded.sum(-2), min=1e-5)


# @torch.jit.script
def tensor_unique(mat):
    # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    b, s = mat.size()
    mat = mat.view(b*s)
    unique, inverse = torch.unique(mat, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(
        0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    rows = torch.div(perm, s, rounding_mode='trunc')
    cols = perm - rows*s
    return unique, rows, cols, inverse.view(b, s)


def accumulate_output(accumulate, name):
    def hook(module, input, output):
        accumulate[f"{name}_{output.device}"] = output
    return hook


def postprocess_detections(args, class_logits, box_regression,
                           proposals, image_shapes):
    self, feat_key, dict_args = args
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0]
                       for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)
    features_list = F.relu(dict_args[f"{feat_key}_{device}"])
    features_list = features_list.split(boxes_per_image, 0)
    all_boxes = []
    all_scores = []
    all_labels = []
    dim_fts = features_list[0].size(-1)
    vect_fts = torch.zeros(
        (len(features_list), self.detections_per_img, dim_fts),
        device=device)
    mask_fts = torch.zeros(
        (len(features_list), self.detections_per_img),
        device=device)
    for idx, (boxes, scores, image_shape, fts_vect) in enumerate(zip(
            pred_boxes_list, pred_scores_list,
            image_shapes, features_list)):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]
        rows = torch.arange(boxes.size(0)).view(-1, 1)
        rows = rows.repeat(1, scores.size(1))

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        rows = rows.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels, rows = boxes[inds], scores[inds], labels[inds], rows[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels, rows = boxes[keep], scores[keep], labels[keep], rows[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(
            boxes, scores, labels, self.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[:self.detections_per_img]
        boxes, scores, labels, rows = boxes[keep], scores[keep], labels[keep], rows[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        vect_fts[idx, :rows.size(0), :] = fts_vect[rows]
        mask_fts[idx, :rows.size(0)] = 1

    dict_args[f"{feat_key}_fts_{device}"] = {
        "vect": vect_fts, "mask": mask_fts}
    return all_boxes, all_scores, all_labels


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'fasterrcnn_resnet101_fpn_coco':
    "https://ababino-models.s3.amazonaws.com/resnet101_7a82fa4a.pth"
}


def fasterrcnn_resnet_fpn(resnet, pretrained=False, progress=True,
                          num_classes=91, pretrained_backbone=True,
                          trainable_backbone_layers=None, **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone(resnet, pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        urls = model_urls[f"fasterrcnn_{resnet}_fpn_coco"]
        state_dict = load_state_dict_from_url(urls, progress=progress)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
    return model
