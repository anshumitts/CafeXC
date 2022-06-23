from transformers import (AutoTokenizer, CLIPTokenizer,
                          DistilBertTokenizerFast)
from tokenizers.processors import TemplateProcessing
from tokenizers import BertWordPieceTokenizer
from xc.libs.utils import pbar, load_file
from xc.libs.custom_dtypes import save
import scipy.sparse as sp
import numpy as np
import argparse
import tempfile
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def tokens(text, _tokenizer, max_len):
    text = _tokenizer(text, truncation=True, padding='max_length',
                      max_length=max_len, add_special_tokens=True,
                      return_tensors="pt", return_length=True)
    return np.int32(text.input_ids), np.int32(text.attention_mask)


def to_sparse(index, mask, max_vocab):
    num_docs = len(index)
    rows = list(map(lambda x: [x[0]]*x[1].size, enumerate(mask)))
    cols = np.asarray(np.concatenate(index), np.int32)
    rows = np.asarray(np.concatenate(rows), np.int32)
    data = np.asarray(np.concatenate(mask), np.int32)
    smat_data = sp.coo_matrix((data, (rows, cols)),
                              shape=(num_docs, max_vocab))
    smat_data.sum_duplicates()
    smat_data.eliminate_zeros()
    return smat_data.tocsr()


def build_idf(lbl_map, trn_map, _tokenizer, args):
    max_vocab = _tokenizer.vocab_size
    lbl_idx, lbl_msk = tokens(lbl_map, _tokenizer, args)
    trn_idx, trn_msk = tokens(trn_map, _tokenizer, args)
    index = np.vstack([lbl_idx, trn_idx])
    masks = np.vstack([lbl_msk, trn_msk])
    print("Building IDFs")
    s_mat = to_sparse(index, masks, max_vocab)
    _idf = np.log(s_mat.shape[0]+1) - np.log(s_mat.getnnz(axis=0)+1) + 1
    _idf[np.logical_or(_idf == 1, _idf == np.log(s_mat.shape[0]+1)+1)] = 0
    print("# of useless tokens", np.where(
        _idf == 0)[0], np.where(_idf == 0)[0].size)
    idf = sp.diags(_idf, shape=(max_vocab, max_vocab)).tocsr()
    idf.eliminate_zeros()
    sp.save_npz(os.path.join(args.out_dir, "idf.npz"), idf)
    return idf


def build_vocab(args):
    _tokenizer = BertWordPieceTokenizer()
    raw_data = tempfile.NamedTemporaryFile(mode="w+")
    with open(args.trn_map, "r", encoding="latin1") as fp:
        for line in pbar(fp, desc=args.trn_map):
            line = line.strip().split("->", 1)[1]
            raw_data.write(f"{line}\n")

    with open(args.lbl_map, "r", encoding="latin1") as fp:
        for line in pbar(fp, desc=args.lbl_map):
            line = line.strip().split("->", 1)[1]
            raw_data.write(f"{line}\n")
    raw_data.seek(0)
    _tokenizer.train([raw_data.name], vocab_size=args.n_vocab)
    _tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", _tokenizer.token_to_id("[CLS]")),
                        ("[SEP]", _tokenizer.token_to_id("[SEP]"))])
    vocab = dict([(v, k) for (k, v) in _tokenizer.get_vocab().items()])
    num_vocab = len(vocab)
    print("Num Vocab", num_vocab)
    vocab_data = tempfile.NamedTemporaryFile(mode="w+")
    for key in range(num_vocab):
        vocab_data.write("{}\n".format(vocab[key]))
    vocab_data.seek(0)
    token = BertWordPieceTokenizer(
        vocab_data.name, model_max_length=args.max_len,
        bos_token="[CLS]", eos_token="[SEP]",
        unk_token="[UNK]", sep_token="[SEP]",
        pad_token="[PAD]", mask_token="[MASK]")
    raw_data.close()
    vocab_data.close()
    return token


def setup_tokenizer(txt_model):
    if txt_model in ["ClipViT"]:
        _tokenizer = CLIPTokenizer.from_pretrained(
            f"openai/clip-vit-base-patch32", do_lower_case=True)
    elif txt_model in ["bert-tiny", "bert-mini-mnli"]:
        _tokenizer = AutoTokenizer.from_pretrained(
            f"prajjwal1/{txt_model}", do_lower_case=True)
    elif txt_model in ["sentencebert"]:
        print("Using sentence bert")
        _tokenizer = DistilBertTokenizerFast.from_pretrained(
            "sentence-transformers/msmarco-distilbert-base-v4",
            do_lower_case=True)
    else:
        _tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)
    # idf = build_idf(args.lbl_map, args.trn_map, _tokenizer, args)
    # _tokenizer.idf = idf
    return _tokenizer


def tokenize(text_map, text_xf, _tokenizer, args):

    if not os.path.exists(text_map):
        print(f"File {text_map} not found!!")
        return
    path = os.path.join(args.raw_dir, text_map)
    text = list(map(lambda x: x.strip().split("->", 1)[1].replace("_", " ").lower(),
                    pbar(open(path, "r", encoding="latin1"), desc=text_map)))
    input_idx, attention = tokens(text, _tokenizer, args.max_len)
    max_vocab = _tokenizer.vocab_size
    if args.txt_model in ["BoW"]:
        s_mat = to_sparse(input_idx, attention, max_vocab)
        s_mat = s_mat.dot(_tokenizer.idf).tocsr()
        s_mat.sort_indices()
        text_xf = text_xf.replace(".seq", ".npz")
        sp.save_npz(text_xf, s_mat)
    else:
        print("Average tokens / documents =",
              (np.sum(attention)-2)/attention.shape[0])
        _tokens = np.stack([input_idx, attention], axis=1)
        save(text_xf, "memmap", _tokens)


def write(tmp_mdata, labels, features):
    print("# labels:", labels.size)
    print("# features:", features.size)
    path = os.path.join(tmp_mdata, 'labels_split.txt')
    np.savetxt(path, labels, fmt='%d')
    path = os.path.join(tmp_mdata, 'features_split.txt')
    np.savetxt(path, features, fmt='%d')


def setup():
    parser = argparse.ArgumentParser("Combine evaluate")
    parser.add_argument('--data_dir', dest='data_dir',
                        action='store', type=str)
    parser.add_argument('--raw_dir', dest='raw_dir',
                        action='store', type=str)
    parser.add_argument('--out_dir', dest='out_dir',
                        action='store', type=str)
    parser.add_argument('--zsh_map', dest='zsh_map', action='store', type=str)
    parser.add_argument('--trn_map', dest='trn_map', action='store', type=str)
    parser.add_argument('--tst_map', dest='tst_map', action='store', type=str)
    parser.add_argument('--lbl_map', dest='lbl_map', action='store', type=str)
    parser.add_argument('--zsh_xf', dest='zsh_xf', action='store', type=str)
    parser.add_argument('--trn_xf', dest='trn_xf', action='store', type=str)
    parser.add_argument('--tst_xf', dest='tst_xf', action='store', type=str)
    parser.add_argument('--lbl_xf', dest='lbl_xf', action='store', type=str)
    parser.add_argument('--trn_y', dest='trn_y', action='store', type=str)

    parser.add_argument('--max_len', dest="max_len", action="store", type=int)
    parser.add_argument('--n_vocab', dest="n_vocab", action="store", type=int)
    parser.add_argument('--txt_model', dest="txt_model",
                        action="store", type=str)
    params = parser.parse_args()

    def apply_conditions(params):
        if params.txt_model == "BoW":
            for _param in ['trn_xf', 'tst_xf', 'lbl_xf', 'zsh_xf']:
                if params.__dict__[_param] is not None:
                    params.__dict__[_param] = params.__dict__[
                        _param].replace(".seq.memmap", ".npz")
        return params
    return apply_conditions(params)


if __name__ == '__main__':
    args = setup()
    print(args)
    tokenizer = setup_tokenizer(args.txt_model)
    tokenize(args.lbl_map, args.lbl_xf, tokenizer, args)
    tokenize(args.trn_map, args.trn_xf, tokenizer, args)
    tokenize(args.tst_map, args.tst_xf, tokenizer, args)
    tokenize(args.zsh_map, args.zsh_xf, tokenizer, args)
    trn_y = load_file(args.trn_y)
    trn_x_xf = load_file(args.trn_xf)
    lbl_x_xf = load_file(args.lbl_xf)
    features = np.arange(tokenizer.vocab_size)
    labels = np.arange(trn_y.shape[1])
    write(args.out_dir, labels, features)
