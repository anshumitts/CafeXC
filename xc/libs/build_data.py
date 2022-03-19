from genericpath import exists
from xc.tools.tokenize_text import setup_tokenizer, tokens
from xc.models.models_img import Model as MI
from xc.models.models_txt import Model as MT
from xc.libs.custom_dtypes import save
from xc.libs.data_txt import load_txt
import numpy as np
import os


def build_img(IMG, params, cached=True):
    pass


def build_txt(TXT, txt_model, data_dir, file_name, max_len, cached=True):
    cached_path = os.path.join(data_dir, txt_model)
    if cached and os.path.exists(cached_path):
        return load_txt(cached_path, file_name)
    _tokenizer = setup_tokenizer(txt_model)
    input_idx, attention = tokens(TXT, _tokenizer, max_len)
    _tokens = np.stack([input_idx, attention], axis=1)
    if cached:
        os.makedirs(cached_path)
        _file = f"{data_dir}/{file_name}"
        save(_file, "memmap", _tokens)
    return 
