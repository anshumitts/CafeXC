from xc.libs.utils import fetch_json
import argparse
import json


def nullORstr(value):
    if value == 'None':
        return None
    return value


class ParameterBase(object):

    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description)
        self.parser.add_argument('--img_db',  default=None, action='store',
                                 type=str, help='img database')
        self.parser.add_argument('--bucket',  default=1, action='store',
                                 type=int, help='img database')
        self.parser.add_argument('--not_use_module2', action='store_true',
                                 help='If True, it will not perform M2')
        
        self.parser.add_argument('--doc_first', action='store_true',
                                 help='If True, mini batch will be made on document side')
        self.params = None
        self._construct()

    def _construct(self):
        pass

    def parse_args(self):
        self.params = self.parser.parse_args()
        _json = fetch_json(self.params.config, self.params)
        for key, val in _json["DEFAULT"].items():
            self.params.__dict__[key] = val

        for key, val in _json[self.params.model_fname].items():
            self.params.__dict__[key] = val

        configs = _json[self.params.model_fname].items()
        for key, val in configs:
            self.params.__dict__[key] = val

        if self.params.module in [0, 4]:
            for key, val in _json[self.params.ranker].items():
                self.params.__dict__[key] = val

        self.apply_conditions()

    def apply_conditions(self):
        if self.params.txt_model == "BoW":
            for params in ['trn_x_txt', 'tst_x_txt', 'lbl_x_txt']:
                if self.params.__dict__[params] is not None:
                    value = self.params.__dict__[params]
                    value = value.replace(".seq.memmap", ".npz")
                    self.params.__dict__[params] = value

        if self.params.ignore_img:
            self.params.extract_x_img = None
            self.params.trn_x_img = None
            self.params.tst_x_img = None
            self.params.lbl_x_img = None

        if self.params.ignore_txt:
            self.params.extract_x_txt = None
            self.params.trn_x_txt = None
            self.params.tst_x_txt = None
            self.params.lbl_x_txt = None

        if self.params.ignore_lbl_imgs:
            self.params.lbl_x_img = None

    def load(self, fname):
        vars(self.params).update(json.load(open(fname)))

    def save(self, fname):
        print(vars(self.params))
        json.dump(vars(self.params), open(fname, 'w'), indent=4)
