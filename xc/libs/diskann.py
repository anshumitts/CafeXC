import numpy as np
import vamanapy as vp
import scipy.sparse as sp
import json
import os
import time
import tempfile


# +
def LoL2Sparse(list_of_list, num_cols, padd=True):
    num_rows = len(list_of_list)
    rows = np.concatenate(list(map(lambda x: [x[0]]*len(x[1]),
                               enumerate(list_of_list))))
    cols = np.concatenate(list_of_list)
    sparse_mat = sp.lil_matrix((num_rows+padd, num_cols+padd), dtype=np.int32)
    sparse_mat[rows, cols] = np.arange(rows.size) + 1
    return sparse_mat.tocsr()


class DiskANN:
    def __init__(self, anns_args= {"L": 300, "R": 100,
                                   "C": 750, "alpha": 1.2,
                                   "saturate_graph": False,
                                   "num_threads": 128},
                 model_path="./"):
        vp.set_num_threads(anns_args["num_threads"])
        self.model_path = model_path
        self.args = {"ANNS": anns_args}
        self.args["META"] = {}
        self.set_params(self.args["ANNS"])
        self.vects = None
        self.index = None
    
    def set_params(self, args):
        self.params = vp.Parameters()
        for key in args.keys():
            self.params.set(key, args[key])
    
    def optimize(self):
        pass
        
    def to_bin(self, vect, path="./"):
        path = os.path.join(path, "vect.bin")
        with open(path, 'wb') as f:
            f.write(vect.shape[0].to_bytes(4, "little"))
            f.write(vect.shape[1].to_bytes(4, "little"))
            f.write(vect.astype(np.float32).tobytes())
        return path
        
    def fit(self, vects, model_path=None):
        if model_path is None:
            model_path = self.model_path
        temp_dir = tempfile.TemporaryDirectory()
        self.model_path = model_path
        path = self.to_bin(vects, temp_dir.name)
        self.vects = vects
        start = time.time()
        self.index = vp.SinglePrecisionIndex(vp.Metric.INNER_PRODUCT, path)
        self.index.build(self.params, [])
        end = time.time()
        num_nodes, num_edges = self.index.get_stats()
        self.args["META"]["num_nodes"] = vects.shape[0]
        print(f"build_time {end-start} num_nodes={num_nodes} num_edges={num_edges}")
        temp_dir.cleanup()
    
    def save(self, out_path=None):
        if out_path is None:
            out_path = self.model_path
        
        self.model_path = out_path
        os.makedirs(out_path, exist_ok=True)
        _ = self.to_bin(self.vects, out_path)
        path = os.path.join(out_path, "index.bin")
        self.index.save(path)
        path = os.path.join(out_path, "args.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.args, f, ensure_ascii=False, indent=4)
    
    def load(self, in_path):
        if in_path is None:
            in_path = self.model_path
        path = os.path.join(in_path, "vect.bin")
        self.index = vp.SinglePrecisionIndex(vp.Metric.INNER_PRODUCT, path)
        path = os.path.join(in_path, "index.bin")
        self.index.load(file_name = path)
        
        path = os.path.join(in_path, "args.json")
        with open(path, 'r', encoding='utf-8') as f:
            self.args = json.load(f)
        self.set_params(self.args["ANNS"])
   
    
    def get_index(self, offset=4, sparse=False):
        path = os.path.join(self.model_path, "index.bin")
        data = np.fromfile(path, dtype='uint32')   #1, 2, 2, 1, 2, 3, 1, 2, 3
        _index = []
        _meta = data[:offset]
        while(offset < data.shape[0]):
            start = offset + 1
            end = start + data[offset]
            _index.append(data[start:end])
            offset = end  
        del data 
        
        if sparse:
            _index = LoL2Sparse(_index, self.args["META"]["num_nodes"], False)
        return _meta, _index
    
    def set_index(self, LoL, _meta, outfile = None):
        if sp.issparse(LoL):
            LoL = LoL.tolil().rows
        
        if outfile is not None:
            self.model_path = outfile
            self.save(self.model_path)

        path = os.path.join(self.model_path, "index.bin")
        
        new_idx = []
        for u in range(len(LoL)):
            new_idx.extend([len(LoL[u])] + LoL[u])
        
        file_size = len(new_idx) * 4 + 16
        x = np.array(new_idx).astype(np.uint32)
        with open(path, 'wb') as f:
            f.write(file_size.to_bytes(8, "little"))
            f.write(_meta[2:4].tobytes())
            f.write(x.tobytes()) 
        
        self.load(self.model_path)
        return True
    
    def update_node_vect(self, node_vect):
        N, d = node_vect.shape
        assert N == self.args["META"]["num_nodes"]
        self.index.update_node_vect(node_vect, N, d)
    
    def _set_query_time_params(self, R):
        anns_args= {"L": 3*R, "R": R,
                    "C": 7*R, "alpha": 1.2,
                    "saturate_graph": False,
                    "num_threads": 128}
        self.args["ANNS"] = anns_args

    def query(self, q_data, R=None, L=None):
        num_q = q_data.shape[0]
        if L is None:
            L = self.args["ANNS"]["L"]
        
        if R is None:
            R = self.args["ANNS"]["R"]
        idx, scr, _, _ = self.index.batch_numpy_query(q_data, R, num_q, L)
        idx = idx.reshape(num_q, R)
        scr = scr.reshape(num_q, R)
        return idx, scr

