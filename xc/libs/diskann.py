import numpy as np
import diskannpy as diks
import scipy.sparse as sp
import json
import os
import time
from xc.libs.utils import KshardedOva
from sklearn.preprocessing import normalize
from sklearn_extra.cluster import KMedoids
import shutil


# +
def LoL2Sparse(list_of_list, num_cols, padd=True, data=None):
    num_rows = len(list_of_list)
    rows = np.concatenate(list(map(lambda x: [x[0]]*len(x[1]),
                               enumerate(list_of_list))))
    cols = np.concatenate(list_of_list)
    sparse_mat = sp.lil_matrix((num_rows+padd, num_cols+padd), dtype=np.float32)
    if data is None:
        data = np.ones(rows.size, dtype=np.float32)
    data = np.ravel(data)
    sparse_mat[rows, cols] = data
    return sparse_mat.tocsr()


class DiskANN:
    def __init__(self, anns_args= {"L": 300, "R": 100,
                                   "C": 750, "alpha": 1.2,
                                   "saturate_graph": False,
                                   "num_threads": 128},
                 metric="l2", dtype=np.single, num_threads=32,
                 model_path="./"):
        self.model_path = model_path
        self.metric = metric
        self.dtype = dtype
        self.num_threads = num_threads
        self.args = {"ANNS": anns_args}
        self.args["META"] = {}
        self.set_params(self.args["ANNS"])
        self.vects = None
        self.index = None
    
    def set_params(self, args):
        self.params = {}
        for key in args.keys():
            self.params[key] = args[key]
    
    def optimize(self):
        pass
        
    def to_bin(self, vect, path="./"):
        path = os.path.join(path, "vect.bin")
        with open(path, 'wb') as f:
            f.write(vect.shape[0].to_bytes(4, "little"))
            f.write(vect.shape[1].to_bytes(4, "little"))
            f.write(vect.astype(np.float32).tobytes())
        return path
    
    def get_vect(self):
        path = os.path.join(self.model_path, "index.bin.data")
        with open(path, 'rb') as f:
            N = int.from_bytes(f.read(4), "little")
            D = int.from_bytes(f.read(4), "little")
            data = np.fromfile(f, dtype="float32")
        return data.reshape(N, D)
        
    def fit(self, vects, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.model_path = model_path
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(self.model_path)
        self.vects = normalize(np.float32(vects))
        start = time.time()
        if os.path.exists(os.path.join(model_path, "vectors.bin")):
            print("Removing old vectors.bin")
            os.remove(os.path.join(model_path, "vectors.bin"))
        
        diks.build_memory_index(
            data=self.vects,
            distance_metric="l2",
            vector_dtype=np.single,
            index_directory=model_path,
            complexity=self.params["L"],
            graph_degree=self.params["R"],
            num_threads=self.num_threads,
            index_prefix="index.bin",
            alpha=1.2,
            use_pq_build=False,
            num_pq_bytes=8,
            use_opq=False,
        )
        self.index = diks.StaticMemoryIndex(
            distance_metric="l2",
            vector_dtype=np.single,
            index_directory=model_path,
            num_threads=self.num_threads,  # this can be different at search time if you would like
            initial_search_complexity=self.params["L"],
            index_prefix="index.bin"
        )
        end = time.time()
        self.args["META"]["num_nodes"] = vects.shape[0]
        print(f"build_time {end-start}")
    
    def save(self, out_path=None):
        if out_path is None:
            out_path = self.model_path        
        self.model_path = out_path
        os.makedirs(out_path, exist_ok=True)
        print("Model is already saved")
        path = os.path.join(out_path, "args.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.args, f, ensure_ascii=False, indent=4)
    
    def load(self, in_path):
        if in_path is None:
            in_path = self.model_path
        self.model_path = in_path
        self.index = diks.StaticMemoryIndex(
            distance_metric="l2",
            vector_dtype=np.single,
            index_directory=in_path,
            num_threads=self.num_threads,  # this can be different at search time if you would like
            initial_search_complexity=self.params["L"],
            index_prefix="index.bin"
        )
        
        path = os.path.join(in_path, "args.json")
        with open(path, 'r', encoding='utf-8') as f:
            self.args = json.load(f)
        self.set_params(self.args["ANNS"])
   
    
    def get_index(self, offset=6, sparse=False, prefix="index.bin"):
        path = os.path.join(self.model_path, f"{prefix}")
        data = np.fromfile(path, dtype='uint32')   #1, 2, 2, 1, 2, 3, 1, 2, 3
        num_nodes = 0
        _index = []
        _meta = data[:offset]
        while(offset < data.shape[0]):
            num_nodes += 1
            start = offset + 1
            end = start + data[offset]
            _index.append(data[start:end])
            offset = end  
        del data 
        if sparse:
            _index = LoL2Sparse(_index, self.args["META"]["num_nodes"], False)
        self._meta = _meta
        return _meta, _index
    
    def update(self, LoL, _meta=None, vect=None, outfile = None):
        
        if _meta is None:
            print("Using preset meta")
            _meta = self._meta
        if outfile is not None:
            if os.path.exists(outfile):
                shutil.rmtree(outfile)
            os.makedirs(outfile, exist_ok=True)
            self.model_path = outfile
            if vect is None:
                vect = self.vects
            else:
                print("Normalizing the vectors")
                vect = normalize(vect)
        
        if vect is None:
            vect = os.path.join(self.model_path, "index.bin.data")
        
        path = os.path.join(self.model_path, "index.bin")
        kmedoids = KMedoids(n_clusters=1, random_state=22).fit(vect)
        _meta[3] = kmedoids.medoid_indices_[0]
        print("New metadata", _meta)
        
        if sp.issparse(LoL):
            LoL = LoL.tolil()
            LoL[_meta[3], :] = 1
            LoL = LoL.rows
        new_idx = []
        for u in range(len(LoL)):
            new_idx.extend([len(LoL[u])] + LoL[u])
        
        file_size = len(new_idx) * 4 + 24
        x = np.array(new_idx).astype(np.uint32)
        with open(path, 'wb') as f:
            f.write(file_size.to_bytes(8, "little"))
            f.write(_meta[2:].tobytes())
            f.write(x.tobytes())
        
        diks.update_memory_index(path, "l2", self.model_path,
                                 self.params["R"], self.params["R"],
                                 self.num_threads, 1.2,
                                 np.single, "index.bin", vect)
        self.save(self.model_path)
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
        q_data = normalize(q_data)
        if L is None:
            L = self.args["ANNS"]["L"]
        
        if R is None:
            R = self.args["ANNS"]["R"]
        idx, scr, = self.index.batch_search(q_data, R, L, self.num_threads)
        return idx, -scr # converting distance to similarity

