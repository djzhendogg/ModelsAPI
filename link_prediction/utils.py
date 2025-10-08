from fastapi import HTTPException
import joblib
import warnings
import numpy as np
from config import ENTITY_JSON


warnings.filterwarnings("ignore")


def load_model():
    return

def load_all_emb():
    with h5py.File(EMB_H5, "r") as hf:
        all_emb = torch.from_numpy(hf["embeddings"][...]).float()
    return all_emb

def load_entity_map():
    with open(ENTITY_JSON, "rt") as tf:
        entity_names = json.load(tf)
    return entity_names

def get_comparator(name: str):
    return {
        "cos": CosComparator(),
        "dot": DotComparator(),
        "l2": L2Comparator(),
        "squared_l2": SquaredL2Comparator(),
    }[name]

def load_linear_W_from_h5(path_h5: str, dim: int) -> torch.Tensor:
    candidates = [
        "model/relations/0/operator/rhs/weight",
        "model/relations/0/operator/weight",
        "model/relations/0/operator/matrix",
        "model/relations/0/weight",
        "model/operator/weight",
    ]
    with h5py.File(path_h5, "r") as hf:
        for key in candidates:
            if key in hf:
                W = torch.from_numpy(hf[key][...]).float()
                if tuple(W.shape) == (dim, dim):
                    return W
        # fallback: ищем любой [dim,dim] с 'operator'
        target = None
        def visit(name, obj):
            nonlocal target
            if target is None and isinstance(obj, h5py.Dataset):
                if obj.shape == (dim, dim) and "operator" in name.lower():
                    target = torch.from_numpy(obj[...]).float()
        hf.visititems(visit)
        if target is not None:
            return target
    raise RuntimeError("Не найден матриц W_r в H5")
