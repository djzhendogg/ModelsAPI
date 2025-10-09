import json, h5py, torch
from torchbiggraph.model import DotComparator, CosComparator, L2Comparator, SquaredL2Comparator
from utils import load_linear_w_from_h5, load_entity_map, get_comparator
from config import TOPK, DIM, COMPARATOR_TYPE

INTERNAL_TO_DBID_MAP = load_entity_map()
EMBEDDINGS_MAP = load_all_emb()
W = load_linear_w_from_h5()
COMPARATOR = get_comparator(COMPARATOR_TYPE)


def predict(dbid: str):
    target_id = INTERNAL_TO_DBID_MAP.index(dbid)
    target_embedding = EMBEDDINGS_MAP[target_id]
    N = EMBEDDINGS_MAP.size(0)
    rhs_proj = EMBEDDINGS_MAP @ W.T

    lhs_prepared = COMPARATOR.prepare(target_embedding.view(1, 1, DIM)).expand(1, N, DIM)
    rhs_prepared = COMPARATOR.prepare(rhs_proj.view(1, N, DIM))
    pos_scores, _, _ = COMPARATOR(
        lhs_prepared,
        rhs_prepared,
        torch.empty(1, 0, DIM),
        torch.empty(1, 0, DIM),
    )
    scores = pos_scores.view(-1)
    scores[target_id] = float("-inf")
    _, idx = torch.topk(scores, k=min(TOPK, scores.numel()))
    result = [entity_names[i] for i in idx.tolist()]
    return result
