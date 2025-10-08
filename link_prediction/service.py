import json, h5py, torch
from torchbiggraph.model import DotComparator, CosComparator, L2Comparator, SquaredL2Comparator
from utils import load_linear_W_from_h5, load_model, load_entity_map

DIM         = 400
# QUERY_NAME  = "110237"
TOPK        = 100
COMPARATOR  = "cos"

entity_names = load_entity_map()
src_idx = entity_names.index(QUERY_NAME)

all_emb = load_all_emb()
src_emb = all_emb[src_idx]
N = all_emb.size(0)

W = load_linear_W_from_h5(MODEL_H5, DIM)

rhs_proj = all_emb @ W.T

# 5. Считаем скоры через comparator.prepare и expand
comp = get_comparator(COMPARATOR)
lhs_prepared = comp.prepare(src_emb.view(1, 1, DIM)).expand(1, N, DIM)  # [1,N,D]
rhs_prepared = comp.prepare(rhs_proj.view(1, N, DIM))                   # [1,N,D]
pos_scores, _, _ = comp(
    lhs_prepared,
    rhs_prepared,
    torch.empty(1, 0, DIM),
    torch.empty(1, 0, DIM),
)
scores = pos_scores.view(-1)  # [N]
scores[src_idx] = float("-inf")

# 6. Top-K
vals, idx = torch.topk(scores, k=min(TOPK, scores.numel()))
# print(f"Top-{TOPK} для '{QUERY_NAME}' (operator=linear, comparator={COMPARATOR}):")
# for r, (i, s) in enumerate(zip(idx.tolist(), vals.tolist()), 1):
#     print(f"{r:2d}. id={i:<8} score={s:.6f}  name={entity_names[i]}")
