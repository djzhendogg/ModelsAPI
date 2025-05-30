import torch.cuda.nvtx
import contextlib

# Патчинг NVTX функций
torch.cuda.nvtx.range_push = lambda *args, **kwargs: None
torch.cuda.nvtx.range_pop = lambda *args, **kwargs: None

# Заглушка для контекстного менеджера
@contextlib.contextmanager
def nvtx_range(*args, **kwargs):
    yield

from fastapi import FastAPI, Query
from starlette.requests import Request

from network.predict_dna_protein import predictorBase
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/dna_protein_binding")
@limiter.limit("121/minute")
async def mfe_rna_rna(
        request: Request,
        protein_sequences: str = Query(default=""),
        dna_sequences: str = Query(default="")
):
    label, avg_lddt, interface_pae = predictorBase.predict(
            protein_seq=protein_sequences,
            dna_seq=dna_sequences,
        )
    return label
