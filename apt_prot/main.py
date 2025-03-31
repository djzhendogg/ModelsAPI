from fastapi import FastAPI, Query
from starlette.requests import Request

from service import predict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/aptamer_mol_energy")
@limiter.limit("121/minute")
async def generate_latent_representations(
        request: Request,
        apt_sequences: str = Query(default=""),
        prot_sequences: str = Query(default="")
):

    return predict(apt_sequences, prot_sequences)
