from fastapi import FastAPI, Query, HTTPException
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
        rna_sequences: str = Query(default=""),
        mol_smiles: str = Query(default="")
):

    return predict(rna_sequences, mol_smiles)
