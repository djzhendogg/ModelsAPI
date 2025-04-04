from fastapi import FastAPI, Query
from starlette.requests import Request

from service import (
    calculate_tanimoto_similarity_smiles,
    calculate_tanimoto_coefficient,
    calculate_alignment_similarity
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# @app.post("/tanimot_sequence")
# @limiter.limit("121/minute")
# async def mfe_rna_rna(
#         request: Request,
#         sequences_1: str = Query(default=""),
#         sequences_2: str = Query(default="")
# ):
#
#     return calculate_tanimoto_coefficient(sequences_1, sequences_2)


@app.post("/tanimot_smiles")
@limiter.limit("121/minute")
async def mfe_rna_rna(
        request: Request,
        smiles_1: str = Query(default=""),
        smiles_2: str = Query(default="")
):

    return calculate_tanimoto_similarity_smiles(smiles_1, smiles_2)



@app.post("/alignment_similarity")
@limiter.limit("121/minute")
async def mfe_rna_rna(
        request: Request,
        sequences_1: str = Query(default=""),
        sequences_2: str = Query(default="")
):
    return calculate_alignment_similarity(sequences_1, sequences_2)
