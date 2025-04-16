from fastapi import FastAPI, Query
from starlette.requests import Request

from results.xspecies.TUnA_seed47.pipeline import main
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/protein_protein_binding")
@limiter.limit("121/minute")
async def protein_protein_binding(
        request: Request,
        sequence_1: str = Query(default=""),
        sequence_2: str = Query(default="")
):

    return main(sequence_1, sequence_2)
