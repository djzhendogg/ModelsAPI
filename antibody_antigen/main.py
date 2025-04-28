from fastapi import FastAPI, Query, HTTPException
from starlette.requests import Request

from prediction_for_one_seq import predict_affinity
from for_one_chain import predict_affinity_one_chain
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/predict_affinity_LH")
@limiter.limit("121/minute")
async def predict_affinity_LH(
        request: Request,
        anitibody: str = Query(default=""),
        antigen: str = Query(default="")

):
    LH = anitibody.split("|")
    if len(LH) < 2:
        return predict_affinity_one_chain(anitibody, antigen)
    return predict_affinity(LH[1], LH[0], antigen)
