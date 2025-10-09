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


@app.post("/link_prediction")
@limiter.limit("121/minute")
async def aptamer_prot_binding(
        target_id: str
):
    if len(target_id) < 1:
        error_text = "ID is invalid"
        raise HTTPException(status_code=404, detail=error_text)
    try:
        return  {"result": predict(target_id)}
    except:
        raise HTTPException(status_code=422)
