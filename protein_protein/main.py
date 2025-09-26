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
        sequences: str = Query(default="")
):
    res = {}
    seq_pair_list = sequences.split(";")

    if len(seq_pair_list) > 502:
        error_text = "The number of sequences in the query exceeds 500"
        raise HTTPException(status_code=429, detail=error_text)

    for seq_pair in seq_pair_list:
        ss = seq_pair.split(">")
        try:
            ans = main(ss[0], ss[1])
        except:
            ans = None
        res[seq_pair] = ans
    return {"result": res}
