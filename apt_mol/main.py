from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from service import predict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/aptamer_mol_binding")
@limiter.limit("121/minute")
async def aptamer_mol_binding(
        request: Request,
        rna_sequences: str,
        mol_smiles: str
):
    # # res = {}
    # res = []
    # seq_pair_list = sequences.split(";")
    #
    # if len(seq_pair_list) > 502:
    #     error_text = "The number of sequences in the query exceeds 500"
    #     raise HTTPException(status_code=429, detail=error_text)
    #
    # for seq_pair in seq_pair_list:
    #     ss = seq_pair.split(":")
    #     try:
    #         # predict(rna_sequences, mol_smiles)
    #         ans = predict(ss[0], ss[1])
    #     except:
    #         ans = None
    #     # res[ss] = ans
    #     res.append(ans)
    #
    # return {"result": res}
    return predict(rna_sequences, mol_smiles)
