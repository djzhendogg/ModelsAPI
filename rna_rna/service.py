import RNA


def predict(rna1: str, rna2: str) -> float:

    input_seq = f"{rna1}&{rna2}"

    # MFE: minimum free energy
    ans = RNA.cofold(input_seq)
    print(ans)
    (fc, mfe) = ans
    return mfe
