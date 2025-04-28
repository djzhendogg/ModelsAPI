import os
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import (
    load_model,
    calculate_molecule_features,
    calculate_rna_features
)

model = load_model("./best_model.pkl")

def predict(seq, smi) -> bool:
    rna_features = calculate_rna_features(seq)
    smi_features = calculate_molecule_features(smi)
    final_df = pd.concat([
        pd.DataFrame(rna_features, index=[0]),
        pd.DataFrame(smi_features, index=[0])
    ], axis=1)
    feature_order = [
        'AATSC2i', 'CCA', 'nG12FRing', 'GATS5i',
        'ACU', 'SMR_VSA9', 'GGC', 'MATS3v',
        'ATSC8i', 'JGI5'
    ]
    X = final_df[feature_order].astype(float)
    ans = np.argmax(model.predict_proba(X))
    if ans == 1:
        return True
    else:
        return False
