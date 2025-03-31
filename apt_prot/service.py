import os
import pandas as pd
from sklearn.feature_selection import SelectFromModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import (
    load_model, calculate_kmers, extractPAAC
)

clf_fs, poly, best_model, config, features_list = load_model()

def predict(apt: str, prot: str) -> bool:
    apt_tr = calculate_kmers(apt)

    prot_tr = extractPAAC(prot, lambda_val=min(len(prot) - 1, 30))
    all_columns_kmers = set()
    all_columns_paac = set()

    all_columns_kmers.update(apt_tr.keys())
    all_columns_paac.update(prot_tr.keys())

    dna_df = pd.DataFrame(apt_tr, index=[0])
    pprot_df = pd.DataFrame(prot_tr, index=[0])

    all_columns_kmers = sorted(list(all_columns_kmers))
    desc_na = dna_df[all_columns_kmers]

    all_columns_paac = sorted(list(all_columns_paac))
    paac_features_df = pprot_df[all_columns_paac]

    full_df = pd.concat([desc_na, paac_features_df], axis=1)

    full_df = full_df.fillna(0)
    selection = SelectFromModel(
        estimator=clf_fs,
        threshold=config['threshold'],
        prefit=config['prefit']
    )
    if set(full_df.columns) != set(features_list):
        print("Предупреждение: новые данные имеют другие признаки. Применяем преобразование.")
        full_df = full_df.reindex(columns=features_list, fill_value=0)

    X_selected = selection.transform(full_df)

    X_poly = poly.transform(X_selected)
    ans = best_model.predict(X_poly).item()
    if ans > 0.9:
        return True
    else:
        return False
