import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from src.config import AA_DATASET


def embed_dict(ab_seq_h, ab_seq_l, prot_seq, seq_natural_embedding):
    names = [ab_seq_h, ab_seq_l, prot_seq]
    seq_features = []
    for i in seq_natural_embedding.index:
        one_line = seq_natural_embedding.iloc[i]
        variable_line = torch.tensor(one_line, dtype=torch.float32)
        seq_features.append(variable_line)
    dictionary = dict(zip(names, seq_features))
    return dictionary


def seq_aaindex_dict(H_chain, L_chain, target_chain, max_len=10 * 256):
    df = pd.read_csv(AA_DATASET, header=0)

    scaler = MinMaxScaler()
    columns_to_normalize = df.columns[1:]
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    feature_selected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    for i in range(len(feature_selected)):
        feature_selected[i] = feature_selected[i] + 1
    column_names = df.columns[feature_selected]
    groups = df.groupby('AA')
    results_dict = {}
    for group_name, group in groups:
        num_rows = group.shape[0]
        values_list = []
        for i in range(num_rows):
            row = group.iloc[i]
            values_list.append(row[column_names].tolist())
        results_dict[group_name] = values_list
    seq_dict = {}
    for seq in [H_chain, L_chain, target_chain]:
        # seq = id2seq(protein, fastaPath)
        seq_feature = []
        for aa in seq:
            if aa in results_dict.keys():
                seq_feature.append(torch.tensor(results_dict[aa][0]))
            if len(seq_feature) < max_len:
                for i in range(max_len - len(seq_feature)):
                    seq_feature.append(torch.zeros(len(feature_selected)))
            else:
                seq_feature = seq_feature[:max_len]
        seq_dict[seq] = torch.stack(seq_feature)
    return seq_dict
