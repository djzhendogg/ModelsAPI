import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from tape import TAPETokenizer
from src.config import AA_DATASET, BERT_PATH

device = torch.device('cpu')


class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        batch_size, channel_size, seq_len, feature_dim = x.size()
        x = x.view(batch_size, channel_size * seq_len, feature_dim)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (feature_dim ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_scores, values)
        weighted_values = weighted_values.view(batch_size, channel_size, seq_len, feature_dim)

        return weighted_values


class ModelAffinity(nn.Module):
    def __init__(
            self,
            bs,
            use_cuda,
    ):
        super(ModelAffinity, self).__init__()
        self.use_cuda = use_cuda
        self.bs = bs
        self.self_attention = SelfAttention(768)

        self.layernorm = nn.LayerNorm(normalized_shape=(4, 768, 768))

        self.conv0 = nn.Conv2d(4, 1, kernel_size=1)
        self.conv1 = nn.Conv1d(768, 384, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(384, 1, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(768, 384, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(384, 1, kernel_size=3, padding=1)
        self.linear = nn.Linear(768, 1)
        self.linear2 = nn.Linear(4, 1)

        self.fusion1 = FusionModel(200 * 256, 256)
        # self.fusion1 = FusionModel(20*256, 256)
        self.fusion2 = FusionModel(200 * 256, 256)
        # self.fusion2 = FusionModel(20*256, 256)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(768 + 512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

        self.aa_conv1 = nn.Conv1d(256 * 3, 256, kernel_size=7, padding=3)
        self.aa_conv2 = nn.Conv1d(256, 1, kernel_size=7, padding=3)
        self.aa_fc = nn.Linear(103, 1)

        self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(768)

    def feature_fusion(self, lchain, hchain, antigen):
        # [1,768]
        antigen = antigen.unsqueeze(1).unsqueeze(2)
        lchain = lchain.unsqueeze(1).unsqueeze(3)
        hchain = hchain.unsqueeze(1).unsqueeze(3)

        l_dif = torch.abs(lchain - antigen)
        l_mul = lchain * antigen
        l_cat = torch.cat([l_dif, l_mul], 1)

        h_dif = torch.abs(hchain - antigen)
        h_mul = hchain * antigen
        h_cat = torch.cat([h_dif, h_mul], 1)

        ab_cat = torch.cat([l_cat, h_cat], 1)
        C = self.layernorm(ab_cat)
        return C

    # module 1: predict from embedding (proteinBert features)
    def map_predict(self, lchain, hchain, antigen):
        if self.use_cuda:
            # [1,768]
            lchain = lchain.cuda()
            hchain = hchain.cuda()
            antigen = antigen.cuda()

        C = self.feature_fusion(lchain, hchain, antigen)
        B = self.conv0(C)
        B = B.squeeze(1)
        B = self.relu(self.conv1(B))
        B = self.conv2(B)
        B = B.squeeze(1)
        B = self.batchnorm2(B)
        return B

    # module 2: predict from original amino acid sequence (aaindex1 feature)
    def aa_predict(self, lchain, hchain, antigen):
        l_ag = self.fusion1(lchain, antigen)
        h_ag = self.fusion2(hchain, antigen)
        x = torch.cat([l_ag, h_ag], 1)
        x = self.batchnorm(x)
        x = F.dropout(x, p=0.5, training=self.training)

        return x

    def predict(self, lchain_aaindex, hchain_aaindex, ag_aaindex, lchain_embedding, hchain_embedding, ag_embedding):
        # B  [1, 768]
        phat_1 = self.map_predict(lchain_embedding, hchain_embedding, ag_embedding)
        # x [1,512]
        phat_2 = self.aa_predict(lchain_aaindex, hchain_aaindex, ag_aaindex)
        x = torch.cat([phat_1, phat_2], 1)
        x = (self.fc2(x))
        x = (self.fc3(x))
        x = (self.fc4(x))
        phat = self.activation(x)
        return phat

    def forward(self, lchain_id, hchain_id, ag_id, aaindex_feature, lchain_embedding, hchain_embedding, ag_embedding):
        return self.predict(lchain_id, hchain_id, ag_id, aaindex_feature, lchain_embedding, hchain_embedding,
                            ag_embedding)


class FusionModel(nn.Module):
    def __init__(self, input_dim, fusion_dim):
        super(FusionModel, self).__init__()
        self.fusion_matrix = nn.Linear(input_dim, fusion_dim)

    def forward(self, x1, x2):
        x1_flat = x1.view(x1.size(0), -1).float()
        x2_flat = x2.view(x2.size(0), -1).float()
        x1_transformed = self.fusion_matrix(x1_flat)
        x2_transformed = self.fusion_matrix(x2_flat)

        fused_tensor = x1_transformed + x2_transformed
        return fused_tensor


def get_feature(Hchain=None, Lchain=None, antigen=None):
    """Генерирует эмбеддинги с сохранением информации об источнике"""
    device = torch.device('cpu')
    model = torch.load(BERT_PATH)
    model = model.to(device)
    model.eval()
    tokenizer = TAPETokenizer(vocab='iupac')

    feature_dict = {}  # Словарь для хранения эмбеддингов

    for source, seq in zip(['H_chain', 'L_chain', 'target_chain'], [Hchain, Lchain, antigen]):
        if seq is None:
            # Нулевой эмбеддинг размерности 768
            null_embedding = [0.0] * 768
            feature_dict[source] = torch.tensor(null_embedding, dtype=torch.float32)  # Store as tensor
        else:
            # Обработка обычных последовательностей
            token_ids = torch.tensor([tokenizer.encode(seq)])
            with torch.no_grad():
                output = model(token_ids.to(device))
                pooled_output = output[1]
                feature_dict[source] = pooled_output[0].cpu().to(device)  # Store as tensor

    return feature_dict


def seq_aaindex_dict(H_chain=None, L_chain=None, target_chain=None, max_len=10 * 256):
    """Создает словарь AAIndex-фичей для последовательностей"""
    # Загрузка и нормализация данных
    df = pd.read_csv(AA_DATASET, header=0)

    scaler = MinMaxScaler()
    columns_to_normalize = df.columns[1:]
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    # Выбор признаков
    feature_selected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    for i in range(len(feature_selected)):
        feature_selected[i] = feature_selected[i] + 1
    column_names = df.columns[feature_selected]

    # Группировка по аминокислотам
    groups = df.groupby('AA')
    results_dict = {}
    for group_name, group in groups:
        num_rows = group.shape[0]
        values_list = []
        for i in range(num_rows):
            row = group.iloc[i]
            values_list.append(row[column_names].tolist())
        results_dict[group_name] = values_list

    # Создание словаря последовательностей
    seq_dict = {}

    # Обработка каждой цепи
    for seq_name, seq in zip(['H_chain', 'L_chain', 'target_chain'], [H_chain, L_chain, target_chain]):
        if seq is not None:
            seq_feature = []
            for aa in seq:
                if aa in results_dict.keys():
                    seq_feature.append(torch.tensor(results_dict[aa][0], dtype=torch.float32))
                else:
                    seq_feature.append(torch.zeros(len(feature_selected), dtype=torch.float32))  # Unknown AA
            # Дополнение последовательности нулями до max_len
            if len(seq_feature) < max_len:
                for i in range(max_len - len(seq_feature)):
                    seq_feature.append(torch.zeros(len(feature_selected), dtype=torch.float32))
            else:
                seq_feature = seq_feature[:max_len]
            seq_dict[seq_name] = torch.stack(seq_feature)
        else:
            # Если цепь отсутствует, создать тензор нулей
            seq_dict[seq_name] = torch.zeros(max_len, len(feature_selected), dtype=torch.float32).to(device)

    return seq_dict


def embed_dict(ab_seq_h=None, ab_seq_l=None, prot_seq=None, seq_natural_embedding=None):
    """Генерирует словарь с автозаполнением отсутствующих цепей"""
    # Создаем словарь с обязательными ключами
    dictionary = {
        'H_chain': None,
        'L_chain': None,
        'target_chain': None
    }

    # Заполняем только существующие последовательности
    if ab_seq_h is not None:
        dictionary['H_chain'] = seq_natural_embedding.get(
            'H_chain') if seq_natural_embedding and 'H_chain' in seq_natural_embedding else [0] * 768
    else:
        dictionary['H_chain'] = torch.zeros(768)
    if ab_seq_l is not None:
        dictionary['L_chain'] = seq_natural_embedding.get(
            'L_chain') if seq_natural_embedding and 'L_chain' in seq_natural_embedding else [0] * 768
    else:
        dictionary['L_chain'] = torch.zeros(768)
    if prot_seq is not None:
        dictionary['target_chain'] = seq_natural_embedding.get(
            'target_chain') if seq_natural_embedding and 'target_chain' in seq_natural_embedding else [0] * 768
    else:
        dictionary['target_chain'] = torch.zeros(768)

        # Конвертируем в тензоры
    for key in dictionary:
        if dictionary[key] is not None:
            dictionary[key] = torch.tensor(dictionary[key], dtype=torch.float32)

    return dictionary
