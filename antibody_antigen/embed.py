import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from protein_bert_pytorch import ProteinBERT
from tape import TAPETokenizer
from tqdm import tqdm

from src.config import BERT_PATH

device = torch.device('cpu')
bert_model = torch.load(BERT_PATH)
bert_model = bert_model.to(device)
for param in bert_model.parameters():
    param.requires_grad = False
bert_model.eval()

def get_feature(_list):
    # load model
    tokenizer = TAPETokenizer(vocab='iupac')  # iupac是TAPE模型的词汇表，UniRep模型使用unirep。
    feature = []
    for seq in tqdm(_list):  # 进度条
        token_ids = torch.tensor([tokenizer.encode(seq)])
        output = bert_model(token_ids.to(device))
        pooled_output = output[1]
        feature.append(pooled_output[0].tolist())
    _df = pd.DataFrame(np.array(feature))
    return _df


def get_feature2():
    model = ProteinBERT(
        num_tokens=21,
        num_annotation=8943,
        dim=512,
        dim_global=256,
        depth=6,
        narrow_conv_kernel=9,
        wide_conv_kernel=9,
        wide_conv_dilation=5,
        attn_heads=8,
        attn_dim_head=64
    )

    seq = torch.randint(0, 21, (2, 2048))
    mask = torch.ones(2, 2048).bool()
    annotation = torch.randint(0, 1, (2, 8943)).float()

    seq_logits, annotation_logits = model(seq, annotation, mask=mask)


def parse(f, comment="#"):
    names = []
    sequences = []

    for record in SeqIO.parse(f, "fasta"):
        names.append(record.name)
        sequences.append(str(record.seq))

    return names, sequences
