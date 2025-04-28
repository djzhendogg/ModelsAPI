import torch
from tape import TAPETokenizer, ProteinBertModel
from src.config import BERT_PATH

model = ProteinBertModel.from_pretrained('bert-base')
torch.save(model, BERT_PATH)