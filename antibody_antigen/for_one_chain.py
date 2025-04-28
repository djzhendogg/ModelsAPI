import torch
from def_for_one_chain import ModelAffinity, get_feature, seq_aaindex_dict, embed_dict
from src.config import MODEL_PATH


batch_size = 16
use_cuda = torch.cuda.is_available()

device = torch.device('cpu')

OneChainmodel = ModelAffinity(bs=batch_size, use_cuda=use_cuda)

model_path = MODEL_PATH
state_dict = torch.load(model_path, map_location='cuda' if use_cuda else 'cpu')
OneChainmodel.load_state_dict(state_dict)


OneChainmodel.eval()

def predict_affinity_one_chain(Lchain=None, Hchain=None, antigen=None):
    """Основная функция предсказания с поддержкой None"""
    # Проверка входных данных
    input_seqs = [seq for seq in [Hchain, Lchain, antigen] if seq is not None]
    if not input_seqs:
        raise ValueError("Требуется хотя бы одна цепь антитела (H или L, или антиген)")
    
    # Генерация эмбеддингов
    seq_natural_embedding = get_feature(Hchain=Hchain, Lchain=Lchain, antigen=antigen)
    #embed_shape = [768]
    embeddings = embed_dict(Hchain, Lchain, antigen, seq_natural_embedding)
    
    # Подготовка AAIndex фич
    #aaind_shape = [2560,12]
    aaindex_data = seq_aaindex_dict(Hchain, Lchain, antigen)
    

    lchain_embeddings = []
    hchain_embeddings = []
    ag_embeddings = []
    lchain_aaindex = []
    hchain_aaindex = []
    ag_aaindex = []

    try:
        hchain_embedding = embeddings['H_chain']
        lchain_embedding = embeddings['L_chain']
        ag_embedding = embeddings['target_chain']
        
        lchain_aa = aaindex_data['L_chain']
        hchain_aa = aaindex_data['H_chain']
        ag_aa = aaindex_data['target_chain']

        # Сохраняем данные для модели
        lchain_embeddings.append(lchain_embedding)
        hchain_embeddings.append(hchain_embedding)
        ag_embeddings.append(ag_embedding)
        
        lchain_aaindex.append(lchain_aa)
        hchain_aaindex.append(hchain_aa)
        ag_aaindex.append(ag_aa)

    except Exception as e:
        print(f"Error processing sample ({Hchain}/{Lchain}/{antigen}): {str(e)}")

    # [1,2560,20]
    h_aa = torch.stack(hchain_aaindex, 0).to(device) if hchain_aaindex is not None else torch.zeros(1, 20).to(device)
    l_aa = torch.stack(lchain_aaindex, 0).to(device) if lchain_aaindex is not None else torch.zeros(1, 20).to(device)
    ag_aa = torch.stack(ag_aaindex, 0).to(device) if ag_aaindex is not None else torch.zeros(1, 20).to(device)
    

    # Эмбеддинги
    #[1,768]
    h_emb = torch.stack(hchain_embeddings, 0).to(device) if embeddings['H_chain'] is not None else torch.zeros(1,768).to(device)
    l_emb = torch.stack(lchain_embeddings, 0).to(device) if embeddings['L_chain'] is not None else torch.zeros(1,768).to(device)
    ag_emb = torch.stack(ag_embeddings, 0).to(device) if embeddings['target_chain'] is not None else torch.zeros(768).to(device)

    with torch.no_grad():
        prediction = OneChainmodel.predict(l_aa, h_aa, ag_aa, l_emb, h_emb, ag_emb)
    
    return prediction.item()
#
#
# H_chain = "AAASDDDSDF"
# target_chain = "MNMNSDAMNR"
#
# print(predict_affinity(Hchain=H_chain, antigen=target_chain))