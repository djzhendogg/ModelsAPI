import torch

from embed import get_feature
from src.models.mvsf_for_one_seq import ModelAffinity
from utils_for_one_seq import embed_dict, seq_aaindex_dict
from src.config import MODEL_PATH

batch_size = 16  # Should match the batch size used during training
use_cuda = torch.cuda.is_available()  # Set to True if using GPU

model = ModelAffinity(bs=batch_size, use_cuda=use_cuda)


# Load the trained weights
model_path = MODEL_PATH
state_dict = torch.load(model_path, map_location='cuda' if use_cuda else 'cpu')
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

if use_cuda:
    model.cuda()


def predict_affinity(Lchain, Hchain, antigen):
    list_for_features = [Hchain, Lchain, antigen]

    seq_natural_embedding = get_feature(list_for_features)
    embedding_tensor = embed_dict(Hchain, Lchain, antigen, seq_natural_embedding)
    aaindex_feature = seq_aaindex_dict(Hchain, Lchain, antigen)
    use_cuda = torch.cuda.is_available()
    model.eval()
    b = len(Hchain)

    # Списки для тензоров (только успешные элементы)
    lchain_embeddings = []
    hchain_embeddings = []
    ag_embeddings = []
    lchain_aaindex = []
    hchain_aaindex = []
    ag_aaindex = []

    device = next(model.parameters()).device

    try:
        # Пытаемся получить все необходимые данные
        lchain_embedding = embedding_tensor[Lchain]
        hchain_embedding = embedding_tensor[Hchain]
        ag_embedding = embedding_tensor[antigen]

        lchain_aa = aaindex_feature[Lchain]
        hchain_aa = aaindex_feature[Hchain]
        ag_aa = aaindex_feature[antigen]

        # Сохраняем данные для модели
        lchain_embeddings.append(lchain_embedding)
        hchain_embeddings.append(hchain_embedding)
        ag_embeddings.append(ag_embedding)

        lchain_aaindex.append(lchain_aa)
        hchain_aaindex.append(hchain_aa)
        ag_aaindex.append(ag_aa)

        # valid_indices.append(i)
        ##success_mask[i] = True

    except Exception as e:
        print(f"Error processing sample ({Hchain}/{Lchain}/{antigen}): {str(e)}")

    device = next(model.parameters()).device
    # Конвертируем в тензоры
    lchain_embeddings = torch.stack(lchain_embeddings, 0).to(device)
    hchain_embeddings = torch.stack(hchain_embeddings, 0).to(device)
    ag_embeddings = torch.stack(ag_embeddings, 0).to(device)

    lchain_aaindex = torch.stack(lchain_aaindex, 0).to(device)
    hchain_aaindex = torch.stack(hchain_aaindex, 0).to(device)
    ag_aaindex = torch.stack(ag_aaindex, 0).to(device)

    if use_cuda:
        lchain_embeddings = lchain_embeddings.cuda()
        hchain_embeddings = hchain_embeddings.cuda()
        ag_embeddings = ag_embeddings.cuda()
        lchain_aaindex = lchain_aaindex.cuda()
        hchain_aaindex = hchain_aaindex.cuda()
        ag_aaindex = ag_aaindex.cuda()

    # Предсказание
    with torch.no_grad():
        ph = model.predict(lchain_aaindex, hchain_aaindex, ag_aaindex,
                           lchain_embeddings, hchain_embeddings, ag_embeddings)

    ph = ph.cpu().numpy().flatten().item()
    return ph
