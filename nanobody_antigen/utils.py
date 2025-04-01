import pickle
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def load_model():
    # Загрузка артефактов
    with open('./model_nanobody.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('./pca.pkl', 'rb') as f:
        pca = pickle.load(f)

    # Загрузка уникальных k-меров
    with open('./unique_kmers.txt', 'r') as f:
        unique_kmers = [line.strip() for line in f]
    return model, pca, unique_kmers

def process_sequence(seq, unique_kmers, k=3):
    """Преобразует последовательность в вектор признаков"""
    # Очистка последовательности
    seq_clean = seq.replace("O","A").replace("B","A").replace("X","A")
    
    # Генерация k-меров
    kmers = [seq_clean[i:i+k] for i in range(len(seq_clean)-k+1)]
    
    # Вектор частот
    freq = [0] * len(unique_kmers)
    for mer in kmers:
        if mer in unique_kmers:
            idx = unique_kmers.index(mer)
            freq[idx] += 1
    
    # Биохимические характеристики
    try:
        X = ProteinAnalysis(seq_clean)
        epsilon_prot = X.molar_extinction_coefficient()
        sec_struc = X.secondary_structure_fraction()
        features = [
            X.charge_at_pH(1),
            X.gravy(),
            X.molecular_weight(),
            X.aromaticity(),
            X.instability_index(),
            X.isoelectric_point(),
            sec_struc[0], sec_struc[1], sec_struc[2],
            epsilon_prot[0], epsilon_prot[1]
        ]
    except:
        features = [0]*11  # обработка ошибок в последовательности
    
    return np.array(freq + features)
