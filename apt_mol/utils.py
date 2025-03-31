import numpy as np
np.float=float
from rdkit import Chem
from mordred import Calculator, descriptors
from collections import Counter

from fastapi import HTTPException


# 1. Загрузка модели
def load_model(path):
    try:
        from joblib import load
        model = load(path)
        return model

    except:
        raise HTTPException(status_code=422, detail="Ошибка загрузки модели")

# 2. Обработка РНК-последовательностей
def calculate_rna_features(sequence):
    """Расчет признаков для RNA последовательности"""
    features = {}
    try:
        dinucleotides = [sequence[i:i+2] for i in range(len(sequence)-1)]
        dinu_counts = Counter(dinucleotides)
        total_dinu = sum(dinu_counts.values()) or 1

        features['CCA'] = dinu_counts.get('CCA', 0) / total_dinu
        features['GGC'] = dinu_counts.get('GGC', 0) / total_dinu
        features['ACU'] = dinu_counts.get('ACU', 0) / total_dinu

        return features
    except:
        err_massage = f"Ошибка при расчете дискрипторов из RNA: {sequence}"
        raise HTTPException(status_code=422, detail=err_massage)

# 3. Обработка молекул
def calculate_molecule_features(smiles):
    """Расчет всех дескрипторов Mordred с последующим выбором нужных"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        err_massage = f"Ошибка при создании молекулы из SMILES: {smiles}"
        raise HTTPException(status_code=422, detail=err_massage)
    
    try:
        calc = Calculator(descriptors, ignore_3D=True)
        desc = calc(mol)
        desc_dict = desc.asdict()
        return {
            'AATSC2i': float(desc_dict.get('AATSC2i', np.nan)),
            'SMR_VSA9': float(desc_dict.get('SMR_VSA9', np.nan)),
            'GATS5i': float(desc_dict.get('GATS5i', np.nan)),
            'ATSC8i': float(desc_dict.get('ATS8i', np.nan)),  
            'JGI5': float(desc_dict.get('JGI5', np.nan)),
            'MATS3v': float(desc_dict.get('MATS3v', np.nan)),
            'nG12FRing': int(desc_dict.get('nR12', 0)) 
        }
    except Exception as e:
        err_massage = f"Ошибка при расчете дискрипторов из SMILES: {smiles}"
        raise HTTPException(status_code=422, detail=err_massage)
