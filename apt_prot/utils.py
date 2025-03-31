from fastapi import HTTPException
import joblib
import warnings
import numpy as np


warnings.filterwarnings("ignore")


def load_model():
    try:
        clf_fs = joblib.load('./models/feature_selector_model.pkl')
        poly = joblib.load('./models/polynomial_transformer.pkl')
        best_model = joblib.load('./models/xgboost_best_model.pkl')
        config = joblib.load('./models/preprocessing_config.pkl')
        features_list = joblib.load('./models/features_list.pkl')
        return clf_fs, poly, best_model, config, features_list
    except:
        err_massage = f"Ошибка при загрузке моделей"
        raise HTTPException(status_code=422, detail=err_massage)

def protcheck(seq):
    """Проверка последовательности на наличие неизвестных аминокислот."""
    valid_amino_acids = set("ARNDCQEGHILKMFPSTWYV")
    return set(seq).issubset(valid_amino_acids)


def extractPAAC(seq, props=["Hydrophobicity", "Hydrophilicity", "SideChainMass"],
                lambda_val=30, w=0.05, customprops=None):
    """
    Расчет псевдоаминокислотного состава (PseAAC) последовательности.

    :param seq: Последовательность белка.
    :param props: Список свойств, используемых для расчета. По умолчанию используются
                  'Hydrophobicity', 'Hydrophilicity', 'SideChainMass'.
    :param lambda_val: Параметр lambda для дескрипторов PseAAC. По умолчанию 30.
    :param w: Весовой коэффициент. По умолчанию 0.05.
    :param customprops: Пользовательские свойства в виде DataFrame.
    :return: Вектор длиной 20 + lambda с именованными значениями.
    """

    if not protcheck(seq):
        raise ValueError("Последовательность содержит неизвестные аминокислоты")

    if len(seq) < lambda_val:
        lambda_val = len(seq) - 1

    if lambda_val <= 0:
        raise ValueError("Длина последовательности должна быть хотя бы 2")

    default_properties = {
        "Hydrophobicity": [0.62, -2.53, -0.78, -0.9, 0.29, -0.74, -0.85, 0.48, -0.4, 1.38, 1.06, -1.5, 0.64, 1.19, 0.12,
                           -0.18, -0.05, 0.81, 0.26, 1.08],
        "Hydrophilicity": [-0.5, 3, 0.2, 3, -1, 3, 0.2, 0, -0.5, -1.8, -1.8, 3, -1.3, -2.5, 0, 0.3, -0.4, -3.4, -2.3,
                           -1.5],
        "SideChainMass": [15, 101, 58, 59, 47, 73, 72, 1, 82, 57, 57, 73, 75, 91, 42, 31, 45, 130, 107, 43]
    }

    all_properties = default_properties.copy()

    if customprops is not None:
        for index, row in customprops.iterrows():
            all_properties[row['AccNo']] = [row['A'], row['R'], row['N'], row['D'], row['C'], row['E'], row['Q'],
                                            row['G'], row['H'], row['I'], row['L'], row['K'], row['M'], row['F'],
                                            row['P'], row['S'], row['T'], row['W'], row['Y'], row['V']]

    standardized_properties = {}
    for prop, values in all_properties.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        standardized_values = [(val - mean_val) / std_val for val in values]
        standardized_properties[prop] = standardized_values

    theta = []
    seq_list = list(seq)
    n = len(seq_list)
    amino_acids = "ARNDCQEGHILKMFPSTWYV"

    for i in range(1, lambda_val + 1):
        theta_i = []
        for j in range(n - i):
            sum_squares = 0
            for prop in props:
                val1 = amino_acids.index(seq_list[j])
                val2 = amino_acids.index(seq_list[j + i])
                sum_squares += (standardized_properties[prop][val1] - standardized_properties[prop][val2]) ** 2
            theta_i.append(sum_squares / len(props))
        theta.append(np.mean(theta_i))

    counts = {aa: seq.count(aa) for aa in amino_acids}
    Xc1 = {f"Xc1.{aa}": count / (1 + w * sum(theta)) for aa, count in counts.items()}

    Xc2 = {f"Xc2.lambda.{i + 1}": w * theta_i / (1 + w * sum(theta)) for i, theta_i in enumerate(theta)}

    Xc = {**Xc1, **Xc2}

    return Xc


def sanitize_sequence(seq):
    return ''.join([c.upper() for c in seq if c.upper() in set('ATCGUatcgu')])


def calculate_kmers(sequence):
    kmers = {}
    clean_seq = sanitize_sequence(sequence)  # <-- очистка

    for k in range(1, 5):
        for i in range(len(clean_seq) - k + 1):
            kmer = clean_seq[i:i + k]
            if f"aptamer_frequency_{kmer}" not in kmers:
                kmers[f"aptamer_frequency_{kmer}"] = 0
            kmers[f"aptamer_frequency_{kmer}"] += 1

    total_length = len(sequence)
    for key in kmers:
        kmers[key] /= total_length

    return kmers


def extract_paac_features(sequence):
    try:
        paac_features = extractPAAC(sequence, lambda_val=min(len(sequence) - 1, 30))
        return paac_features
    except Exception as e:
        print(f"Ошибка для последовательности {sequence}: {e}")
        return {}