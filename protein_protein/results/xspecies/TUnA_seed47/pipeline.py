from .modeltools import TesterPipeline
from .custom_loader import load_model_custom
from .utils import (
    load_configuration,
    set_random_seed,
    get_computation_device,
)

import esm
import torch
from .config import CONFIG_PATH


big_model = load_model_custom()
model_esm, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model_esm.to("cpu")
batch_converter = alphabet.get_batch_converter()


def process_interactions_with_sequences(sequence_a, sequence_b, device):
    """
    Обрабатывает файл взаимодействий (с последовательностями) и возвращает эмбеддинги белков.

    Аргументы:
        interaction_file (str): Путь к файлу с взаимодействиями (sequence A, sequence B, label).
        device (torch.device): Устройство для вычислений ('cuda' или 'cpu').

    Возвращает:
        embedding_dict (dict): Словарь уникальных последовательностей и их эмбеддингов.
    """

    # Шаг 3: Извлечение эмбеддингов для уникальных последовательностей
    unique_sequences = set()
    unique_sequences.add(sequence_a)
    unique_sequences.add(sequence_b)

    embedding_dict = {}
    for sequence in unique_sequences:
        # Конвертируем последовательность в токены
        _, _, batch_tokens = batch_converter([("protein", sequence)])
        batch_tokens = batch_tokens.to(device)

        # Прогоняем через модель и извлекаем эмбеддинги
        with torch.no_grad():
            results = model_esm(batch_tokens, repr_layers=[30], return_contacts=True)
            embedding = results["representations"][30][0, 1:-1, :].to("cpu")  # Убираем токены [START] и [END]
            embedding_dict[sequence] = embedding

    return embedding_dict


def main(sequence_a, sequence_b):
    """
    Главная функция для предсказания взаимодействия двух белков.

    Аргументы:
        sequence_a (str): Последовательность первого белка.
        sequence_b (str): Последовательность второго белка.
        device (str): Устройство для вычислений ('cuda' или 'cpu').
    """
    # --- Pre-Training Setup ---
    # Load configs. Use config file to change hyperparameters.
    config = load_configuration(CONFIG_PATH)

    # Set random seed for reproducibility
    set_random_seed(config['other']['random_seed'])

    # Determine the computation device (CPU or GPU)
    device = get_computation_device(config['other']['cuda_device'])

    # Initialize the testing modules
    tester = TesterPipeline(big_model)

    # Обрабатываем последовательности и получаем эмбеддинги
    embedding_dict = process_interactions_with_sequences(sequence_a, sequence_b, device)

    # Извлекаем эмбеддинги из словаря
    embedding_a = embedding_dict[sequence_a]
    embedding_b = embedding_dict[sequence_b]

    # Создаем датасет из одной пары
    protA_lens = len(embedding_a)  # Длина первой последовательности
    protB_lens = len(embedding_b)  # Длина второй последовательности
    batch_protA_max_length = protA_lens  # Максимальная длина первой последовательности в батче
    batch_protB_max_length = protB_lens  # Максимальная длина второй последовательности в батче

    # Создаем датасет с 7 элементами в каждом кортеже

    # dataset = [(embedding_a, embedding_b, 0, protA_lens, protB_lens, batch_protA_max_length, batch_protB_max_length)] # Метка 0 используется как заглушка
    dataset = [(embedding_a, embedding_b, 0)]

    # --- Make Prediction ---
    max_seq_length = 100
    protein_dim = config['model']['protein_embedding_dim']

    loss, T, Y, S = tester.test(dataset, max_seq_length, protein_dim, last_epoch=True)

    # Вывод результата
    # print("Predicted Interaction Label:", Y[0])  # Предсказанная метка (0 или 1)
    # print("Predicted Interaction Score:", S[0])  # Вероятность взаимодействия (от 0 до 1)
    if Y[0] == 1:
        return True
    else:
        return False


# Execute the main function when the script is run
if __name__ == "__main__":
    sequence_a = "MKTAYIAKQRQISFVKSHFSRQDLDLK"  # Последовательность первого белка
    sequence_b = "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKSEDEMKASEDLKKHG"  # Второго белка

    main(sequence_a, sequence_b)
