from Bio import pairwise2
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem


def calculate_tanimoto_coefficient(molecule1_sequence, molecule2_sequence, kmer_size=3):
    """
    Рассчитывает коэффициент Танимото (Jaccard index) для двух молекул
    на основе их последовательностей.  Использует k-меры.

    Args:
        molecule1_sequence (str): Последовательность первой молекулы (например, ДНК, РНК, белок).
        molecule2_sequence (str): Последовательность второй молекулы.
        kmer_size (int): Размер k-мера (подстроки).  Значение по умолчанию: 3.

    Returns:
        float: Коэффициент Танимото, значение между 0 и 1.  Возвращает 0, если одна или обе последовательности пустые.
    """

    if not molecule1_sequence or not molecule2_sequence:
        return 0.0  # Обработка пустых последовательностей

    def get_kmers(sequence, k):
        """
        Извлекает все k-меры из последовательности.
        """
        kmers = set()
        for i in range(len(sequence) - k + 1):
            kmers.add(sequence[i:i + k])
        return kmers

    kmers1 = get_kmers(molecule1_sequence, kmer_size)
    kmers2 = get_kmers(molecule2_sequence, kmer_size)

    intersection = len(kmers1.intersection(kmers2))
    union = len(kmers1.union(kmers2))

    if union == 0:
        return 0.0  # Обработка случая, когда обе последовательности не содержат kmers (например, если k > длина последовательности)

    tanimoto_coefficient = float(intersection) / union
    return tanimoto_coefficient


def calculate_alignment_similarity(sequence1, sequence2, alignment_type="global", match_score=2, mismatch_penalty=-1, gap_penalty=-0.5, extension_penalty=-0.1):
    """
    Рассчитывает процент совпадения по попарному выравниванию двух последовательностей (ДНК или белок).

    Args:
        sequence1 (str): Первая последовательность (ДНК или белок).
        sequence2 (str): Вторая последовательность.
        alignment_type (str): Тип выравнивания: "global" (алгоритм Нидлмана-Вунша) или "local" (алгоритм Смита-Вотермана). Значение по умолчанию: "global".
        match_score (int): Оценка за совпадение символов. Значение по умолчанию: 2.
        mismatch_penalty (int): Штраф за несовпадение символов. Значение по умолчанию: -1.
        gap_penalty (float): Штраф за гэп (пропуск). Значение по умолчанию: -0.5.
        extension_penalty (float): Штраф за расширение гэпа. Значение по умолчанию: -0.1.

    Returns:
        float: Процент совпавших символов в двух строковых выравниваниях (между 0 и 100). Возвращает 0, если выравнивание не удалось.
    """

    if alignment_type not in ["global", "local"]:
        raise ValueError("alignment_type должен быть 'global' или 'local'")

    if not sequence1 or not sequence2:
        return 0.0

    try:
        if alignment_type == "global":
            # Глобальное выравнивание (алгоритм Нидлмана-Вунша)
            alignments = pairwise2.align.globalms(sequence1, sequence2, match_score, mismatch_penalty, gap_penalty, extension_penalty)
        else:
            # Локальное выравнивание (алгоритм Смита-Вотермана)
            alignments = pairwise2.align.localms(sequence1, sequence2, match_score, mismatch_penalty, gap_penalty, extension_penalty)
    except Exception as e:
        print(f"Ошибка при выравнивании: {e}")
        return 0.0

    if not alignments:
        return 0.0

    # Берем лучшее выравнивание (первое в списке)
    alignment1, alignment2, score, begin, end = alignments[0]

    # Подсчет совпадающих символов
    matches = sum(1 for i in range(len(alignment1)) if alignment1[i] == alignment2[i] and alignment1[i] != '-')

    # Общая длина выравнивания (включая гэпы)
    alignment_length = len(alignment1)

    # Процент совпадения
    similarity_percentage = (float(matches) / alignment_length) * 100 if alignment_length > 0 else 0.0

    return similarity_percentage


def calculate_tanimoto_similarity_smiles(smiles1, smiles2, use_morgan_fp=True, radius=2, nBits=2048):
    """
    Рассчитывает коэффициент Танимото для двух молекул, представленных SMILES-строками.

    Args:
        smiles1 (str): SMILES-строка первой молекулы.
        smiles2 (str): SMILES-строка второй молекулы.
        use_morgan_fp (bool): Использовать ли Morgan Fingerprints (ECFP4). Если False, используются RDKit Fingerprints.  Значение по умолчанию: True.
        radius (int): Радиус для Morgan Fingerprints.  Значение по умолчанию: 2 (что соответствует ECFP4).
        nBits (int): Количество битов для Fingerprints.  Значение по умолчанию: 2048.

    Returns:
        float: Коэффициент Танимото (значение между 0 и 1). Возвращает None, если одна или обе SMILES недействительны.
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return None

        if use_morgan_fp:
            # Morgan Fingerprints
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=radius, nBits=nBits)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=radius, nBits=nBits)
        else:
            # RDKit Fingerprints
            fp1 = Chem.RDKFingerprint(mol1, maxPath=7, fpSize=nBits)
            fp2 = Chem.RDKFingerprint(mol2, maxPath=7, fpSize=nBits)

        tanimoto_coefficient = TanimotoSimilarity(fp1, fp2)
        return tanimoto_coefficient

    except Exception as e:
        print(f"Ошибка при расчете Танимото: {e}")
        return None