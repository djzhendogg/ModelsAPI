from .modeltools import (IntraEncoder, InterEncoder, ProteinInteractionNet, TesterPipeline)
from .utils import set_random_seed, load_configuration, \
    get_computation_device
from uncertaintyAwareDeepLearn import VanillaRFFLayer
import torch
from .config import CONFIG_PATH, MODEL_PATH


def load_model_custom():
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

    # --- Model Initialization ---
    # Initialize the Encoder, Decoder, and overall model
    intra_encoder = IntraEncoder(config['model']['protein_embedding_dim'], config['model']['hid_dim'],
                                 config['model']['n_layers'],
                                 config['model']['n_heads'], config['model']['ff_dim'], config['model']['dropout'],
                                 config['model']['activation_function'], device)
    inter_encoder = InterEncoder(config['model']['protein_embedding_dim'], config['model']['hid_dim'],
                                 config['model']['n_layers'],
                                 config['model']['n_heads'], config['model']['ff_dim'], config['model']['dropout'],
                                 config['model']['activation_function'], device)
    gp_layer = VanillaRFFLayer(in_features=config['model']['hid_dim'], RFFs=config['model']['gp_layer']['rffs'],
                               out_targets=config['model']['gp_layer']['out_targets'],
                               gp_cov_momentum=config['model']['gp_layer']['gp_cov_momentum'],
                               gp_ridge_penalty=config['model']['gp_layer']['gp_ridge_penalty'],
                               likelihood=config['model']['gp_layer']['likelihood_function'],
                               random_seed=config['other']['random_seed'])
    model = ProteinInteractionNet(intra_encoder, inter_encoder, gp_layer, device)
    model.load_state_dict(torch.load(config['directories']['model_output'], map_location=device))
    model.eval()
    model.to(device)
    return model