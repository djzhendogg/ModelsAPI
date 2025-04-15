import os
from dotenv import load_dotenv

load_dotenv()


PROJECT_PATH = os.getenv('PROJECT_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')
CONFIG_PATH = "/protein_protein_model/results/xspecies/TUnA_seed47/config.yaml"