import os
from dotenv import load_dotenv

load_dotenv()

ENTITY_JSON = os.getenv('ENTITY_JSON')
EMB_H5 = os.getenv('EMB_H5')
MODEL_H5 = os.getenv('MODEL_H5')
DIM = 400
TOPK = 100
COMPARATOR_TYPE = "cos"