import os
PROJECT_PATH = os.getcwd()
CONFIG_PATH = os.path.join(PROJECT_PATH, 'protein_protein/results/xspecies/TUnA_seed47/config.yaml')
MODEL_PATH = os.path.join(PROJECT_PATH, 'results/xspecies/TUnA_seed47/output/model')

if __name__ == "__main__":
    envs = f'{PROJECT_PATH=}\n' \
           f'{MODEL_PATH=}\n' \
           f'{CONFIG_PATH=}\n'

    with open('.env', 'w') as f:
        f.write(envs)