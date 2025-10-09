import os
PROJECT_PATH = os.getcwd()
ENTITY_JSON = os.path.join(PROJECT_PATH, 'data', 'entity_names_molecules_0.json')
EMB_H5 = os.path.join(PROJECT_PATH, 'data', 'embeddings_molecules_0.v48.h5')
MODEL_H5 = os.path.join(PROJECT_PATH, 'data', 'model.v48.h5')

if __name__ == "__main__":
    envs = f'{PROJECT_PATH=}\n' \
           f'{ENTITY_JSON=}\n' \
           f'{EMB_H5=}\n' \
           f'{MODEL_H5=}\n'

    with open('.env', 'w') as f:
        f.write(envs)