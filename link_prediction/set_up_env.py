import os
PROJECT_PATH = os.getcwd()
MODEL_PATH = os.path.join(PROJECT_PATH, 'best_model.pkl')

if __name__ == "__main__":
    envs = f'{PROJECT_PATH=}\n' \
           f'{MODEL_PATH=}\n' \

    with open('.env', 'w') as f:
        f.write(envs)