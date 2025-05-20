# Базовый образ с поддержкой Conda и CUDA
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Установка базовых утилит
RUN apt-get update && apt-get install -y \
    wget git curl bzip2 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка Miniconda
ENV CONDA_DIR=/opt/conda
RUN curl -sLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Установка mamba для более быстрой установки окружений
RUN conda install -n base -c conda-forge mamba -y

# Копируем файл окружения в контейнер
COPY RF2na-linux.yml /app/RF2na-linux.yml

# Создание Conda окружения
RUN mamba env create -f /app/RF2na-linux.yml

# Активация окружения
SHELL ["conda", "run", "-n", "RF2NA", "/bin/bash", "-c"]

# Копируем код проекта (предполагаем, что SE3Transformer находится в той же директории)
COPY . /app
WORKDIR /app

# Установка SE3Transformer
RUN cd SE3Transformer && \
    pip install --no-cache-dir -r requirements.txt && \
    python setup.py install

# Загрузка весов
RUN cd network && \
    wget https://files.ipd.uw.edu/dimaio/RF2NA_apr23.tgz && \
    tar xvfz RF2NA_apr23.tgz

# По умолчанию запуск интерактивной оболочки (можно заменить на запуск скрипта)
CMD ["conda", "run", "-n", "RF2NA", "bash"]
