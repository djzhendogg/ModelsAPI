FROM ubuntu:22.04

# Установка базовых пакетов
RUN apt-get update && apt-get install -y \
    wget git curl bzip2 build-essential \
    ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Установка Miniconda
ENV CONDA_DIR=/opt/conda
RUN curl -sLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Установка mamba
RUN conda install -n base -c conda-forge mamba -y

# Копируем файл окружения
COPY RF2na-linux.yml /app/RF2na-linux.yml

# Создаём окружение
RUN mamba env create -f /app/RF2na-linux.yml

# Копируем проект в контейнер
COPY . /app
WORKDIR /app

RUN conda run -n RF2NA pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Установка зависимостей SE3Transformer
SHELL ["conda", "run", "-n", "RF2NA", "/bin/bash", "-c"]
RUN cd SE3Transformer && \
    pip install --no-cache-dir -r requirements.txt && \
    python setup.py install

# Установка FastAPI-зависимостей
RUN pip install \
    uvicorn==0.30.1 \
    fastapi==0.112.0 \
    fastapi-cli \
    flatbuffers \
    slowapi

RUN pip install --force-reinstall -v "numpy==1.25.2"
RUN pip install dgl==0.9.1 --no-cache-dir -f https://data.dgl.ai/wheels/cu102/repo.html
# Загрузка весов
RUN cd network && \
    wget https://files.ipd.uw.edu/dimaio/RF2NA_apr23.tgz && \
    tar xvfz RF2NA_apr23.tgz
