FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir /antibody_antigen_model
WORKDIR /antibody_antigen_model

RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
RUN python3 /antibody_antigen_model/set_up.py
