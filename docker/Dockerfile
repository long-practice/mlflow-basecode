FROM ghcr.io/mlflow/mlflow:v2.7.1

WORKDIR /work/

ADD ./ ./
CMD chmod +x ./run_main.sh

RUN apt-get update && \
    apt-get install -y \
    vim \
    build-essential \
    python3-pip \
    python3-dev && \
    pip3 install --upgrade pip && \
    rm -rf ./venv/
