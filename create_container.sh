#!bin/bash

TARGET_DOCKER_IMAGE=mlflow-practice-base
CONTAINER_NAME="model-training"
echo $CONTAINER_NAME

ARTIFACT_DIR=/Users/llong/mlflow-basecode/artifact/
LOG_DIR=/Users/llong/mlflow-basecode/logs/
DATA_DIR=/Users/llong/mlflow-basecode/data/
PREDICTION_DIR=/Users/llong/mlflow-basecode/prediction/

docker run --name $CONTAINER_NAME   \
    -v $ARTIFACT_DIR:/work/artifact/   \
    -v $LOG_DIR:/work/logs/   \
    -v $DATA_DIR:/work/data/   \
    -v $PREDICTION_DIR:/work/prediction/   \
    -p 5000:5000 \
    --rm -it $TARGET_DOCKER_IMAGE:latest /bin/bash
