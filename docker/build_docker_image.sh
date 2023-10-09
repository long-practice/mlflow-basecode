#!/bin/bash

echo "Build Base docker image"
docker build --rm -f ./docker/Dockerfile -t mlflow-practice-base .