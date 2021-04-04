#!/bin/sh

DOCKER_NAME=jbrugues/m6:mxnet
DOCKERFILE_NAME=Dockerfile_nvidia_mxnet.gpu

docker build -t $DOCKER_NAME -f $DOCKERFILE_NAME . --no-cache
