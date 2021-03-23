#!/bin/sh

DOCKER_NAME=jbrugues/m6:tfmodels
DOCKERFILE_NAME=Dockerfile_nvidia_tf_2.4.1.gpu

docker build -t $DOCKER_NAME -f $DOCKERFILE_NAME . --no-cache
