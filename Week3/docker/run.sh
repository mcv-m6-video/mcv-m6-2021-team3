#!/bin/sh

if [ "$1" = "" ]
then
    DOCKER_NAME=jbrugues/m6:tfmodels
else
    DOCKER_NAME=jbrugues/m6:$1
fi

XSOCK=/tmp/.X11-unix
XAUTH=/home/$USER/.Xauthority
SHARED_DIR=/home/josep/shared_dir
HOST_DIR=/home/$USER/Documents/Git/mcv-m6-2021-team3

mkdir -p $HOST_DIR
echo "Shared directory:" ${HOST_DIR}

docker run \
    -it --rm \
    --volume=$HOST_DIR:$SHARED_DIR:rw \
    --volume=$XSOCK:$XSOCK:rw \
    --volume=$XAUTH:$XAUTH:rw \
    --env="XAUTHORITY=${XAUTH}" \
    --env="DISPLAY=${DISPLAY}" \
    --env="QT_X11_NO_MITSHM=1" \
    -u josep \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /dev/video0:/dev/video0 \
    --net=host \
    --runtime=nvidia \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus all \
    $DOCKER_NAME \
    bash

