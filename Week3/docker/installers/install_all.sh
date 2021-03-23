#!/usr/bin/env bash

# Fail on first error.
set -e

echo "Preparing package installation..."
apt-get update -y && \
    apt-get install -y \
    apt-transport-https \
    build-essential \
    software-properties-common \
    bc \
    sudo \
    cmake \
    libgl1-mesa-dev \
    libgtk2.0-dev \
    make \
    g++ \
    cppcheck \
    unzip \
    zip \
    wget \
    locate \
    git \
    nano \
    libpcap-dev \
    autoconf \
    automake \
    libtool \
    curl \
    python3-dev \
    python3-pip \
    screen
    
pip3 install --upgrade pip
pip3 install numpy
pip3 install scipy
pip3 install opencv-python
pip3 install opencv-contrib-python
pip3 install mypy
pip3 install easydict
pip3 install matplotlib
pip3 install scikit-image
pip3 install tqdm
pip3 install psutil
