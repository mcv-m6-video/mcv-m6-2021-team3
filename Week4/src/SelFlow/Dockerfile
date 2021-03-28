FROM nvcr.io/nvidia/tensorflow:21.03-tf1-py3
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl-dev

COPY ./requirements.txt /
RUN pip3 install -r /requirements.txt
COPY . /SelFlow
