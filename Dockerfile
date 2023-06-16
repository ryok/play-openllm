FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN apt-get update && apt-get upgrade -y

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /workspace