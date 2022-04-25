# syntax=docker/dockerfile:1

# Base image: TF 2.7.0 GPU
FROM tensorflow/tensorflow:2.4.0-gpu

# Working directory (in the container, not in host)
WORKDIR /home/project

# Requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y python3-opencv
RUN pip install opencv-python
