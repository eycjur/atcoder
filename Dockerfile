FROM ubuntu:22.04

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND noninteractive

RUN apt update && \
	apt install -y \
		sudo make vim zsh git neovim gdb bzip2 curl \
		build-essential gfortran g++ libopenblas-dev liblapack-dev \
		software-properties-common pkg-config

WORKDIR /opt
# pypy3のインストール
ARG PYPY_VERSION=pypy3.10-v7.3.12
RUN curl -O https://downloads.python.org/pypy/${PYPY_VERSION}-aarch64.tar.bz2 && \
		tar xjf ${PYPY_VERSION}-aarch64.tar.bz2 && \
		rm ${PYPY_VERSION}-aarch64.tar.bz2 && \
		ln -s /opt/${PYPY_VERSION}-aarch64/bin/pypy3 /usr/local/bin/pypy3 && \
		ln -s /opt/${PYPY_VERSION}-aarch64/bin/pypy3 /usr/local/bin/pypy

# python3.11のインストール
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
		apt update && \
		apt install -y python3.11 python3-pip && \
		ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
		ln -sf /usr/bin/python3.11 /usr/local/bin/python


WORKDIR /workspace
RUN git clone https://github.com/atcoder/ac-library.git /lib/ac-library
ENV CPLUS_INCLUDE_PATH /lib/ac-library

COPY ./requirements_pypy.txt /requirements_python.txt /workspace/
COPY ./ac-library-python/ /workspace/ac-library-python/

# pypyのライブラリインストール
RUN pypy3 -m ensurepip && \
		pypy3 -m pip install -U pip wheel && \
		pypy3 -m pip install -r requirements_pypy.txt && \
		pypy3 -m pip install -e ac-library-python

# pythonのライブラリインストール
RUN python3 -m pip install -U pip && \
		python3 -m pip install -r requirements_python.txt && \
		python3 -m pip install -e ac-library-python
