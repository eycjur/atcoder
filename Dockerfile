FROM ubuntu:22.04

COPY --from=ghcr.io/astral-sh/uv:0.7.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        sudo make vim zsh git neovim gdb curl locales \
        build-essential gfortran libopenblas-dev \
        pkg-config g++-12 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    sudo ln -s /usr/bin/g++-12 /usr/local/bin/g++

RUN echo "ja_JP UTF-8" > /etc/locale.gen && \
    locale-gen ja_JP.UTF-8
ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8
ENV TZ=Asia/Tokyo

WORKDIR /workspace

RUN uv venv --python 3.11.4 .python3.11.4
RUN uv venv --python pypy@3.10 .pypy3.10

RUN git clone https://github.com/atcoder/ac-library.git /lib/ac-library
ENV CPLUS_INCLUDE_PATH=/lib/ac-library

COPY ./requirements_pypy.txt /requirements_python.txt /workspace/
COPY ./ac-library-python/ /workspace/ac-library-python/

RUN uv pip install -r requirements_python.txt --python .python3.11.4 && \
    uv pip install -e ac-library-python --python .python3.11.4 --link-mode=copy && \
    uv pip install -r requirements_pypy.txt --python .pypy3.10 && \
    uv pip install -e ac-library-python --python .pypy3.10 --link-mode=copy
