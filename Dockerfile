FROM pypy:3.10-7.3.12
# pythonの場合
# FROM python:3.11
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt update && apt install -y \
    sudo make vim zsh git neovim gdb gfortran libopenblas-dev liblapack-dev

RUN git clone https://github.com/atcoder/ac-library.git /lib/ac-library
ENV CPLUS_INCLUDE_PATH /lib/ac-library

COPY ./requirements.txt /workspace/
COPY ./ac-library-python/ /workspace/ac-library-python/
RUN pypy -m pip install -U pip && pip install -r requirements.txt
RUN pypy -m pip install -e ac-library-python
# pythonの場合
# RUN pip install -U pip && pip install -r requirements.txt
# RUN pip install -e ac-library-python
