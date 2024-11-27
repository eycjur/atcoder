FROM python:3.11
WORKDIR /workspace

RUN apt update && apt install -y sudo make vim zsh git neovim gdb

RUN git clone https://github.com/atcoder/ac-library.git /lib/ac-library
ENV CPLUS_INCLUDE_PATH /lib/ac-library

COPY ./requirements.txt /workspace/
RUN pip install -U pip && pip install -r requirements.txt
