FROM ubuntu:20.04
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y sudo make vim build-essential clang libssl-dev g++ gdb zsh git neovim python3.8 python3-pip pypy3

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 30 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 30 && \
    update-alternatives --install /usr/bin/pypy pypy /usr/bin/pypy3 30

RUN git clone https://github.com/atcoder/ac-library.git /lib/ac-library
ENV CPLUS_INCLUDE_PATH /lib/ac-library

RUN pip install -U pip && \
    pip install numpy scipy scikit-learn numba

RUN git clone https://github.com/eycjur/dotfiles.git ~/dotfiles
RUN ~/dotfiles/install.sh

RUN chsh -s /bin/zsh
