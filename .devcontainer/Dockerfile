FROM amd64/ubuntu:20.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y sudo make vim build-essential clang libssl-dev g++ gdb zsh git neovim

ARG ROOT_PASSWORD=pass
RUN echo "root:${ROOT_PASSWORD}" | chpasswd

ARG DOCKER_UID=1000
ARG DOCKER_USER=user
ARG DOCKER_PASSWORD=pass
RUN useradd -m --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} \
    && echo "${DOCKER_USER}:${DOCKER_PASSWORD}" | chpasswd

USER ${DOCKER_USER}

CMD ["/bin/zsh"]
