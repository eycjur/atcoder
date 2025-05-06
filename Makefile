# ビルドを強制するため、make -Bで実行してください

target := main
SHELL := /bin/bash

all: run-cpp

${target}.o: ${target}.cpp
	g++ -std=gnu++20 -o ${target}.o ${target}.cpp -Wall -Wextra -I/opt/boost/gcc/include -L/opt/boost/gcc/lib

.PHONY: run-cpp
run-cpp: ${target}.o
	./${target}.o < input.txt

.PHONY: run-cpp-all
run-cpp-all: ${target}.o
	for f in in/*; \
		do ./${target}.o < "$$f"; \
	done

.PHONY: run-pypy
run-pypy:
	time uv run --python .pypy3.10 pypy main.py < input.txt

.PHONY: run-python
run-python:
	time uv run --python .python3.11.4 python main.py < input.txt

.PHONY: python
python:
	uv run --python .python3.11.4 python

# コンテナのビルド・起動
.PHONY: up
up:
	docker compose up -d --build

# コンテナを停止して削除
.PHONY: down
down:
	docker compose down --remove-orphans

# push
.PHONY: push
push:
	docker build -t eycjur/atcoder .
	docker push eycjur/atcoder

# pull
.PHONY: pull
pull:
	docker pull eycjur/atcoder
