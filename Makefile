# ビルドを強制するため、make -Bで実行してください

target := main
SHELL := bash
.SHELLFLAGS := -euo pipefail -c
.ONESHELL:
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

all: run-cpp

${target}.o: ${target}.cpp
	g++ -std=gnu++20 -o ${target}.o ${target}.cpp -Wall -Wextra -I/opt/boost/gcc/include -L/opt/boost/gcc/lib

.PHONY: run-cpp
run-cpp: ${target}.o
	./${target}.o < input.txt

.PHONY: run-cpp-all
run-cpp-all: ${target}.o
	mkdir -p out
	for f in in/*; do \
		echo "$$f"; \
		./${target}.o < "$$f" > out/$$(basename "$$f"); \
	done

.PHONY: run-pypy
run-pypy:
	time uv run --python .pypy3.10 pypy main.py < input.txt

.PHONY: run-pypy-all
run-pypy-all:
	mkdir -p out
	for f in in/*; do \
		echo "$$f"; \
		uv run --python .pypy3.10 pypy main.py  < "$$f" > out/$$(basename "$$f"); \
	done

.PHONY: run-python
run-python:
	time uv run --python .python3.11.4 python main.py < input.txt

.PHONY: run-python-all
run-python-all:
	mkdir -p out
	for f in in/*; do \
		echo "$$f"; \
		uv run --python .python3.11.4 python main.py  < "$$f" > out/$$(basename "$$f"); \
	done

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
