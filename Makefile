target := main

all: run-cpp

${target}.o: ${target}.cpp
	g++ -o ${target}.o ${target}.cpp

.PHONY: run-cpp
run-cpp: ${target}.o
	./${target}.o < input.txt


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
