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
