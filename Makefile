target := main

all: run-cpp

${target}.o: ${target}.cpp
	g++ -o ${target}.o ${target}.cpp

.PHONY: run-cpp
run-cpp: ${target}.o
	./${target}.o < input.txt


## dockerコマンド
# 確認
images:
	docker images
ps:
	docker ps -a
volume:
	docker volume ls
logs:
	docker compose logs
