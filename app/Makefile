
target := main

all: exec

${target}.o: ${target}.cpp
	g++ -o ${target}.o ${target}.cpp

exec: ${target}.o
	./${target}.o
