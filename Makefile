

.PHONY: run

run:
	gcc main.c ./c/idxfile.c ./c/dsa/arrays.c ./c/mnist.c -o main && ./main
