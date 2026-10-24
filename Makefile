

.PHONY: run

run:
	gcc main.c ./c/Tensor.c ./c/MLP.c ./c/dsa/Data.c ./c/MNIST.c ./c/dsa/arrays.c -o main && ./main
