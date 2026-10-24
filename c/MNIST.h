#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "Tensor.h"

typedef enum {
    TEST_IMGS,
    TEST_LABELS,
    TRAIN_IMGS,
    TRAIN_LABELS,
} MNIST_TYPE;

typedef struct {
    Tensor idxfile;
    MNIST_TYPE type;
} MNIST;


MNIST MNIST_fromIDXFile(const char* filepath, MNIST_TYPE type);
void MNIST_dtor(MNIST mnist);

void MNIST_desc(MNIST mnist);
void MNIST_printSample(MNIST mnist, size_t index);

#endif
