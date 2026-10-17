#ifndef MNIST_H
#define MNIST_H

#include "idxfile.h"

typedef enum {
    TEST_IMGS,
    TEST_LABELS,
    TRAIN_IMGS,
    TRAIN_LABELS,
} MNIST_TYPE;

typedef struct {
    IDXFile idxfile;
    MNIST_TYPE type;
} MNIST;

MNIST MNIST_ctor(const char* idxfile_path, MNIST_TYPE filetype);
void MNIST_dtor(MNIST mnist);

void MNIST_struct(MNIST mnist);
void MNIST_sample(MNIST mnist, size_t index);

#endif
