#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

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

Tensor from_IDXFile(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "couldn't fopen: %s", filepath);
        exit(1);
    }

    // ------------ MAGIC NUMBER ------------

    // the first two bytes are always 0
    fseek(f, 2, SEEK_CUR);

    Tensor idxfile;

    /*
    * 0x08: unsigned byte
    * 0x09: signed byte
    * 0x0B: short (2 bytes)
   * 0x0C: int (4 bytes)
    * 0x0D: float (4 bytes)
    * 0x0E: double (8 bytes)
    *
    * We can safely assume that it'll be a ubyte
    **/
    fseek(f, 1, SEEK_CUR);
    idxfile.type = UBYTE;

    fread(&idxfile.rank, sizeof(idxfile.rank), 1, f);

    // ------------ SHAPE ------------

    size_t total = 1;
    idxfile.shape = (size_t*) malloc(sizeof(size_t)*idxfile.rank);
    for (int i=0;i<idxfile.rank;i++) {
        size_t buffer;
        // the size of each dimension; 4-byte integers (big-endian, like in most non-Intel processors).
        fread(&buffer, sizeof(size_t), 1, f);
        idxfile.shape[i] = ((buffer << 24) & 0xFF000000) |
                           ((buffer << 8) & 0x00FF0000) |
                           ((buffer >> 24) & 0x000000FF) |
                           ((buffer >> 8) & 0x0000FF00);
        total *= idxfile.shape[i];
    }

    // ------------ DATA ------------

    idxfile.data = malloc(total);
    fread(idxfile.data, 1, total, f);

    return idxfile;

}

MNIST MNIST_ctor(const char* idxfile_path, MNIST_TYPE filetype);
void MNIST_dtor(MNIST mnist);

void MNIST_struct(MNIST mnist);
void MNIST_sample(MNIST mnist, size_t index);

#endif
