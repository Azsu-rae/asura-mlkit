#ifndef TENSOR_H
#define TENSOR_H

#include "./dsa/Data.h"

#define RANK_LIMIT 20

typedef struct {
    DType type;
    // number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
    size_t rank;
    size_t* shape;
    // The data is expected to be stored like in a C array, i.e. the index in
    // the last dimension changes the fastest.
    void* data;
} Tensor;


void Tensor_dtor(Tensor tensor);
void Tensor_desc(Tensor tensor);

#endif
