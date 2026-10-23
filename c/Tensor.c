
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "./dsa/Data.h"

#include "Tensor.h"

Tensor Tensor_ctor(DType type, size_t layer, ...) {
    va_list args;
    va_start(args, layer);

    size_t buffer[RANK_LIMIT], rank=0;
    while (layer != -1 && rank < RANK_LIMIT) {
        buffer[rank++] = layer;
        layer = va_arg(args, size_t);
    }

    va_end(args);

    // persist in heap memory
    size_t* shape = (size_t*) malloc(sizeof(size_t)*rank);
    for (int i=0;i<rank;i++) {
        shape[i] = buffer[i];
    }

    size_t data_size;
    for (int i=0;i<rank;i++) {
        data_size *= shape[i];
    }

    return (Tensor) {
        .type = type,
        .rank = rank,
        .shape = shape,
        .data = malloc(DType_sizeof(type)*data_size),
    };
}
