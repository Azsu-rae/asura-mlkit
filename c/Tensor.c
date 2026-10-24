
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "./dsa/Data.h"
#include "./dsa/arrays.h"

#include "Tensor.h"

void Tensor_dtor(Tensor tensor) {
    free(tensor.shape);
    free(tensor.data);
}


void Tensor_desc(Tensor tensor) {
    printf("\ntype: "); DType_desc(tensor.type);
    printf("\nrank: %lu", tensor.rank);
    arrays_desc("shape: ", tensor.shape, tensor.rank, SIZE_T);
}
