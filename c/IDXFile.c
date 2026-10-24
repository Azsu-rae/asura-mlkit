
#include "tensor.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>


void IDXFile_dtor(IDXFile idxfile) {
    free(idxfile.shape);
    free(idxfile.data);
}

void IDXFile_print(IDXFile idxfile) {
    printf("\n\ndtype: 0x%" PRIX8, idxfile.dtype);
    printf("\nrank: %u", idxfile.rank);
    printf("\nshape: (");
    for (int i=0;i<idxfile.rank;i++) {
        printf(" %d ", idxfile.shape[i]);
    } printf(")");
}
