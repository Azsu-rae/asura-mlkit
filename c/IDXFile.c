
#include "tensor.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>


Tensor IDXFile_ctor(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "couldn't fopen: %s", filepath);
        exit(1);
    }

    // ------------ MAGIC NUMBER ------------

    // the first two bytes are always 0
    fseek(f, 2, SEEK_CUR);

    IDXFile idxfile;
    fread(&idxfile.dtype, sizeof(idxfile.dtype), 1, f);
    fread(&idxfile.rank, sizeof(idxfile.rank), 1, f);

    // ------------ SHAPE ------------

    size_t total = 1;
    idxfile.shape = malloc(sizeof(uint32_t)*idxfile.rank);
    for (int i=0;i<idxfile.rank;i++) {
        uint32_t unknown_name;
        fread(&unknown_name, sizeof(uint32_t), 1, f);
        idxfile.shape[i] = ((unknown_name << 24) & 0xFF000000) |
                           ((unknown_name << 8) & 0x00FF0000) |
                           ((unknown_name >> 24) & 0x000000FF) |
                           ((unknown_name >> 8) & 0x0000FF00);
        total *= idxfile.shape[i];
    }

    // ------------ DATA ------------

    idxfile.data = malloc(total);
    fread(idxfile.data, 1, total, f);

    return idxfile;
}

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
