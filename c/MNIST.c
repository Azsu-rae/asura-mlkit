#include "MNIST.h"
#include "Tensor.h"
#include "dsa/Data.h"

#include <stdio.h>

void MNIST_dtor(MNIST mnist) {
    Tensor_dtor(mnist.idxfile);
}

void MNIST_desc(MNIST mnist) {
    switch (mnist.type) {
        case TEST_IMGS:
            printf("\n\n------------ TEST_IMGS ------------");
            break;
        case TRAIN_IMGS:
            printf("\n\n------------ TRAIN_IMGS ------------");
            break;
        case TEST_LABELS:
            printf("\n\n------------ TEST_LABELS ------------");
            break;
        case TRAIN_LABELS:
            printf("\n\n------------ TRAIN_LABELS ------------");
            break;
    }
    Tensor_desc(mnist.idxfile);
}

void MNIST_printSample(MNIST mnist, size_t index) {
    if (mnist.type == TEST_IMGS || mnist.type == TRAIN_IMGS) {
        printf("\n\n");
        const char* shades = " .:-=+*#%@";
        for (int i=0;i<28;i++) {
            for (int j=0;j<28;j++) {
                // divide by 26 to get roughly 10 range of values
                uint8_t* data = mnist.idxfile.data;
                printf("%c ", shades[data[784*index +  28*i + j] / 26]);
            } printf("\n");
        } printf("\n");
    } else {
        printf("\n\nLabel: %u", ((uint8_t*)mnist.idxfile.data)[index]);
    }
}

/*
 * https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
 ***/
MNIST MNIST_fromIDXFile(const char* filepath, MNIST_TYPE type) {

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
    * We can safely assume that the content's type is ubytes
    **/
    fseek(f, 1, SEEK_CUR);
    idxfile.type = UBYTE;

    // the number of dimensions, aka the 'rank'
    uint8_t rank_buffer;
    fread(&rank_buffer, 1, 1, f);
    idxfile.rank = rank_buffer;

    // ------------ SHAPE ------------

    size_t total = 1;
    idxfile.shape = malloc(sizeof(size_t)*idxfile.rank);
    for (int i=0;i<idxfile.rank;i++) {
        uint32_t buffer;
        // the size of each dimension; 4-byte integers (big-endian, like in most non-Intel processors).
        fread(&buffer, sizeof(uint32_t), 1, f);
        idxfile.shape[i] = ((buffer << 24) & 0xFF000000) |
                           ((buffer << 8) & 0x00FF0000) |
                           ((buffer >> 24) & 0x000000FF) |
                           ((buffer >> 8) & 0x0000FF00);
        total *= idxfile.shape[i];
    }

    // ------------ DATA ------------

    idxfile.data = malloc(total);
    fread(idxfile.data, 1, total, f);

    return (MNIST) {
        .idxfile = idxfile,
        .type = type,
    };
}
