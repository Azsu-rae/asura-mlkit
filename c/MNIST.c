#include "mnist.h"
#include "tensor.h"

#include <stdio.h>

MNIST MNIST_ctor(const char* idxfile_path, MNIST_TYPE filetype) {
    return (MNIST) {
        .idxfile = MNIST_from_IDXFile(idxfile_path),
        .type = filetype,
    };
}

void MNIST_dtor(MNIST mnist) {
    Tensor_dtor(mnist.idxfile);
}

void MNIST_struct(MNIST mnist) {
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
    Tensor_print(mnist.idxfile);
}

void MNIST_sample(MNIST mnist, size_t index) {
    if (mnist.type == TEST_IMGS || mnist.type == TRAIN_IMGS) {
        printf("\n\n");
        const char* shades = " .:-=+*#%@";
        for (int i=0;i<28;i++) {
            for (int j=0;j<28;j++) {
                // divide by 26 to get roughly 10 range of values
                printf("%c ", shades[mnist.idxfile.data[784*index +  28*i + j ] / 26]);
            } printf("\n");
        } printf("\n");
    } else {
        printf("\n\nLabel: %u", mnist.idxfile.data[index]);
    }
}

