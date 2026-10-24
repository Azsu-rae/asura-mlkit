
#include "c/MNIST.h"
#include "c/MLP.h"

#define TEST_IMAGES_PATH "./dataset/t10k-images.idx3-ubyte"
#define TEST_LABELS_PATH "./dataset/t10k-labels.idx1-ubyte"

#define TRAIN_IMAGES_PATH "./dataset/train-images.idx3-ubyte"
#define TRAIN_LABELS_PATH "./dataset/train-labels.idx1-ubyte"

int main() {

    MNIST train_imgs = MNIST_fromIDXFile(TRAIN_IMAGES_PATH, TRAIN_IMGS);
    MNIST train_labels = MNIST_fromIDXFile(TRAIN_LABELS_PATH, TRAIN_LABELS);

    MNIST_desc(train_imgs);
    MNIST_desc(train_labels);

    MNIST_printSample(train_labels, 0);
    MNIST_printSample(train_imgs, 0);

    MLP mlp = MLP_CTOR(784, 16, 16, 10);

    return 0;
}
