
#include "c/mnist.h"

#define TEST_IMAGES_PATH "./dataset/t10k-images.idx3-ubyte"
#define TEST_LABELS_PATH "./dataset/t10k-labels.idx1-ubyte"

#define TRAIN_IMAGES_PATH "./dataset/train-images.idx3-ubyte"
#define TRAIN_LABELS_PATH "./dataset/train-labels.idx1-ubyte"

int main() {

    MNIST train_imgs = MNIST_ctor(TRAIN_IMAGES_PATH, TRAIN_IMGS);
    MNIST train_labels = MNIST_ctor(TRAIN_LABELS_PATH, TRAIN_LABELS);

    MNIST_struct(train_imgs);
    MNIST_struct(train_labels);

    MNIST_sample(train_labels, 0);
    MNIST_sample(train_imgs, 0);

    MNIST_dtor(train_imgs);
    MNIST_dtor(train_labels);

    return 0;
}
