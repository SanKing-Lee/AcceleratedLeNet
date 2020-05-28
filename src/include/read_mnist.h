#ifndef _READ_MNIST_H_
#define _READ_MNIST_H_

#include "types.h"

// #include <vector>
#define TEST_IMG_SET_PATH "./mnist/t10k-images-idx3-ubyte"
#define TEST_LABEL_SET_PATH "./mnist/t10k-labels-idx1-ubyte"

void read_mnist_image(float *images, float minScale, float maxScale, int imageNum);
void read_mnist_label(int *labels, int labelNum);
void read_mnist_test();

#endif