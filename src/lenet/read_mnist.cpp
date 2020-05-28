#include "read_mnist.h"
#include "constants.h"

#include <iostream>
#include <fstream>
#include <string>
using std::ifstream;
using std::ofstream;
using std::ios;
using std::string;
using std::cout;
using std::endl;

int reverse_integer(int i) {
    unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_labels(string filename, int *labels, int labelNum)
{
	cout << "Reading MNIST label..." << endl;
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = reverse_integer(magic_number);
		number_of_images = reverse_integer(number_of_images);
		// cout << "magic number = " << magic_number << endl;
		// cout << "number of images = " << number_of_images << endl;
		int batch = number_of_images > labelNum ? labelNum : number_of_images;
		for (int i = 0; i < batch; i++)
		{
			unsigned char label = 0;
			// int index = 1;
			// for (int j = 0; j < index; j++) {
			// 	file.read((char*)&label, sizeof(label));
			// }
			file.read((char*)&label, sizeof(label));
			labels[i] = label;
		}
		cout << "Reading MNIST label succeed!" << endl;
	}
	else {
		cout << "Reading MNIST label failed!" << endl;
	}
    file.clear();
    file.close();
}

void read_images(string filename, float *images, float minScale, float maxScale, int imageNum)
{
	cout << "Reading MNIST images..." << endl;
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = reverse_integer(magic_number);
		number_of_images = reverse_integer(number_of_images);
		n_rows = reverse_integer(n_rows);
		n_cols = reverse_integer(n_cols);
		int batch = number_of_images > imageNum ? imageNum : number_of_images;
		for (int i = 0; i < batch; i++)
		{
            // cout << "4" << endl;
			// skip the front image
			// int index = 1;
			// for (int j = 0; j < index*n_rows*n_cols; i++) {
			// 	float pixel;
			// 	file.read((char*)&pixel, sizeof(pixel));
			// }
			for (int r = 0; r < MNIST_PAD_WH; r++)
			{
				for (int c = 0; c < MNIST_PAD_WH; c++)
				{	
					float pixel;
					if (c < 2 || c >= 30 || r < 2 || r >= 30) {
						pixel = minScale;
						continue;
					}
					else {
						unsigned char _pixel = 0;
						file.read((char*)&_pixel, sizeof(_pixel));
						pixel = ((float)_pixel/255.0) *(maxScale-minScale) + minScale;
					}
					images[i*MNIST_PAD_SIZE + r*MNIST_PAD_WH + c] = pixel;
				}
			}
		}
		cout << "Reading MNIST images succeed!" << endl;
	}
	else {
		cout << "Reading MNIST images failed!" << endl;
	}
    file.clear();
    file.close();
}

void read_mnist_image(float *images, float minScale, float maxScale, int imageNum) {
	read_images(TEST_IMG_SET_PATH, images, minScale, maxScale, imageNum);
}

void read_mnist_label(int *labels, int labelNum){
	read_labels(TEST_LABEL_SET_PATH, labels, labelNum);
}

void read_mnist_test() {
	int imageNum = 5;
	int imageSize = imageNum*MNIST_PAD_SIZE*sizeof(float);
	int labelSize = imageNum*sizeof(int);
	float *images = (float*)malloc(imageSize);
	int *labels = (int*)malloc(labelSize);
	
	read_mnist_image(images, FLOAT_MIN_SCALE, FLOAT_MAX_SCALE, imageNum);
	read_mnist_label(labels, imageNum);

	// for (int i = 0; i < imageNum; i++) {
	// 	cout << labels[i] << endl;
	// 	for (int r = 0; r < MNIST_WH; r++){
	// 		for (int c = 0; c < MNIST_WH; c++) {
	// 			cout.width(10);
	// 			cout << images[r*MNIST_WH+c];
	// 		}
	// 		cout << endl;
	// 	}
	// }

	free(images);
	free(labels);
}