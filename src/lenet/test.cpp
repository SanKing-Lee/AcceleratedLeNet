#include "test.h"
#include "lenet.h"
#include "constants.h"
#include "read_mnist.h"
#include "load_para.h"
#include "types.h"
#ifdef SDS
#include "sds_lib.h"
#endif

// #define WRITE_DEBUG

#include <iostream>
#include <fstream>
#include <ctime>
using namespace std;
ofstream result_file;

int max_out(float* arr, int size=10) {
	int max_arg = 0;
	float max = 0;
	for (int i = 0; i < size; i++) {
		if (arr[i] > max) {
			max_arg = i;
			max = arr[i];
		}
	}
	return max_arg;
}

double test_sw(int imageNum) {
    cout << "Starting software testing..." << endl;
    int imgNum = (!imageNum)?IMG_NUM:imageNum;

    float *wConv1 = (float*)malloc(CONV_1_WEIGHT_SIZE);
    float *bConv1 = (float*)malloc(CONV_1_BIAS_SIZE);
    float *wPool2 = (float*)malloc(POOL_2_WEIGHT_SIZE);
    float *bPool2 = (float*)malloc(POOL_2_BIAS_SIZE);
    float *wConv3 = (float*)malloc(CONV_3_WEIGHT_SIZE);
    float *bConv3 = (float*)malloc(CONV_3_BIAS_SIZE);
    float *wPool4 = (float*)malloc(POOL_4_WEIGHT_SIZE);
    float *bPool4 = (float*)malloc(POOL_4_BIAS_SIZE);
    float *wFC5 = (float*)malloc(FC_5_WEIGHT_SIZE);
    float *bFC5 = (float*)malloc(FC_5_BIAS_SIZE);
    float *wFC6 = (float*)malloc(FC_6_WEIGHT_SIZE);
    float *bFC6 = (float*)malloc(FC_6_BIAS_SIZE);
    float *wFC7 = (float*)malloc(FC_7_WEIGHT_SIZE);
    float *bFC7 = (float*)malloc(FC_7_BIAS_SIZE);

    load_conv_1_para(wConv1, bConv1);
    load_pool_2_para(wPool2, bPool2);
    load_conv_3_para(wConv3, bConv3);
    load_pool_4_para(wPool4, bPool4);
    load_fc_5_para(wFC5, bFC5);
    load_fc_6_para(wFC6, bFC6);
    load_fc_7_para(wFC7, bFC7);

    int inputSize = MNIST_PAD_SIZE*sizeof(DTYPE);
    int conv1_outputSize = CONV_1_OUTPUT_SIZE*CONV_1_CHANNEL_NUM*sizeof(DTYPE);
    int pool2_outputSize = POOL_2_OUTPUT_SIZE*POOL_2_CHANNEL_NUM*sizeof(DTYPE);
    int conv3_outputSize = CONV_3_OUTPUT_SIZE*CONV_3_CHANNEL_NUM*sizeof(DTYPE);
    int pool4_outputSize = POOL_4_OUTPUT_SIZE*POOL_4_CHANNEL_NUM*sizeof(DTYPE);
    int fc5_outputSize = FC_5_OUTPUT_NEURON_NUM*sizeof(DTYPE);
    int fc6_outputSize = FC_6_OUTPUT_NEURON_NUM*sizeof(DTYPE);
    int fc7_outputSize = FC_7_OUTPUT_NEURON_NUM*sizeof(DTYPE);

    float *images = (float*)malloc(imgNum*inputSize);
    int *labels = (int*)malloc(imgNum*sizeof(int));
    float *input = (float*)malloc(inputSize);
    DTYPE *conv1_output = (DTYPE*)malloc(conv1_outputSize);
    DTYPE *pool2_output = (DTYPE*)malloc(pool2_outputSize);
    DTYPE *conv3_output = (DTYPE*)malloc(conv3_outputSize);
    DTYPE *pool4_output = (DTYPE*)malloc(pool4_outputSize);
    DTYPE *fc5_output = (DTYPE*)malloc(fc5_outputSize);
    DTYPE *fc6_output = (DTYPE*)malloc(fc6_outputSize);
    DTYPE *fc7_output = (DTYPE*)malloc(fc7_outputSize);

    read_mnist_image(images, -1.0f, 1.0f, imgNum);
    read_mnist_label(labels, imgNum);

    int result_sw = 0;
    clock_t start = clock();
    double sw_conv1_time = 0;
    double sw_conv3_time = 0;
    for (int i = 0; i < imgNum; i++) {
        for (int j = 0; j < MNIST_PAD_SIZE; j++) 
        {
            input[j] = images[i*MNIST_PAD_SIZE+j];
        }
        if (!((i+1)%100))
        cout << "Processing " << i+1 << "/" << imgNum << " image..." << endl;

        clock_t sw_conv1_start = clock();
        convolve_layer_1_sw(input, wConv1, bConv1, conv1_output);
        clock_t sw_conv1_end = clock();
        sw_conv1_time += (double)(sw_conv1_end-sw_conv1_start)/CLOCKS_PER_SEC;

        pooling_layer_2_sw(conv1_output, wPool2, bPool2, pool2_output);

        clock_t sw_conv3_start = clock();
        convolve_layer_3_sw(pool2_output, wConv3, bConv3, conv3_output);
        clock_t sw_conv3_end = clock();
        sw_conv3_time += (double)(sw_conv3_end-sw_conv3_start)/CLOCKS_PER_SEC;

        pooling_layer_4_sw(conv3_output, wPool4, bPool4, pool4_output);
        
        fully_connected_layer_5_sw(pool4_output, wFC5, bFC5, fc5_output);

        fully_connected_layer_6_sw(fc5_output, wFC6, bFC6, fc6_output);
        
        fully_connected_layer_7_sw(fc6_output, wFC7, bFC7, fc7_output);
        
        if(labels[i] == max_out(fc7_output)) {
            result_sw++;
        }
        
    }
    clock_t end = clock();
    double time = (double)(end-start)/CLOCKS_PER_SEC;
    double accuracy = (double)result_sw/imgNum;
    cout << "Software test accuracy = " << accuracy*100 << "%" << endl;
    cout << "总共耗时：" << time << "s" << endl;
    cout << "卷积层1耗时：" << sw_conv1_time << "s" << endl;
    cout << "卷积层3耗时：" << sw_conv3_time << "s" << endl;
    result_file << time << "\t" << sw_conv1_time << "\t" << sw_conv3_time << "\t" << accuracy << "\t";

    free(wConv1);
    free(bConv1);
    free(wPool2);
    free(bPool2);
    free(wConv3);
    free(bConv3);
    free(wPool4);
    free(bPool4);
    free(wFC5);
    free(bFC5);
    free(wFC6);
    free(bFC6);
    free(wFC7);
    free(bFC7);
    
    free(images);
    free(input);
    free(labels);
    free(conv1_output);
    free(pool2_output);
    free(conv3_output);
    free(pool4_output);
    free(fc5_output);
    free(fc6_output);
    free(fc7_output);

    return time;
}

double test_hw(int imageNum) {
    cout << "Starting hardware testing..." << endl;
    int imgNum = (!imageNum)?IMG_NUM:imageNum;

    #ifndef SDS
    float *wConv1 = (float*)malloc(CONV_1_WEIGHT_SIZE);
    float *bConv1 = (float*)malloc(CONV_1_BIAS_SIZE);
    float *wConv3 = (float*)malloc(CONV_3_WEIGHT_SIZE);
    float *bConv3 = (float*)malloc(CONV_3_BIAS_SIZE);
    #else
    float *wConv1 = (float*)sds_alloc(CONV_1_WEIGHT_SIZE);
    float *bConv1 = (float*)sds_alloc(CONV_1_BIAS_SIZE);
    float *wConv3 = (float*)sds_alloc(CONV_3_WEIGHT_SIZE);
    float *bConv3 = (float*)sds_alloc(CONV_3_BIAS_SIZE);
    #endif
    float *wPool2 = (float*)malloc(POOL_2_WEIGHT_SIZE);
    float *bPool2 = (float*)malloc(POOL_2_BIAS_SIZE);
    float *wPool4 = (float*)malloc(POOL_4_WEIGHT_SIZE);
    float *bPool4 = (float*)malloc(POOL_4_BIAS_SIZE);
    float *wFC5 = (float*)malloc(FC_5_WEIGHT_SIZE);
    float *bFC5 = (float*)malloc(FC_5_BIAS_SIZE);
    float *wFC6 = (float*)malloc(FC_6_WEIGHT_SIZE);
    float *bFC6 = (float*)malloc(FC_6_BIAS_SIZE);
    float *wFC7 = (float*)malloc(FC_7_WEIGHT_SIZE);
    float *bFC7 = (float*)malloc(FC_7_BIAS_SIZE);

    load_conv_1_para(wConv1, bConv1);
    load_pool_2_para(wPool2, bPool2);
    load_conv_3_para(wConv3, bConv3);
    load_pool_4_para(wPool4, bPool4);
    load_fc_5_para(wFC5, bFC5);
    load_fc_6_para(wFC6, bFC6);
    load_fc_7_para(wFC7, bFC7);

    int inputSize = MNIST_PAD_SIZE*sizeof(DTYPE);
    int conv1_outputSize = CONV_1_OUTPUT_SIZE*CONV_1_CHANNEL_NUM*sizeof(DTYPE);
    int pool2_outputSize = POOL_2_OUTPUT_SIZE*POOL_2_CHANNEL_NUM*sizeof(DTYPE);
    int conv3_outputSize = CONV_3_OUTPUT_SIZE*CONV_3_CHANNEL_NUM*sizeof(DTYPE);
    int pool4_outputSize = POOL_4_OUTPUT_SIZE*POOL_4_CHANNEL_NUM*sizeof(DTYPE);
    int fc5_outputSize = FC_5_OUTPUT_NEURON_NUM*sizeof(DTYPE);
    int fc6_outputSize = FC_6_OUTPUT_NEURON_NUM*sizeof(DTYPE);
    int fc7_outputSize = FC_7_OUTPUT_NEURON_NUM*sizeof(DTYPE);

    float *images = (float*)malloc(imgNum*inputSize);
    int *labels = (int*)malloc(imgNum*sizeof(int));
    #ifndef SDS
    float *input = (float*)malloc(inputSize);
    DTYPE *conv1_output = (DTYPE*)malloc(conv1_outputSize);
    DTYPE *pool2_output = (DTYPE*)malloc(pool2_outputSize);
    DTYPE *conv3_output = (DTYPE*)malloc(conv3_outputSize);
    #else
    float *input = (float*)sds_alloc(inputSize);
    DTYPE *conv1_output = (DTYPE*)sds_alloc(conv1_outputSize);
    DTYPE *pool2_output = (DTYPE*)sds_alloc(pool2_outputSize);
    DTYPE *conv3_output = (DTYPE*)sds_alloc(conv3_outputSize);
    #endif
    DTYPE *pool4_output = (DTYPE*)malloc(pool4_outputSize);
    DTYPE *fc5_output = (DTYPE*)malloc(fc5_outputSize);
    DTYPE *fc6_output = (DTYPE*)malloc(fc6_outputSize);
    DTYPE *fc7_output = (DTYPE*)malloc(fc7_outputSize);

    read_mnist_image(images, -1.0f, 1.0f, imgNum);
    read_mnist_label(labels, imgNum);

    clock_t start = clock();
    double hw_conv1_time = 0;
    double hw_conv3_time = 0;
    int result_hw = 0;
    for (int i = 0; i < imgNum; i++) {
        for (int j = 0; j < MNIST_PAD_SIZE; j++) 
        {
            input[j] = images[i*MNIST_PAD_SIZE+j];
        }
        if (!((i+1)%100))
        cout << "Processing " << i+1 << "/" << imgNum << " image..." << endl;
        
        clock_t hw_conv1_start = clock();
        convolve_layer_1_hw(input, wConv1, bConv1, conv1_output, i==0);
        clock_t hw_conv1_end = clock();
        hw_conv1_time += (double)(hw_conv1_end-hw_conv1_start)/CLOCKS_PER_SEC;

        pooling_layer_2_sw(conv1_output, wPool2, bPool2, pool2_output);
        
        clock_t hw_conv3_start = clock();
        convolve_layer_3_hw(pool2_output, wConv3, bConv3, conv3_output, i==0);
        clock_t hw_conv3_end = clock();
        hw_conv3_time += (double)(hw_conv3_end-hw_conv3_start)/CLOCKS_PER_SEC;

        pooling_layer_4_sw(conv3_output, wPool4, bPool4, pool4_output);

        fully_connected_layer_5_sw(pool4_output, wFC5, bFC5, fc5_output);
        
        fully_connected_layer_6_sw(fc5_output, wFC6, bFC6, fc6_output);
        
        fully_connected_layer_7_sw(fc6_output, wFC7, bFC7, fc7_output);

        if(labels[i] == max_out(fc7_output)) {
            result_hw++;
        }
    }
    clock_t end = clock();
    double time = (double)(end-start)/CLOCKS_PER_SEC;
    double accuracy = (double)result_hw/imgNum;
    cout << "Hardware test accuracy = " << accuracy*100 << "%" << endl;
    cout << "总共耗时：" << time << " s" << endl;
    cout << "卷积层1耗时：" << hw_conv1_time << "s" << endl;
    cout << "卷积层3耗时：" << hw_conv3_time << "s" << endl;
    result_file << time << "\t" << hw_conv1_time << "\t" << hw_conv3_time << "\t" << accuracy << "\t";

    #ifdef SDS
    sds_free(wConv1);
    sds_free(bConv1);
    sds_free(wConv3);
    sds_free(bConv3);
    #else
    free(wConv1);
    free(bConv1);
    free(wConv3);
    free(bConv3);
    #endif

    free(wPool2);
    free(bPool2);
    free(wPool4);
    free(bPool4);
    free(wFC5);
    free(bFC5);
    free(wFC6);
    free(bFC6);
    free(wFC7);
    free(bFC7);
    
    free(images);
    free(labels);
    #ifdef SDS
    sds_free(input);
    sds_free(conv1_output);
    sds_free(pool2_output);
    sds_free(conv3_output);
    #else
    free(input);
    free(conv1_output);
    free(pool2_output);
    free(conv3_output);
    #endif
    free(pool4_output);
    free(fc5_output);
    free(fc6_output);
    free(fc7_output);
    return time;
}

void test(int imageNum) {
    result_file.open("result.txt", ios_base::app);
    
    if (imageNum!=0) {
        double time_sw = test_sw(imageNum);
        cout << "Software time: " << time_sw << "s" << endl;

        // double time_hw = test_hw(imageNum);
        // cout << "Hardware time: " << time_hw << "s" << endl;
        // cout << "Speed up: " << time_sw/time_hw << endl;
    }
    else {
        result_file << "Image Num\t" << "SW time\t" << "SW_CONV1 time\t" << "SW_CONV3 time\t" << "SW accuracy\t"
            << "HW time\t" << "HW_CONV1 time\t" << "HW_CONV3 time\t" << "HW accuracy\t" 
            << "speed up\t" << "CONV1 speed up\t" << "CONV3 speed up\t" << endl;
        for (int imgNum = 1; imgNum <= 10000; imgNum*=10) {
            result_file << imgNum << "\t";
            double time_sw = test_sw(imgNum);
            double time_hw = test_hw(imgNum);
            double speed_up = time_sw/time_hw;
            cout << "Software time: " << time_sw << "s" << endl;
            cout << "Hardware time: " << time_hw << "s" << endl;
            cout << "Speed up: " << time_sw/time_hw << endl;
            result_file << speed_up << endl;
        }
    }
    result_file.clear();
    result_file.close();
}

