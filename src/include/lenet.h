#ifndef _LENET_H_
#define _LENET_H_
#include "types.h"
#include "constants.h"

// #define SDS
#define MATRIX_2 // NORMAL MATRIX_1 MATRIX_2 MATRIX_3

void convolve_layer_1_sw(   DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output);

void pooling_layer_2_sw(    DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output);


void convolve_layer_3_sw(   DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output);

void pooling_layer_4_sw(    DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output);

void fully_connected_layer_5_sw(    DTYPE *input,
                                    PTYPE *weights,
                                    PTYPE *bias,
                                    DTYPE *output);

void fully_connected_layer_6_sw(    DTYPE *input,
                                    PTYPE *weights,
                                    PTYPE *bias,
                                    DTYPE *output);

void fully_connected_layer_7_sw(    DTYPE *input,
                                    PTYPE *weights,
                                    PTYPE *bias,
                                    DTYPE *output);                                    

#pragma SDS data access_pattern(input:SEQUENTIAL, filter:SEQUENTIAL, bias:SEQUENTIAL, output:SEQUENTIAL)
#pragma SDS data zero_copy(input[0:CONV_1_INPUT_SIZE], filter[0:CONV_1_CHANNEL_NUM*CONV_1_FILTER_SIZE], bias[0:CONV_1_CHANNEL_NUM], output[0:CONV_1_CHANNEL_NUM*CONV_1_OUTPUT_SIZE])
void convolve_layer_1_hw(   DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output,
                            int init);

void pool_layer_2_hw(   DTYPE *input,
                        PTYPE *filter,
                        PTYPE *bias,
                        DTYPE *output,
                        int init);      

#pragma SDS data access_pattern(input:SEQUENTIAL, filter:SEQUENTIAL, bias:SEQUENTIAL, output:SEQUENTIAL)
#pragma SDS data zero_copy(input[0:POOL_2_CHANNEL_NUM*CONV_3_INPUT_SIZE], filter[0:CONV_3_CHANNEL_NUM*POOL_2_CHANNEL_NUM*CONV_3_FILTER_SIZE], bias[0:CONV_3_CHANNEL_NUM], output[0:CONV_3_CHANNEL_NUM*CONV_3_OUTPUT_SIZE])
void convolve_layer_3_hw(   DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output,
                            int init);

void pool_layer_4_hw(   DTYPE *input,
                        PTYPE *filter,
                        PTYPE *bias,
                        DTYPE *output,
                        int init);                                                        

#endif 
