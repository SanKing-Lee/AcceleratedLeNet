#ifndef _LOAD_PARAMETERS_H_
#define _LOAD_PARAMETERS_H_

#define CONV_1_WEIGHT_PATH "./parameters/conv1.weight"
#define CONV_1_BIAS_PATH "./parameters/conv1.bias"
#define POOL_2_WEIGHT_PATH "./parameters/pool2.weight"
#define POOL_2_BIAS_PATH "./parameters/pool2.bias"
#define CONV_3_WEIGHT_PATH "./parameters/conv3.weight"
#define CONV_3_BIAS_PATH "./parameters/conv3.bias"
#define POOL_4_WEIGHT_PATH "./parameters/pool4.weight"
#define POOL_4_BIAS_PATH "./parameters/pool4.bias"
#define FC_5_WEIGHT_PATH "./parameters/fc5.weight"
#define FC_5_BIAS_PATH "./parameters/fc5.bias"
#define FC_6_WEIGHT_PATH "./parameters/fc6.weight"
#define FC_6_BIAS_PATH "./parameters/fc6.bias"
#define FC_7_WEIGHT_PATH "./parameters/fc7.weight"
#define FC_7_BIAS_PATH "./parameters/fc7.bias"

void load_conv_1_para(float *weight, float *bias);
void load_pool_2_para(float *weight, float *bias);
void load_conv_3_para(float *weight, float *bias);
void load_pool_4_para(float *weight, float *bias);
void load_fc_5_para(float *weight, float *bias);
void load_fc_6_para(float *weight, float *bias);
void load_fc_7_para(float *weight, float *bias);

#endif