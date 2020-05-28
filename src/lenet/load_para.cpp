#include "load_para.h"
#include "constants.h"

#include <fstream>
#include <iostream>
#include <string>
using namespace std;

void load_para(string filename, float *para, int size) 
{
    ifstream file(filename.c_str(), ios::in);
	if (file.is_open()) {
		for (int i = 0; i < size; i++) {
			float temp = 0.0;
			file >> temp;
			para[i] = temp;
		}
		cout << "Loading parameters succeed: " << filename << endl;
	}else{
		cout<<"Loading parameters failed: "<<filename<<endl;
	}
    file.clear();
    file.close();
}

void load_conv_1_para(float *weight, float *bias)
{   
    load_para(CONV_1_WEIGHT_PATH, weight, CONV_1_FILTER_SIZE*CONV_1_CHANNEL_NUM);
    load_para(CONV_1_BIAS_PATH, bias, CONV_1_CHANNEL_NUM);
}

void load_pool_2_para(float *weight, float *bias)
{
	load_para(POOL_2_WEIGHT_PATH, weight, POOL_2_CHANNEL_NUM*POOL_2_FILTER_SIZE);
	load_para(POOL_2_BIAS_PATH, bias, POOL_2_CHANNEL_NUM);
}

void load_conv_3_para(float *weight, float *bias) 
{
	load_para(CONV_3_WEIGHT_PATH, weight, CONV_3_CHANNEL_NUM*CONV_3_FILTER_SIZE*POOL_2_CHANNEL_NUM);
	load_para(CONV_3_BIAS_PATH, bias, CONV_3_CHANNEL_NUM);
}

void load_pool_4_para(float *weight, float *bias) 
{
	load_para(POOL_4_WEIGHT_PATH, weight, POOL_4_CHANNEL_NUM*POOL_4_FILTER_SIZE);
	load_para(POOL_4_BIAS_PATH, bias, POOL_4_CHANNEL_NUM);
}

void load_fc_5_para(float *weight, float *bias)
{
	load_para(FC_5_WEIGHT_PATH, weight, FC_5_INPUT_NEURON_NUM*FC_5_OUTPUT_NEURON_NUM);
	load_para(FC_5_BIAS_PATH, bias, FC_5_OUTPUT_NEURON_NUM);
}

void load_fc_6_para(float *weight, float *bias)
{
	load_para(FC_6_WEIGHT_PATH, weight, FC_6_INPUT_NEURON_NUM*FC_6_OUTPUT_NEURON_NUM);
	load_para(FC_5_BIAS_PATH, bias, FC_6_OUTPUT_NEURON_NUM);
}

void load_fc_7_para(float *weight, float *bias)
{
	load_para(FC_7_WEIGHT_PATH, weight, FC_7_INPUT_NEURON_NUM*FC_7_OUTPUT_NEURON_NUM);
	load_para(FC_7_BIAS_PATH, bias, FC_7_BIAS_NUM);
}
