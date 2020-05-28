#include "lenet.h"
#include "constants.h"

#include <iostream>
#include <cmath>
using namespace std;

/* name:   pooling2
 * input:  28*28*6
 * filter: 2*2*6
 * output: 14*14*6
 */
void pooling_layer_2_sw(    DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output)
{
    // cout << "Starting layer 2: pooling..." << endl;
    // channel
    for (int channelCount = 0; channelCount < POOL_2_CHANNEL_NUM; channelCount++)
    {
        int inChannelIndex = channelCount*POOL_2_INPUT_SIZE;
        int outChannelIndex = channelCount*POOL_2_OUTPUT_SIZE;
        // cout << "Processing channel " << channelCount << "/" << POOL_2_CHANNEL_NUM << "..." << endl;
        // input feature
        for (int row = 0; row < POOL_2_OUTPUT_WH; row++) 
        {
            int inRowIndex1 = (row * POOL_2_STRIDE_VER)*POOL_2_INPUT_WH;
            int inRowIndex2 = (row * POOL_2_STRIDE_VER + 1)*POOL_2_INPUT_WH;
            int outRowIndex = row*POOL_2_OUTPUT_WH;
            for ( int col = 0; col < POOL_2_OUTPUT_WH; col++) 
            {
                int inColIndex1 = (col*POOL_2_STRIDE_HOR);
                int inColIndex2 = (col*POOL_2_STRIDE_HOR + 1);
                // // cout << "Processing pooling " << row*POOL_2_OUTPUT_WH+col << "/" << POOL_2_OUTPUT_SIZE << "..." << endl;
                // pooling
                DTYPE value = input[inChannelIndex+inRowIndex1+inColIndex1]
                            + input[inChannelIndex+inRowIndex1+inColIndex2]
                            + input[inChannelIndex+inRowIndex2+inColIndex1]
                            + input[inChannelIndex+inRowIndex2+inColIndex2];
                
                PTYPE weight = filter[channelCount]*0.25;
                value = tanhf(value*weight+bias[channelCount]);
                output[outChannelIndex+outRowIndex+col] = value;
            }
        }
    }
    // cout << "Finished!" << endl;
}                            

/* name:   pooling4
 * input:  10*10*16
 * filter: 2*2*16
 * output: 5*5*16
 */
void pooling_layer_4_sw(    DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output)
{
    // cout << "Starting layer 4: pooling..." << endl;
    // input
    float value;
    for (int channelCount = 0; channelCount < POOL_4_CHANNEL_NUM; channelCount++)
    {
        int inChannelIndex = channelCount*POOL_4_INPUT_SIZE;
        int outChannelIndex = channelCount*POOL_4_OUTPUT_SIZE;
        // // cout << "Processing channel " << channelCount << "/" << POOL_4_CHANNEL_NUM << "..." << endl;
        // input feature
        for (int row = 0; row < POOL_4_OUTPUT_WH; row++) 
        {
            int inRowIndex1 = (row * POOL_4_STRIDE_VER)*POOL_4_INPUT_WH;
            int inRowIndex2 = (row * POOL_4_STRIDE_VER+1)*POOL_4_INPUT_WH;
            int outRowIndex = row*POOL_4_OUTPUT_WH;
            for ( int col = 0; col < POOL_4_OUTPUT_WH; col++) 
            {
                int inColIndex1 = (col*POOL_4_STRIDE_HOR);
                int inColIndex2 = (col*POOL_4_STRIDE_HOR + 1);
                // // cout << "Processing pooling " << row*POOL_4_OUTPUT_WH+col << "/" << POOL_4_OUTPUT_SIZE << "..." << endl;
                // pooling
                DTYPE value1 = input[inChannelIndex+inRowIndex1+inColIndex1];
                DTYPE value2 = input[inChannelIndex+inRowIndex1+inColIndex2];
                DTYPE value3 = input[inChannelIndex+inRowIndex2+inColIndex1];
                DTYPE value4 = input[inChannelIndex+inRowIndex2+inColIndex2];
                value = value1+value2+value3+value4;
                PTYPE weight = filter[channelCount]*0.25;
                value = tanhf(value*weight + bias[channelCount]);
                output[outChannelIndex+outRowIndex + col] = value;
            }
        }
    }
    // cout << "Finished!" << endl;
}                            