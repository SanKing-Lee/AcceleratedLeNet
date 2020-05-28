#include "lenet.h"
#include "constants.h"
#include "read_mnist.h"
#include "load_para.h"

#include <math.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

/* name:   convolve1
 * input:  32*32
 * filter: 5*5*6
 * output: 28*28*6
 */
void convolve_layer_1_sw(   DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output)
{
    // cout << "Starting layer1: convolution..." << endl;
    for (int channelCount = 0; channelCount < CONV_1_CHANNEL_NUM; channelCount++) 
    {
        for (int row = 0; row < CONV_1_OUTPUT_WH; row+= CONV_1_STRIDE_VER) 
        {
            // image column
            for (int col = 0; col < CONV_1_OUTPUT_WH; col+= CONV_1_STRIDE_HOR) 
            {
                // convolve
                DTYPE tempResult = 0;
                // filter row
                for (int row_f = 0; row_f < CONV_1_FILTER_WH; row_f++)
                {
                    // filter column
                    for (int col_f = 0; col_f < CONV_1_FILTER_WH; col_f++) 
                    {
                        int inRowIndex = (row+row_f)*CONV_1_INPUT_WH;
                        int inColIndex = col+col_f;
                        DTYPE inputVal = input[inRowIndex+inColIndex];
                        // DTYPE inputVal = input[inMapIndex+inRowIndex+inColIndex];
                        int filterIndex = channelCount*CONV_1_FILTER_SIZE;
                        int rowfIndex = row_f*CONV_1_FILTER_WH;
                        int colfIndex = col_f;
                        PTYPE weightVal = filter[filterIndex+rowfIndex+colfIndex];
                        tempResult += inputVal*weightVal;
                    }
                    // cout << inMapIndex +inRowIndex<< " ";
                }
                PTYPE biasVal = bias[channelCount];
                tempResult += biasVal;
                // cout << outMapIndex+outChannelIndex+outRowIndex+outColIndex << " ";
                int outChannelIndex = channelCount*CONV_1_OUTPUT_SIZE;
                int outRowIndex = row*CONV_1_OUTPUT_WH;
                int outColIndex = col;
                output[outChannelIndex+outRowIndex+outColIndex] = tanhf(tempResult);
            }
            // cout << endl;
        }

    }
    // cout << "Finished!" << endl;
}                            

/* name:   convolve3
 * input:  14*14*6
 * filter: 5*5*16
 * output: 10*10*16
 */
void convolve_layer_3_sw(   DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output)
{
    // cout << "Starting laryer 3: convolution..." << endl;
    for (int i = 0; i < 1600; i++) {
        output[i] = 0;
    }
    for (int channelOutCount = 0; channelOutCount < CONV_3_CHANNEL_NUM; channelOutCount++) 
    {
        int outChannelIndex = channelOutCount*CONV_3_OUTPUT_SIZE;
        int fltOutIndex = channelOutCount*POOL_2_CHANNEL_NUM*CONV_3_FILTER_SIZE;
        // // cout << "Processing channel out: " << channelOutCount << "/" << CONV_3_CHANNEL_NUM << "..."<<endl;
        // channel in convolution layer 1
        for (int channelInCount = 0; channelInCount < POOL_2_CHANNEL_NUM; channelInCount++)
        {
            // access the table to decide whether do convolution
            if (!CONV_3_TABLE[channelInCount*16+channelOutCount]) {
                continue;
            }
            int inChannelIndex = channelInCount*POOL_2_OUTPUT_SIZE;
            int fltInIndex = (channelInCount)*CONV_3_FILTER_SIZE;
            // // cout <<"Processing channel in: " << channelInCount << "..." << endl;
            // output row
            for (int row = 0; row < CONV_3_OUTPUT_WH; row+=CONV_3_STRIDE_VER) 
            {
                int outRowIndex = row*CONV_3_OUTPUT_WH;
                // output column
                for (int col = 0; col < CONV_3_OUTPUT_WH; col+=CONV_3_STRIDE_HOR)
                {
                    // // cout << "Performing convolution: " << row << ", " << col << endl;
                    // convolve
                    DTYPE tempResult = 0;
                    // filter row
                    for (int row_f = 0; row_f < CONV_3_FILTER_WH; row_f++) 
                    {
                        int inRowIndex = (row+row_f)*CONV_3_INPUT_WH;    
                        int fltRowIndex = row_f*CONV_3_FILTER_WH;
                        // filter col
                        for (int col_f = 0; col_f < CONV_3_FILTER_WH; col_f++) 
                        {
                            int inColIndex = col+col_f;
                            DTYPE inputVal = input[inChannelIndex+inRowIndex+inColIndex];
                            PTYPE weightVal = filter[fltOutIndex+fltInIndex+fltRowIndex+col_f];
                            tempResult += inputVal*weightVal;
                        }
                    }
                    output[outChannelIndex+outRowIndex + col] += tempResult;
                }
            }
        }
        // add bias
        for (int i = 0; i < CONV_3_OUTPUT_SIZE; i++) 
        {
            DTYPE oldVal = output[outChannelIndex + i];
            DTYPE biasVal = bias[channelOutCount];
            DTYPE newVal = oldVal + biasVal;
            output[outChannelIndex+i] = newVal;
        }
    }
    // tanhf
    for (int i = 0; i < CONV_3_OUTPUT_SIZE*CONV_3_CHANNEL_NUM; i++) {
        output[i] = tanhf(output[i]);
    }
    // cout << "Finished!" << endl;
}                            