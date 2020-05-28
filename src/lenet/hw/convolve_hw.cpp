#include "constants.h"
#include "types.h"
#include "lenet.h"
#include "test.h"
// #include "mmult.h"
#ifdef SDS
#include "sds_lib.h"
#endif
#include <cstdio>

void convolve_layer_1_hw(   DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output,
                            int init)
{
    DTYPE local_input[CONV_1_INPUT_WH][CONV_1_INPUT_WH];
    DTYPE local_filter[CONV_1_CHANNEL_NUM][CONV_1_FILTER_WH][CONV_1_FILTER_WH];
    DTYPE local_bias[CONV_1_CHANNEL_NUM];
    DTYPE local_output[CONV_1_CHANNEL_NUM][CONV_1_OUTPUT_WH][CONV_1_OUTPUT_WH];
    #pragma HLS array_partition variable=local_input complete dim=2
    #pragma HLS array_partition variable=local_filter complete dim=1
    #pragma HLS array_partition variable=local_bias complete dim=0

    
    for (int loc = 0, i = 0, j = 0; loc < CONV_1_INPUT_SIZE; loc++, j++) {
        #pragma HLS PIPELINE
        if(j == CONV_1_INPUT_WH) {i++; j=0;}
        local_input[i][j] = input[loc];
    }   

    for (int i = 0; i < CONV_1_CHANNEL_NUM; i++) {
        #pragma HLS PIPELINE
        local_bias[i] = bias[i];
        for (int j = 0;j < CONV_1_FILTER_WH; j++) {
            for (int k = 0; k < CONV_1_FILTER_WH; k++) {
                local_filter[i][j][k] = filter[i*CONV_1_FILTER_SIZE+j*CONV_1_FILTER_WH+k];
            }
        }
    }
    #ifdef NORMAL
    {
        for (int channelCount = 0; channelCount < CONV_1_CHANNEL_NUM; channelCount++) 
        {
            for (int row = 0; row < CONV_1_OUTPUT_WH; row+= CONV_1_STRIDE_VER) 
            {
                // image column
                for (int col = 0; col < CONV_1_OUTPUT_WH; col+= CONV_1_STRIDE_HOR) 
                {
                    #pragma HLS PIPELINE
                    // convolve
                    DTYPE tempResult = 0;
                    // filter row
                    for (int row_f = 0; row_f < CONV_1_FILTER_WH; row_f++)
                    {
                        // filter column
                        for (int col_f = 0; col_f < CONV_1_FILTER_WH; col_f++) 
                        {
                            DTYPE inputVal = local_input[row+row_f][col+col_f];
                            PTYPE weightVal = local_filter[channelCount][row_f][col_f];
                            tempResult += inputVal*weightVal;
                        }
                    }
                    PTYPE biasVal = local_bias[channelCount];
                    tempResult += biasVal;
                    local_output[channelCount][row][col] = tanhf(tempResult);
                }
            }
        }
    }
    #endif

    #ifdef MATRIX_2
    {
        DTYPE tempFilter[CONV_1_CHANNEL_NUM][CONV_1_FILTER_SIZE];
        #pragma HLS array_partition variable=tempFilter complete dim=1
        for (int i = 0; i < CONV_1_CHANNEL_NUM; i++) {
            for (int j = 0; j < CONV_1_FILTER_WH; j++) {
                for (int k = 0; k < CONV_1_FILTER_WH; k++) {
                    #pragma HLS PIPELINE
                    tempFilter[i][j*CONV_1_FILTER_WH+k] = local_filter[i][j][k];
                }
            }
        }
        for (int row = 0; row < CONV_1_OUTPUT_WH; row+= CONV_1_STRIDE_VER) 
        {
            DTYPE tempData[CONV_1_FILTER_SIZE][CONV_1_OUTPUT_WH];
            #pragma HLS array_partition variable=tempData complete dim=2
            DTYPE tempResult[CONV_1_CHANNEL_NUM][CONV_1_OUTPUT_WH];
            #pragma HLS array_partition variable=tempResult complete dim=0
            // image column
            for (int col = 0; col < CONV_1_OUTPUT_WH; col+= CONV_1_STRIDE_HOR) 
            {
                #pragma HLS PIPELINE
                // filter row
                for (int row_f = 0; row_f < CONV_1_FILTER_WH; row_f++)
                {
                    // filter column
                    for (int col_f = 0; col_f < CONV_1_FILTER_WH; col_f++) 
                    {
                        tempData[row_f*CONV_1_FILTER_WH+col_f][col] = local_input[row+row_f][col+col_f];
                    }
                }
            }
            for (int k = 0; k < CONV_1_FILTER_SIZE; k++) 
            {
                for (int i = 0; i < CONV_1_CHANNEL_NUM; i++)
                {
                    for (int j = 0; j < CONV_1_OUTPUT_WH; j++) 
                    {
                        #pragma HLS PIPELINE
                        DTYPE last = (k==0)?0:tempResult[i][j];
                        tempResult[i][j] = tempFilter[i][k]*tempData[k][j] + last;
                    }
                }
            }
            for (int i = 0; i < CONV_1_CHANNEL_NUM; i++) 
            {
                for (int j = 0; j < CONV_1_OUTPUT_WH; j++) 
                {
                    #pragma HLS PIPELINE
                    local_output[i][row][j] = tanhf(tempResult[i][j]+local_bias[i]);
                }
            }
        }
    }
    #endif

    for(int i = 0; i < CONV_1_CHANNEL_NUM; i++) {
        for (int j = 0; j < CONV_1_OUTPUT_WH; j++) {
            for (int k = 0; k < CONV_1_OUTPUT_WH; k++) {
                #pragma HLS PIPELINE
                output[i*CONV_1_OUTPUT_SIZE+j*CONV_1_OUTPUT_WH+k] = local_output[i][j][k];
            }
        }
    }
}                            

void convolve_layer_3_hw(   DTYPE *input,
                            PTYPE *filter,
                            PTYPE *bias,
                            DTYPE *output,
                            int init)
{
    DTYPE local_input[POOL_2_CHANNEL_NUM][CONV_3_INPUT_WH][CONV_3_INPUT_WH];
    DTYPE local_filter[CONV_3_CHANNEL_NUM][POOL_2_CHANNEL_NUM][CONV_3_FILTER_WH][CONV_3_FILTER_WH];
    DTYPE local_bias[CONV_3_CHANNEL_NUM];
    DTYPE local_output[CONV_3_CHANNEL_NUM][CONV_3_OUTPUT_WH][CONV_3_OUTPUT_WH];
    #pragma HLS array_partition variable=local_input complete dim=2
    #pragma HLS array_partition variable=local_filter complete dim=1
    #pragma HLS array_partition variable=local_bias complete dim=0

    for (int i = 0; i < POOL_2_CHANNEL_NUM; i++) {
        for (int j = 0; j < CONV_3_INPUT_WH; j++) {
            for (int k = 0; k < CONV_3_INPUT_WH; k++) {
        #pragma HLS PIPELINE
                local_input[i][j][k] = input[i*CONV_3_INPUT_SIZE+j*CONV_3_INPUT_WH+k];
            }
        }
    }     

    for (int i = 0; i < CONV_3_CHANNEL_NUM; i++) {
        local_bias[i] = bias[i];
        for (int h = 0; h < POOL_2_CHANNEL_NUM; h++) {
            for (int j = 0;j < CONV_3_FILTER_WH; j++) {
                for (int k = 0; k < CONV_3_FILTER_WH; k++) {
        #pragma HLS PIPELINE
                    local_filter[i][h][j][k] = filter[i*POOL_2_CHANNEL_NUM*CONV_3_FILTER_SIZE+h*CONV_3_FILTER_SIZE+j*CONV_3_FILTER_WH+k];
                }
            }
        }
    }
    for (int i = 0; i < CONV_3_CHANNEL_NUM; i++) {
        for (int j = 0; j < CONV_3_OUTPUT_WH; j++) {
            for (int k = 0; k < CONV_3_OUTPUT_WH; k++) {
                #pragma HLS PIPELINE
                local_output[i][j][k] = 0;
            }
        }
    }
    for (int channelOutCount = 0; channelOutCount < CONV_3_CHANNEL_NUM; channelOutCount++) 
    {
        // // cout << "Processing channel out: " << channelOutCount << "/" << CONV_3_CHANNEL_NUM << "..."<<endl;
        // channel in convolution layer 1
        for (int channelInCount = 0; channelInCount < POOL_2_CHANNEL_NUM; channelInCount++)
        {
            // access the table to decide whether do convolution
            if (!CONV_3_TABLE[channelInCount*16+channelOutCount]) {
                continue;
            }
            #pragma HLS PIPELINE
            // output row
            for (int row = 0; row < CONV_3_OUTPUT_WH; row+=CONV_3_STRIDE_VER) 
            {
                // output column
                for (int col = 0; col < CONV_3_OUTPUT_WH; col+=CONV_3_STRIDE_HOR)
                {
                    // // cout << "Performing convolution: " << row << ", " << col << endl;
                    // convolve
                    DTYPE tempResult = 0;
                    // filter row
                    for (int row_f = 0; row_f < CONV_3_FILTER_WH; row_f++) 
                    {
                        // filter col
                        for (int col_f = 0; col_f < CONV_3_FILTER_WH; col_f++) 
                        {
                            DTYPE inputVal = local_input[channelInCount][row+row_f][col+col_f];
                            PTYPE weightVal = local_filter[channelOutCount][channelInCount][row_f][col_f];
                            tempResult += inputVal*weightVal;
                        }
                    }
                    local_output[channelOutCount][row][col] += tempResult;
                }
            }
        }
    }

    // write c
    for (int i = 0; i < CONV_3_CHANNEL_NUM; i++) {
        PTYPE biasVal = local_bias[i];
        for (int j = 0; j < CONV_3_OUTPUT_WH; j++) {
            for (int k = 0; k < CONV_3_OUTPUT_WH; k++) {
                #pragma HLS PIPELINE
                output[i*CONV_3_OUTPUT_SIZE+j*CONV_3_OUTPUT_WH+k] 
                    = tanhf(local_output[i][j][k]+biasVal);
            }
        }
    }
}                            
                                        