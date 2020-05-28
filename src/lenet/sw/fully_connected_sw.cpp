#include "lenet.h"
#include "constants.h"

#include <iostream>
#include <cmath>    
using namespace std;

/* name:   fully connected 5
 * input:  5*5*16->400
 * filter: /
 * output: 120
 */
void fully_connected_layer_5_sw(    DTYPE *input,
                                    PTYPE *weights,
                                    PTYPE *bias,
                                    DTYPE *output)
{
    // cout << "Starting layer 5: fully connected..." << endl;
    for (int neuronOutCount = 0; neuronOutCount < FC_5_OUTPUT_NEURON_NUM; neuronOutCount++) 
    {
        int weightIndex = neuronOutCount*FC_5_INPUT_NEURON_NUM;
        DTYPE tempResult = 0;
        // cout << "Processing neuron " << neuronOutCount << "/" << FC_5_OUTPUT_NEURON_NUM << "..." << endl;
        for (int neuronInCount = 0; neuronInCount < FC_5_INPUT_NEURON_NUM; neuronInCount++)
        {
            DTYPE inVal = input[neuronInCount];
            PTYPE weightVal = weights[weightIndex+neuronInCount];
            tempResult+=inVal*weightVal;
        }
        PTYPE biasVal = bias[neuronOutCount];
        output[neuronOutCount] = tanhf(tempResult+biasVal);
    }
    // cout << "Finished!" << endl;
}                                    

/* name:   fully connected layer 6
 * input:  120
 * filter: /
 * output: 84
 */
void fully_connected_layer_6_sw(    DTYPE *input,
                                    PTYPE *weights,
                                    PTYPE *bias,
                                    DTYPE *output)
{
    // cout << "Starting layer 6: fully connected..." << endl;
    // cout << "Processing input " << batchCount << "/" << BATCH_SIZE << "..." << endl;
    for (int neuronOutCount = 0; neuronOutCount < FC_6_OUTPUT_NEURON_NUM; neuronOutCount++)
    {
        DTYPE tempResult = 0;
        // cout << "Processing neuron " << neuronOutCount << "/" << FC_6_OUTPUT_NEURON_NUM << "..." << endl;
        for (int neuronInCount = 0; neuronInCount < FC_6_INPUT_NEURON_NUM; neuronInCount++)
        {
            DTYPE inputVal = input[neuronInCount];
            PTYPE weightVal = weights[neuronInCount*FC_6_OUTPUT_NEURON_NUM + neuronOutCount];
            tempResult += inputVal*weightVal;
        }
        PTYPE biasVal = bias[neuronOutCount];
        output[neuronOutCount] = tanhf(tempResult+biasVal);
    }
    // cout << "Finished!" << endl;
}              

void fully_connected_layer_7_sw(    DTYPE *input,
                                    PTYPE *weights,
                                    PTYPE *bias,
                                    DTYPE *output)
{
    // cout << "Starting layer 7: fully connected..." << endl;
    // cout << "Processing input " << batchCount << "/" << BATCH_SIZE << "..." << endl;
    for (int neuronOutCount = 0; neuronOutCount < FC_7_OUTPUT_NEURON_NUM; neuronOutCount++)
    {
        DTYPE tempResult = 0;
        // cout << "Processing neuron " << neuronOutCount << "/" << FC_7_OUTPUT_NEURON_NUM << "..." << endl;
        for (int neuronInCount = 0; neuronInCount < FC_7_INPUT_NEURON_NUM; neuronInCount++)
        {
            DTYPE inputVal = input[neuronInCount];
            PTYPE weightVal = weights[neuronInCount*FC_7_OUTPUT_NEURON_NUM + neuronOutCount];
            tempResult += inputVal*weightVal;
        }
        PTYPE biasVal = bias[neuronOutCount];
        output[neuronOutCount] = tanhf(tempResult+biasVal);
    }
    // cout << "Finished!" << endl;
}                                