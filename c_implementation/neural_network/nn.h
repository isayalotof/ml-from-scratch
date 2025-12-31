#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../utils/data_loader.h"
#include "activation.h"

// Neural Network structure
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    
    // Weights and biases
    double **weights_input_hidden;   // [input_size x hidden_size]
    double *bias_hidden;              // [hidden_size]
    double **weights_hidden_output;  // [hidden_size x output_size]
    double *bias_output;              // [output_size]
    
    // Activations (for forward pass)
    double *hidden_activation;
    double *output_activation;
    
    // Pre-activations (z values, needed for backprop)
    double *hidden_z;
    double *output_z;
    
    // Learning rate
    double learning_rate;
} NeuralNetwork;

// Function declarations
NeuralNetwork* nn_create(int input_size, int hidden_size, int output_size, double learning_rate);
void nn_free(NeuralNetwork *nn);
void nn_forward(NeuralNetwork *nn, double *input);
void nn_backward(NeuralNetwork *nn, double *input, int true_label);
void nn_train(NeuralNetwork *nn, Dataset *train_data, int epochs);
int nn_predict(NeuralNetwork *nn, double *input);
double nn_evaluate(NeuralNetwork *nn, Dataset *test_data);

#endif