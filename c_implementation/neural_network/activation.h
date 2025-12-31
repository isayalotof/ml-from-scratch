#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>

// Activation functions
double sigmoid(double x);
double sigmoid_derivative(double x);

double relu(double x);
double relu_derivative(double x);

double tanh_activation(double x);
double tanh_derivative(double x);

// Softmax for output layer
void softmax(double *input, double *output, int length);

#endif