#include "activation.h"

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Sigmoid derivative
double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

// ReLU activation function
double relu(double x) {
    return (x > 0) ? x : 0;
}

// ReLU derivative
double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

// Tanh activation function
double tanh_activation(double x) {
    return tanh(x);
}

// Tanh derivative
double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

// Softmax activation (for output layer)
void softmax(double *input, double *output, int length) {
    double max = input[0];
    
    // Find max for numerical stability
    for (int i = 1; i < length; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }
    
    // Compute exp and sum
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }
    
    // Normalize
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}