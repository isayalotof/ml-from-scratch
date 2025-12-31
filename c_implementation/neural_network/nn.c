#include "nn.h"

// Random weight initialization (Xavier initialization)
double random_weight(int fan_in, int fan_out) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    return ((double)rand() / RAND_MAX) * 2 * limit - limit;
}

// Create neural network
NeuralNetwork* nn_create(int input_size, int hidden_size, int output_size, double learning_rate) {
    srand(time(NULL));
    
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;
    nn->learning_rate = learning_rate;
    
    // Allocate weights: input -> hidden
    nn->weights_input_hidden = (double**)malloc(input_size * sizeof(double*));
    for (int i = 0; i < input_size; i++) {
        nn->weights_input_hidden[i] = (double*)malloc(hidden_size * sizeof(double));
        for (int j = 0; j < hidden_size; j++) {
            nn->weights_input_hidden[i][j] = random_weight(input_size, hidden_size);
        }
    }
    
    // Allocate bias: hidden layer
    nn->bias_hidden = (double*)calloc(hidden_size, sizeof(double));
    
    // Allocate weights: hidden -> output
    nn->weights_hidden_output = (double**)malloc(hidden_size * sizeof(double*));
    for (int i = 0; i < hidden_size; i++) {
        nn->weights_hidden_output[i] = (double*)malloc(output_size * sizeof(double));
        for (int j = 0; j < output_size; j++) {
            nn->weights_hidden_output[i][j] = random_weight(hidden_size, output_size);
        }
    }
    
    // Allocate bias: output layer
    nn->bias_output = (double*)calloc(output_size, sizeof(double));
    
    // Allocate activation arrays
    nn->hidden_activation = (double*)malloc(hidden_size * sizeof(double));
    nn->output_activation = (double*)malloc(output_size * sizeof(double));
    nn->hidden_z = (double*)malloc(hidden_size * sizeof(double));
    nn->output_z = (double*)malloc(output_size * sizeof(double));
    
    printf("Created Neural Network: %d -> %d -> %d\n", input_size, hidden_size, output_size);
    printf("Learning rate: %.4f\n", learning_rate);
    
    return nn;
}

// Free neural network memory
void nn_free(NeuralNetwork *nn) {
    if (nn == NULL) return;
    
    for (int i = 0; i < nn->input_size; i++) {
        free(nn->weights_input_hidden[i]);
    }
    free(nn->weights_input_hidden);
    free(nn->bias_hidden);
    
    for (int i = 0; i < nn->hidden_size; i++) {
        free(nn->weights_hidden_output[i]);
    }
    free(nn->weights_hidden_output);
    free(nn->bias_output);
    
    free(nn->hidden_activation);
    free(nn->output_activation);
    free(nn->hidden_z);
    free(nn->output_z);
    
    free(nn);
}

// Forward propagation
void nn_forward(NeuralNetwork *nn, double *input) {
    // Input -> Hidden layer
    for (int j = 0; j < nn->hidden_size; j++) {
        nn->hidden_z[j] = nn->bias_hidden[j];
        
        for (int i = 0; i < nn->input_size; i++) {
            nn->hidden_z[j] += input[i] * nn->weights_input_hidden[i][j];
        }
        
        // Apply ReLU activation
        nn->hidden_activation[j] = relu(nn->hidden_z[j]);
    }
    
    // Hidden -> Output layer
    for (int j = 0; j < nn->output_size; j++) {
        nn->output_z[j] = nn->bias_output[j];
        
        for (int i = 0; i < nn->hidden_size; i++) {
            nn->output_z[j] += nn->hidden_activation[i] * nn->weights_hidden_output[i][j];
        }
    }
    
    // Apply softmax to output
    softmax(nn->output_z, nn->output_activation, nn->output_size);
}

// Backpropagation
void nn_backward(NeuralNetwork *nn, double *input, int true_label) {
    // Output layer gradients (cross-entropy + softmax)
    double *output_delta = (double*)malloc(nn->output_size * sizeof(double));
    
    for (int i = 0; i < nn->output_size; i++) {
        // Derivative of cross-entropy with softmax
        output_delta[i] = nn->output_activation[i];
        if (i == true_label) {
            output_delta[i] -= 1.0;
        }
    }
    
    // Hidden layer gradients
    double *hidden_delta = (double*)malloc(nn->hidden_size * sizeof(double));
    
    for (int i = 0; i < nn->hidden_size; i++) {
        hidden_delta[i] = 0.0;
        
        for (int j = 0; j < nn->output_size; j++) {
            hidden_delta[i] += output_delta[j] * nn->weights_hidden_output[i][j];
        }
        
        // Apply ReLU derivative
        hidden_delta[i] *= relu_derivative(nn->hidden_z[i]);
    }
    
    // Update weights: hidden -> output
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->output_size; j++) {
            nn->weights_hidden_output[i][j] -= nn->learning_rate * output_delta[j] * nn->hidden_activation[i];
        }
    }
    
    // Update bias: output layer
    for (int j = 0; j < nn->output_size; j++) {
        nn->bias_output[j] -= nn->learning_rate * output_delta[j];
    }
    
    // Update weights: input -> hidden
    for (int i = 0; i < nn->input_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->weights_input_hidden[i][j] -= nn->learning_rate * hidden_delta[j] * input[i];
        }
    }
    
    // Update bias: hidden layer
    for (int j = 0; j < nn->hidden_size; j++) {
        nn->bias_hidden[j] -= nn->learning_rate * hidden_delta[j];
    }
    
    free(output_delta);
    free(hidden_delta);
}

// Train neural network
void nn_train(NeuralNetwork *nn, Dataset *train_data, int epochs) {
    printf("\nTraining neural network for %d epochs...\n", epochs);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        for (int i = 0; i < train_data->num_samples; i++) {
            // Forward pass
            nn_forward(nn, train_data->features[i]);
            
            // Calculate loss (cross-entropy)
            int true_label = train_data->labels[i];
            total_loss += -log(nn->output_activation[true_label] + 1e-10);
            
            // Backward pass
            nn_backward(nn, train_data->features[i], true_label);
        }
        
        // Print progress every 100 epochs
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            double avg_loss = total_loss / train_data->num_samples;
            printf("Epoch %d/%d - Loss: %.4f\n", epoch + 1, epochs, avg_loss);
        }
    }
    
    printf("Training complete!\n");
}

// Predict class for a single sample
int nn_predict(NeuralNetwork *nn, double *input) {
    nn_forward(nn, input);
    
    // Find class with highest probability
    int predicted_class = 0;
    double max_prob = nn->output_activation[0];
    
    for (int i = 1; i < nn->output_size; i++) {
        if (nn->output_activation[i] > max_prob) {
            max_prob = nn->output_activation[i];
            predicted_class = i;
        }
    }
    
    return predicted_class;
}

// Evaluate neural network on test data
double nn_evaluate(NeuralNetwork *nn, Dataset *test_data) {
    int correct = 0;
    
    printf("\nEvaluating neural network...\n");
    
    for (int i = 0; i < test_data->num_samples; i++) {
        int predicted = nn_predict(nn, test_data->features[i]);
        int actual = test_data->labels[i];
        
        if (predicted == actual) {
            correct++;
        }
        
        // Print first 10 predictions
        if (i < 10) {
            printf("Sample %d: Predicted=%d, Actual=%d %s\n", 
                   i, predicted, actual,
                   (predicted == actual) ? "[CORRECT]" : "[WRONG]");
        }
    }
    
    double accuracy = (double)correct / test_data->num_samples;
    printf("\nAccuracy: %.2f%% (%d/%d correct)\n", 
           accuracy * 100, correct, test_data->num_samples);
    
    return accuracy;
}