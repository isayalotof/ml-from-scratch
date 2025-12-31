#include <stdio.h>
#include "nn.h"
#include "../utils/data_loader.h"

int main() {
    printf("=== Neural Network Classifier for Iris Dataset ===\n\n");
    
    // Load dataset
    Dataset *data = load_iris_data("../../data/iris.csv");
    if (data == NULL) {
        printf("Failed to load dataset\n");
        return 1;
    }
    
    // Shuffle and split dataset
    printf("Shuffling and splitting dataset (80/20 train/test)...\n");
    dataset_shuffle(data);
    
    Dataset *train_data, *test_data;
    dataset_split(data, &train_data, &test_data, 0.8);
    
    // Create neural network (4 inputs, 8 hidden neurons, 3 outputs)
    printf("\n");
    NeuralNetwork *nn = nn_create(4, 8, 3, 0.01);
    
    // Train network
    nn_train(nn, train_data, 1000);
    
    // Evaluate on test set
    nn_evaluate(nn, test_data);
    
    // Cleanup
    nn_free(nn);
    dataset_free(data);
    dataset_free(train_data);
    dataset_free(test_data);
    
    printf("\n=== Completed successfully! ===\n");
    return 0;
}