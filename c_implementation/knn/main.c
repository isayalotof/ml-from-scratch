#include <stdio.h>
#include "knn.h"
#include "../utils/data_loader.h"

int main() {
    printf("=== KNN Classifier for Iris Dataset ===\n\n");
    
    // Load dataset
    Dataset *data = load_iris_data("../../data/iris.csv");
    if (data == NULL) {
        printf("Failed to load dataset\n");
        return 1;
    }
    
    // Print first 3 samples
    printf("\nFirst 3 samples:\n");
    for (int i = 0; i < 3; i++) {
        dataset_print_sample(data, i);
    }
    
    // Shuffle and split dataset
    printf("\nShuffling and splitting dataset (80/20 train/test)...\n");
    dataset_shuffle(data);
    
    Dataset *train_data, *test_data;
    dataset_split(data, &train_data, &test_data, 0.8);
    
    // Train KNN with different k values
    int k_values[] = {3, 5, 7};
    int num_k = 3;
    
    printf("\n=== Testing different K values ===\n");
    
    for (int i = 0; i < num_k; i++) {
        int k = k_values[i];
        printf("\n--- K = %d ---\n", k);
        
        KNN_Model *model = knn_create(train_data, k);
        knn_evaluate(model, test_data);
        
        knn_free(model);
    }
    
    // Cleanup
    dataset_free(data);
    dataset_free(train_data);
    dataset_free(test_data);
    
    printf("\n=== Completed successfully! ===\n");
    return 0;
}