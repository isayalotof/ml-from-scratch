#include "knn.h"
#include <string.h>

// Comparison function for sorting neighbors by distance
int compare_neighbors(const void *a, const void *b) {
    Neighbor *n1 = (Neighbor*)a;
    Neighbor *n2 = (Neighbor*)b;
    
    if (n1->distance < n2->distance) return -1;
    if (n1->distance > n2->distance) return 1;
    return 0;
}

// Create KNN model
KNN_Model* knn_create(Dataset *train_data, int k) {
    KNN_Model *model = (KNN_Model*)malloc(sizeof(KNN_Model));
    model->train_data = train_data;
    model->k = k;
    
    printf("Created KNN model with k=%d\n", k);
    return model;
}

// Free KNN model (doesn't free train_data, just the model structure)
void knn_free(KNN_Model *model) {
    if (model) free(model);
}

// Predict class for a single sample
int knn_predict(KNN_Model *model, double *features) {
    int n_samples = model->train_data->num_samples;
    int n_features = model->train_data->num_features;
    
    // Calculate distances to all training samples
    Neighbor *neighbors = (Neighbor*)malloc(n_samples * sizeof(Neighbor));
    
    for (int i = 0; i < n_samples; i++) {
        neighbors[i].index = i;
        neighbors[i].distance = euclidean_distance(features, 
                                                   model->train_data->features[i], 
                                                   n_features);
        neighbors[i].label = model->train_data->labels[i];
    }
    
    // Sort neighbors by distance
    qsort(neighbors, n_samples, sizeof(Neighbor), compare_neighbors);
    
    // Count votes from k nearest neighbors
    int votes[3] = {0, 0, 0};  // 3 classes in Iris dataset
    
    for (int i = 0; i < model->k && i < n_samples; i++) {
        int label = neighbors[i].label;
        if (label >= 0 && label < 3) {
            votes[label]++;
        }
    }
    
    // Find class with most votes
    int predicted_class = 0;
    int max_votes = votes[0];
    
    for (int i = 1; i < 3; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            predicted_class = i;
        }
    }
    
    free(neighbors);
    return predicted_class;
}

// Evaluate model on test dataset
double knn_evaluate(KNN_Model *model, Dataset *test_data) {
    int correct = 0;
    
    printf("\nEvaluating KNN model...\n");
    
    for (int i = 0; i < test_data->num_samples; i++) {
        int predicted = knn_predict(model, test_data->features[i]);
        int actual = test_data->labels[i];
        
        if (predicted == actual) {
            correct++;
        }
        
        // Print first 10 predictions for verification
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