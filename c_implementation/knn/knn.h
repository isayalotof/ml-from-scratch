#ifndef KNN_H
#define KNN_H

#include "../utils/data_loader.h"
#include "../utils/matrix.h"

// Neighbor structure for sorting
typedef struct {
    int index;
    double distance;
    int label;
} Neighbor;

// KNN Model
typedef struct {
    Dataset *train_data;
    int k;  // Number of neighbors
} KNN_Model;

// Function declarations
KNN_Model* knn_create(Dataset *train_data, int k);
void knn_free(KNN_Model *model);
int knn_predict(KNN_Model *model, double *features);
double knn_evaluate(KNN_Model *model, Dataset *test_data);

#endif