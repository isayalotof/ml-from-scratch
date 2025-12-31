#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Dataset structure
typedef struct {
    double **features;    // Feature matrix
    int *labels;          // Class labels (0, 1, 2)
    int num_samples;      // Number of samples
    int num_features;     // Number of features
} Dataset;

// Function declarations
Dataset* load_iris_data(const char *filename);
void dataset_free(Dataset *data);
void dataset_print_sample(Dataset *data, int index);
void dataset_shuffle(Dataset *data);
void dataset_split(Dataset *data, Dataset **train, Dataset **test, double train_ratio);

#endif