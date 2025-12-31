#include "data_loader.h"
#include <time.h>

// Convert class name to integer label
int class_to_label(const char *class_name) {
    if (strcmp(class_name, "Iris-setosa") == 0) return 0;
    if (strcmp(class_name, "Iris-versicolor") == 0) return 1;
    if (strcmp(class_name, "Iris-virginica") == 0) return 2;
    return -1;
}

// Load Iris dataset from CSV
Dataset* load_iris_data(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Could not open file %s\n", filename);
        return NULL;
    }
    
    Dataset *data = (Dataset*)malloc(sizeof(Dataset));
    data->num_samples = 150;  // Iris has 150 samples
    data->num_features = 4;   // 4 features
    
    // Allocate memory
    data->features = (double**)malloc(data->num_samples * sizeof(double*));
    for (int i = 0; i < data->num_samples; i++) {
        data->features[i] = (double*)malloc(data->num_features * sizeof(double));
    }
    data->labels = (int*)malloc(data->num_samples * sizeof(int));
    
    // Read data
    char line[256];
    int sample_idx = 0;
    
    while (fgets(line, sizeof(line), file) && sample_idx < data->num_samples) {
        // Skip empty lines
        if (strlen(line) < 5) continue;
        
        char class_name[50];
        int parsed = sscanf(line, "%lf,%lf,%lf,%lf,%s",
                           &data->features[sample_idx][0],
                           &data->features[sample_idx][1],
                           &data->features[sample_idx][2],
                           &data->features[sample_idx][3],
                           class_name);
        
        if (parsed == 5) {
            data->labels[sample_idx] = class_to_label(class_name);
            sample_idx++;
        }
    }
    
    data->num_samples = sample_idx;  // Update in case we read fewer lines
    fclose(file);
    
    printf("Loaded %d samples with %d features\n", data->num_samples, data->num_features);
    return data;
}

// Free dataset memory
void dataset_free(Dataset *data) {
    if (data == NULL) return;
    
    for (int i = 0; i < data->num_samples; i++) {
        free(data->features[i]);
    }
    free(data->features);
    free(data->labels);
    free(data);
}

// Print a single sample
void dataset_print_sample(Dataset *data, int index) {
    if (index >= data->num_samples) {
        printf("Error: index out of bounds\n");
        return;
    }
    
    printf("Sample %d: [", index);
    for (int i = 0; i < data->num_features; i++) {
        printf("%.2f", data->features[index][i]);
        if (i < data->num_features - 1) printf(", ");
    }
    printf("] -> Class %d\n", data->labels[index]);
}

// Shuffle dataset (for train/test split)
void dataset_shuffle(Dataset *data) {
    srand(time(NULL));
    
    for (int i = data->num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // Swap features
        double *temp_features = data->features[i];
        data->features[i] = data->features[j];
        data->features[j] = temp_features;
        
        // Swap labels
        int temp_label = data->labels[i];
        data->labels[i] = data->labels[j];
        data->labels[j] = temp_label;
    }
}

// Split dataset into train and test
void dataset_split(Dataset *data, Dataset **train, Dataset **test, double train_ratio) {
    int train_size = (int)(data->num_samples * train_ratio);
    int test_size = data->num_samples - train_size;
    
    // Create train dataset
    *train = (Dataset*)malloc(sizeof(Dataset));
    (*train)->num_samples = train_size;
    (*train)->num_features = data->num_features;
    (*train)->features = (double**)malloc(train_size * sizeof(double*));
    (*train)->labels = (int*)malloc(train_size * sizeof(int));
    
    for (int i = 0; i < train_size; i++) {
        (*train)->features[i] = (double*)malloc(data->num_features * sizeof(double));
        memcpy((*train)->features[i], data->features[i], 
               data->num_features * sizeof(double));
        (*train)->labels[i] = data->labels[i];
    }
    
    // Create test dataset
    *test = (Dataset*)malloc(sizeof(Dataset));
    (*test)->num_samples = test_size;
    (*test)->num_features = data->num_features;
    (*test)->features = (double**)malloc(test_size * sizeof(double*));
    (*test)->labels = (int*)malloc(test_size * sizeof(int));
    
    for (int i = 0; i < test_size; i++) {
        (*test)->features[i] = (double*)malloc(data->num_features * sizeof(double));
        memcpy((*test)->features[i], data->features[train_size + i], 
               data->num_features * sizeof(double));
        (*test)->labels[i] = data->labels[train_size + i];
    }
    
    printf("Split dataset: %d train samples, %d test samples\n", train_size, test_size);
}