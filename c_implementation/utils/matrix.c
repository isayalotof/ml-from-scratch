#include "matrix.h"
#include <time.h>

// Create matrix with allocated memory
Matrix* matrix_create(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    
    m->data = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        m->data[i] = (double*)calloc(cols, sizeof(double));
    }
    
    return m;
}

// Free matrix memory
void matrix_free(Matrix *m) {
    if (m == NULL) return;
    
    for (int i = 0; i < m->rows; i++) {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

// Print matrix
void matrix_print(Matrix *m) {
    printf("Matrix [%d x %d]:\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%.4f ", m->data[i][j]);
        }
        printf("\n");
    }
}

// Matrix multiplication
Matrix* matrix_multiply(Matrix *a, Matrix *b) {
    if (a->cols != b->rows) {
        printf("Error: incompatible dimensions for multiplication\n");
        return NULL;
    }
    
    Matrix *result = matrix_create(a->rows, b->cols);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i][k] * b->data[k][j];
            }
            result->data[i][j] = sum;
        }
    }
    
    return result;
}

// Matrix transpose
Matrix* matrix_transpose(Matrix *m) {
    Matrix *result = matrix_create(m->cols, m->rows);
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[j][i] = m->data[i][j];
        }
    }
    
    return result;
}

// Fill matrix with random values
void matrix_fill_random(Matrix *m, double min, double max) {
    srand(time(NULL));
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double random = (double)rand() / RAND_MAX;
            m->data[i][j] = min + random * (max - min);
        }
    }
}

// Euclidean distance between two vectors
double euclidean_distance(double *a, double *b, int length) {
    double sum = 0.0;
    
    for (int i = 0; i < length; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return sqrt(sum);
}

// Print vector
void vector_print(double *v, int length) {
    printf("[");
    for (int i = 0; i < length; i++) {
        printf("%.4f", v[i]);
        if (i < length - 1) printf(", ");
    }
    printf("]\n");
}