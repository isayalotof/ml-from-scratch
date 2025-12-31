#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Matrix structure
typedef struct {
    double **data;
    int rows;
    int cols;
} Matrix;

// Matrix operations
Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix *m);
void matrix_print(Matrix *m);
Matrix* matrix_multiply(Matrix *a, Matrix *b);
Matrix* matrix_transpose(Matrix *m);
void matrix_fill_random(Matrix *m, double min, double max);

// Vector operations
double euclidean_distance(double *a, double *b, int length);
void vector_print(double *v, int length);

#endif