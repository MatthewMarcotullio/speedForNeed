#ifndef MATRIXUTILS_H
#define MATRIXUTILS_H


using namespace std;

void matrixPrint(float* matrix, int rows, int cols);  

void matrixProduct(float* a, int rows_a, int cols_a, float* b, int rows_b, int cols_b, float* c); 

void matrixTranspose(float* a, int rows, int cols, float* aTranspose);

void matrixExtractCol(float*a, int rows, int cols, int col, float* column);


#endif

