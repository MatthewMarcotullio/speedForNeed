#ifndef UTILS_H
#define UTILS_H

using namespace std;

void vectorPrint(float* u , int rows); 

void vectorScale(float* u, int rows, float alpha , float* v); 

float vectorDotProduct(float* u, float* v, int u_rows, int v_rows);

void vectorSubtract(float* u, float* v, int u_rows, int v_rows, float* output_vect);

float vectorMag(float* u, int rows);

float* vectorNorm(float* u, int rows);

#endif