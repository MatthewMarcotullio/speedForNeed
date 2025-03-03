#ifndef UTILS_H
#define UTILS_H

using namespace std;

void vectorPrint(float *u, int rows);

void vectorScale(float *vIn, int rows, float alpha, float *vOut);

void vectorDotProduct(float *vA, float* vB, int rows, float* vOut);

void vectorSubtract(float *vA, float* vB, int rows, float* vOut);

float vectorMag(float* v, int rows);

void vectorNorm(float *v, int rows, float* vOut);

#endif
