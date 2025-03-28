#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <string>
#include <sstream>

using namespace std;

void vectorPrint(float *u, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        printf("%7.1f \n", u[i]);
    }
    printf("\n");
}

void vectorScale(float *vIn, int rows, float alpha, float *vOut)
{
    for (int i = 0; i < rows; i++)
    {
        vOut[i] = alpha * vIn[i];
    }
}

void vectorDotProduct(float *vA, float *vB, int rows, float *vOut)
{
    for (int i = 0; i < rows; i++)
    {
        vOut[i] += vA[i] * vB[i];
    }
}

void vectorSubtract(float *vA, float *vB, int rows, float *vOut)
{
    for (int i = 0; i < rows; i++)
    {
        vOut[i] = vA[i] - vB[i];
    }
}

float vectorMag(float *v, int rows)
{
    float squareSum = 0;

    for (int i = 0; i < rows; i++)
    {
        squareSum += v[i] * v[i];
    }

    float mag = sqrt(squareSum);

    return mag;
}

void vectorNorm(float *v, int rows, float *vOut)
{
    float mag = vectorMag(v, rows);

    for (int i = 0; i < rows; i++)
    {
        vOut[i] += v[i] / mag;
    }
}