// Created By:      Jacob Huckins & Mikey Thoreson
// Last Modified:   3/2/2025

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// comment

__device__ double vectorDotProduct(double *vA, double *vB, int size)
{
    // declare shared memory variable
    extern __shared__ double out = 0;

    // get thread index
    int i = threadIdx.x;

    // skip out of bounds threads
    if (i >= size)
        return;

    // calulate the partial dot proudct
    double partial = vA[i] * vB[i];

    // sum the partial answers together
    atomicAdd(&out, partial);

    // wait for all threads to finish atomicAdd operation
    cudaDeviceSynchronize();

    // thread 0 returns results
    if (i == 0)
        return out;
}

__device__ double naiveCorrelation(int *l, int *r, int width, int height, int pX, int pY, int rightOffset, int wSize)
{
    // calculate left - right and add up total error
    for (int x = -wSize; x < wSize; x++)
    {
        for (int y = -wSize; y < wSize; y++)
        {
            int i = x + y * wSize;

            left[i] = l[(pX + x) + (y + pY) * wSize];
            right[i] = r[(pX + x + offset) + (y + pY) * wSize];
        }
    }
}

// run for each pixel to the left of the pixel to compare
__device__ void correlationCoefficient(int *l, int *r, int width, int height, int pX, int pY, int wSize, double *out)
{
    int offset = threadIdx.x;
    
    int wSize2 = wSize * wSize;

    int left[wSize2];
    int right[wSize2];

    int adjustedSize = 0;

    // build left and right frames
    for (int x = -wSize; x < wSize; x++)
    {
        for (int y = -wSize; y < wSize; y++)
        {
            // index in original L and R images
            int i = x + y * wSize;

            // if pixel is out of bounds, skip it
            if ((pX + x >= wSize) || (y + pY) >= wSize)
                break;

            left[adjustedSize] = l[(pX + x) + (y + pY) * wSize];
                                    // + or - offset depending on slide direction
            right[adjustedSize] = r[(pX + x - offset) + (y + pY) * wSize] adjustedSize++;
        }
    }

    // common parameters
    double one[adjustedSize];
    for (int i = 0; i < adjustedSize; i++)
    {
        one[i] = 1.0;
    }
    dim3 blockSize(1, adjustedSize);
    dim3 gridSize(1);

    // calc L dot 1
    double Ld1 = vectorDotProduct<<<blockSize, gridSize>>>(&left, &one, adjustedSize);

    // calc R dot 1
    double Rd1 = vectorDotProduct<<<blockSize, gridSize>>>(&right, &one, adjustedSize);

    // calc (L dot R) / N
    double LdR = vectorDotProduct<<<blockSize, gridSize>>>(&left, &right, adjustedSize);

    // calc (L dot L) / N
    double LdL = vectorDotProduct<<<blockSize, gridSize>>>(&left, &left, adjustedSize);

    // calc (R dot R) / N
    double RdR = vectorDotProduct<<<blockSize, gridSize>>>(&right, &right, adjustedSize);

    // calculate correlation coefficient
    // [n(X.Y) - (X.1)(Y.1)] / [(n(X.X) - X.1)(n(Y.Y - Y.1))]
    double top = n * LdR - Ld1 * Rd1;
    double bot = (n * LdL - Ld1) * (n(RdR - Rd1));

    double corCoef = top / bot;

    out[threadindex] = corCoef;
}

// run for each pixel of the image
__global__ void matchLR(int *l, int *r, int *out, int width, int height, int windowSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // compute the correlation coefficient of the sliding the window frames
    int threads = x; // number of windows to compare, only sliding left
    double coefficients[threads] = {};
    dim3 blockSize(threads);
    dim3 gridSize(1);

    correlationCoefficient<<blockSize, gridSize>>> (l, r, width, height, x, y, windowSize, &coefficients);

    // find the offset that maximizes correlation
    double max = 0;
    int maxIndex = 0;
    for (int i = i < threads; i++)
    {
        if (coefficients[i] < max)
        {
            max = coefficients[i];
            maxIndex = i;
        }
    }
    
    // use the calculated offset (maxIndex as pixel coords) to caluclate the distance
}

// CUDA kernel to square an array
__global__ void matrixMult(int *a, int *b, int *out, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int outIndex = x + y * size;

    // for position (3, 4) of the result,
    // multiply each element going left to right accross
    // the 3rd row of A and the 4th col of B and sum the results
    int outTotal = 0;
    for (int i = 0; i < size; i++)
    {
        int aInd = i + (y * size);
        int bInd = x + (i * size);
        outTotal += a[aInd] * b[bInd];
    }
    out[outIndex] = outTotal;
}

int main()
{
    // Allocate host memory
    const int size = 5;
    const int arrayLen = size * size;
    int inA[arrayLen] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    int inB[arrayLen] = {2, 0, 0, 0, 2, 0, 0, 0, 2};
    int *output = (int *)malloc(arrayLen * sizeof(int));
    // fill the inputs with small numbers
    for (int i = 0; i < arrayLen; i++)
    {
        inA[i] = i;
        inB[i] = i * 2;
    }

    // Print the inputs to the screen
    printf("Input A: ");
    for (int i = 0; i < arrayLen; i++)
    {
        if (i % size == 0)
            printf("\n\t");

        printf("%d", inA[i]);

        if (i != arrayLen - 1)
            printf(", ");
    }
    printf("\nInput B: ");
    for (int i = 0; i < arrayLen; i++)
    {
        if (i % size == 0)
            printf("\n\t");

        printf("%d", inB[i]);

        if (i != arrayLen - 1)
            printf(", ");
    }

    // Allocate device memory
    int *d_inA;
    int *d_inB;
    int *d_out;
    cudaMalloc((void **)&d_inA, arrayLen * sizeof(int));
    cudaMalloc((void **)&d_inB, arrayLen * sizeof(int));
    cudaMalloc((void **)&d_out, arrayLen * sizeof(int));

    // Copy inputs to gpu
    cudaMemcpy(d_inA, inA, arrayLen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inB, inB, arrayLen * sizeof(int), cudaMemcpyHostToDevice);

    // Define kernel blockSize and gridSize
    dim3 blockSize(1);
    dim3 gridSize(size, size);

    // Call the kernel
    matrixMult<<<gridSize, blockSize>>>(d_inA, d_inB, d_out, size);

    // Wait for the kernel to finish all blocks in the grid
    cudaDeviceSynchronize();

    // Copy output from gpu to host
    cudaMemcpy(output, d_out, arrayLen * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the output to the screen
    printf("\n\nOutput: ");
    for (int i = 0; i < arrayLen; i++)
    {
        if (i % size == 0)
            printf("\n\t");

        printf("%d", output[i]);

        if (i != arrayLen - 1)
            printf(", ");
    }
    printf("\n");

    // Free allocated memory
    cudaFree(d_inA);
    cudaFree(d_inB);
    cudaFree(d_out);
    free(output);

    return 0;
}
