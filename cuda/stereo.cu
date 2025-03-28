// Created By:      Jacob Huckins & Mikey Thoreson
// Last Modified:   2/23/2025

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float vectorDotProduct(float *vA, float *vB, int size)
{
    // declare shared memory variable
    extern __shared__ double out = 0;

    // get thread index
    int i = threadIdx.x;

    // skip out of bounds threads
    if (i >= size)
        return;

    // calulate the partial dot proudct
    float partial = vA[i] * vB[i];

    // sum the partial answers together
    atomicAdd(&out, partial);

    // wait for all threads to finish atomicAdd operation
    cudaDeviceSynchronize();

    // thread 0 returns results
    if (i == 0)
        return out;
}

__device__ float correlationCoefficient(int *l, int *r, int width, int height, int pX, int pY, int rightOffset, int wSize)
{
    int wSize2 = wSize * wSize;

    int left[wSize2];
    int right[wSize2];

    // build left and right frames
    for (int x = -wSize; x < wSize; x++)
    {
        for (int y = -wSize; y < wSize; y++)
        {
            int i = x + y * wSize;

            left[i] = l[(pX + x) + (y + pY) * wSize];
            right[i] = r[(pX + x + offset) + (y + pY) * wSize]
                }
    }

    // common parameters
    float one[] = {1.0};
    dim3 blockSize(1, wSize2);
    dim3 gridSize(1);

    // calc L dot 1
    float Ld1 = vectorDotProduct<<<blockSize, gridSize>>>(&left, &one, wSize2);

    // calc R dot 1
    float Rd1 = vectorDotProduct<<<blockSize, gridSize>>>(&right, &one, wSize2);

    // calc (L dot R) / N
    float LdR = vectorDotProduct<<<blockSize, gridSize>>>(&left, &right, wSize2);

    // calc (L dot L) / N
    float LdL = vectorDotProduct<<<blockSize, gridSize>>>(&left, &left, wSize2);

    // calc (R dot R) / N
    float RdR = vectorDotProduct<<<blockSize, gridSize>>>(&right, &right, wSize2);

    // calculate similarity heuristic

}

__global__ void matchLR(int *l, int *r, int *out, int width, int height, int windowSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = x + y * size;

    // for each pixel
    // get a window around it
    // compute the correlation coefficient of a sliding the window
    // find the offset that maximizes correlation
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
