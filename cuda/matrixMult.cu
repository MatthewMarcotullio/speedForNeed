// Cmeated By:      Jacob Huckins & Mikey Thoreson
// Last Modified:   03/04/2025

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int WINDOW_DIM = 9;
const int PIC_WIDTH = 640;
const int PIC_HEIGHT = 480;

// implementation taken from pg 109 of nvidia cuda C programming guide ver 4.2
__device__ double doubleAtomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull=
						(unsigned long long int*) address;
	unsigned long long int old=*address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));
	} while (assumed!= old);
	return __longlong_as_double(old);
}

__device__ void winDotProduct(int *l_pic, int *r_pic, int x_center, int y_center, int offset, double &retVal){
	int half_win = (WINDOW_DIM / 2);
	int l_idx = (x_center - half_win) + ((y_center * PIC_WIDTH) - half_win);
	int r_idx = (x_center + offset - half_win) + ((y_center * PIC_WIDTH) - half_win);
	int l_idx_win = 0;
	int r_idx_win = 0;

	retVal = 0;

	for(int x = 0; x < WINDOW_DIM * WINDOW_DIM; x++){
		for(int y = 0; y < WINDOW_DIM * WINDOW_DIM; y++){
		l_idx_win = l_idx + (x + (y * WINDOW_DIM));
		r_idx_win = r_idx + (x + (y * WINDOW_DIM));
		retVal += l_pic[l_idx_win] * r_pic[r_idx_win];
		}
	}
}

__device__ void windowSum(int *pic, int x_center, int y_center, double &retVal){
	int half_win = WINDOW_DIM / 2;
	int idx = (x_center - half_win) + ((y_center * PIC_WIDTH) - half_win);
	int idx_win = 0;

	retVal = 0;

	for(int x= 0; x < WINDOW_DIM * WINDOW_DIM; x++){
		for(int y = 0; y < WINDOW_DIM * WINDOW_DIM; y++){
		idx_win = idx + (x + (y * WINDOW_DIM));
		retVal += pic[idx_win];
		}
	}
}

__global__ void correlationCoefficient(int *l, int *r, int row, double *out)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // calc L dot 1
    double Ld1;
    windowSum(l, x, row, Ld1);

    // calc R dot 1
    double Rd1;
    windowSum(r, x + y, row, Rd1);


    // calc (L dot R) / N
    double LdR;
    winDotProduct(l, r, x, row, y, LdR);
    // calc (L dot L) / N
    double LdL;
    winDotProduct(l, l, x, row, 0, LdR);

    // calc (R dot R) / N
    double RdR;
    winDotProduct(r, r, x+y, row, 0, LdR);

    // calculate correlation coefficient
    // [n(X.Y) - (X.1)(Y.1)] / [(n(X.X) - X.1)(n(Y.Y - Y.1))]
    double top = (WINDOW_DIM * WINDOW_DIM) * LdR - Ld1 * Rd1;
    double bot = ((WINDOW_DIM * WINDOW_DIM) * LdL - Ld1) * ((WINDOW_DIM * WINDOW_DIM) * (RdR - Rd1));

    double corCoef = top / bot;

    out[x + (y*PIC_WIDTH)] = corCoef;
}

int main()
{
//    my reference code
//    int* helloMtx = (int*) malloc(sizeof(int) * 9);
//    int* d_helloMtx;
//    size_t size = 9 * sizeof(int);
//
//    cudaMalloc(&d_helloMtx, 9 * sizeof(int));
//    helloWorld<<<1,9>>>(d_helloMtx);
//    cudaMemcpy(helloMtx, d_helloMtx, size, cudaMemcpyDeviceToHost);
//    cudaFree(&d_helloMtx);
//    for(int i = 0; i < 9; i++){
//        printf("%d", helloMtx[i]);
//    }

    int* leftmtx = (int*) malloc(sizeof(int) * PIC_WIDTH * PIC_HEIGHT);
    int* rightmtx = (int*) malloc(sizeof(int) * PIC_WIDTH * PIC_HEIGHT);
    double* h_LR = (double*) malloc(sizeof(double) * PIC_WIDTH * PIC_HEIGHT);

    for(int i = 0; i < PIC_WIDTH * PIC_HEIGHT; i++){
        leftmtx[i] = i;
        rightmtx[i] = i;
    }

    int dims_2d_mtx = PIC_WIDTH * PIC_HEIGHT;
    size_t size_double_mtx = sizeof(double) *dims_2d_mtx;

    int * d_leftmtx;
    cudaMalloc(&d_leftmtx, sizeof(int) * PIC_WIDTH * PIC_HEIGHT);
    int * d_rightmtx;
    cudaMalloc(&d_rightmtx, sizeof(int) * PIC_WIDTH * PIC_HEIGHT);
    double * d_CorrCoef;
    cudaMalloc(&d_CorrCoef, sizeof(double) * PIC_WIDTH * PIC_HEIGHT);
    // setup the 2d matrices that will hold the result of our matrix
    // mult operations, for each combination of pixels on each pixel

    cudaMemcpy(d_leftmtx, leftmtx, sizeof(int) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rightmtx, rightmtx, sizeof(int) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_LR, h_LR, sizeof(double) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyDeviceToHost);

    for(int i = 0; i < PIC_WIDTH * PIC_HEIGHT; i++){
        printf("%f", h_LR[i]);
    }

    cudaFree(&d_leftmtx);
    cudaFree(&d_rightmtx);
    cudaFree(&d_CorrCoef);
    cudaFree(&d_distanceCalc);
    return 0;
}
