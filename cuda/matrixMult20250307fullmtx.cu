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

//__global__ void bake
//
//__device__ void correlationCoefficient(int **l_win, int **r_win, int mtx_size, double *out)
//{
//    int x = threadIdx.x + blockDim.x * blockIdx.x;
//    int y = threadIdx.y + blockDim.y * blockIdx.y;
//    int z = threadIdx.z + blockDim.z * blockIdx.z;
//
//    // calc L dot 1
//    double Ld1;
//
//
//    // calc R dot 1
//    double Rd1;
//    sumVect<<<1, mtx_size>>>(r_win, mtx_size, &Rd1);
//
//    // calc (L dot R) / N
//    double LdR;
//    vectorDotProduct<<<1, mtx_size>>>(l_win, r_win, mtx_size, &LdR);
//
//    // calc (L dot L) / N
//    double LdL;
//    vectorDotProduct<<<1, mtx_size>>>(l_win, l_win, mtx_size, &LdL);
//
//    // calc (R dot R) / N
//    double RdR;
//    vectorDotProduct<<<1, mtx_size>>>(r_win, r_win, mtx_size, &RdR);
//
//    // calculate correlation coefficient
//    // [n(X.Y) - (X.1)(Y.1)] / [(n(X.X) - X.1)(n(Y.Y - Y.1))]
//    double top = mtx_size * LdR - Ld1 * Rd1;
//    double bot = (mtx_size * LdL - Ld1) * (mtx_size * (RdR - Rd1));
//
//    double corCoef = top / bot;
//
//    *out = corCoef;
//}
//
__global__ void helloWorld (int *l, int *r){
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;

}
__device__ void mtxMult(int* l, int* r, int x, int y, int offset, double *result){
    int l_idx;
    int r_idx;
  for(int i = 0; i < WINDOW_DIM * WINDOW_DIM; i++){
      l_idx = (i / WINDOW_DIM + l_idx + ((i % WINDOW_DIM) *PIC_WIDTH));
      r_idx = (i / WINDOW_DIM + r_idx + offset + ((i % WINDOW_DIM) *PIC_WIDTH));
      result+=l[l_idx] * r[r_idx];
  } 
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

    int dims_3d_mtx = PIC_WIDTH * PIC_HEIGHT * PIC_WIDTH;
    size_t size_double_mtx = sizeof(double) *dims_3d_mtx;

    double* h_LL;
    cudaMalloc(&h_LL, size_double_mtx);
    double* h_RR;
    cudaMalloc(&h_RR, size_double_mtx);
    double* h_RL;
    cudaMalloc(&h_RL, size_double_mtx);
    double* h_LR;
    cudaMalloc(&h_LR, size_double_mtx);
    double* h_CorrCoef;
    cudaMalloc(&h_CorrCoef, size_double_mtx);
    double* h_distanceCalc;
    cudaMalloc(&h_distanceCalc, size_double_mtx);


    //get 3d matrix of windows
    //get 2d matrix of max
    cudaFree(&h_LL);
    cudaFree(&h_RR);
    cudaFree(&h_RL);
    cudaFree(&h_LR);
    cudaFree(&h_CorrCoef);
    cudaFree(&h_distanceCalc);
    return 0;
}
