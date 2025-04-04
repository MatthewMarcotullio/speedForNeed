// Cmeated By:      Jacob Huckins & Mikey Thoreson
// Last Modified:   03/04/2025

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <jpeglib.h>

const int WINDOW_DIM = 9;
const int PIC_WIDTH = 32;
const int PIC_HEIGHT = 32;

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
	int l_idx = (x_center - half_win) + ((y_center - half_win) * PIC_WIDTH);
	int r_idx = (x_center + offset - half_win) + ((y_center  - half_win) * PIC_WIDTH);
	int l_idx_win = 0;
	int r_idx_win = 0; 
	retVal = 0;

	for(int x = 0; x < WINDOW_DIM; x++){
		for(int y = 0; y < WINDOW_DIM; y++){
			l_idx_win = l_idx + (x + (y * PIC_WIDTH));
			r_idx_win = r_idx + (x + (y * PIC_WIDTH));
			retVal += l_pic[l_idx_win] * r_pic[r_idx_win];
		}
	}
}

__device__ void windowSum(int *pic, int x_center, int y_center, double &retVal){
	int half_win = WINDOW_DIM / 2;
	int idx = (x_center - half_win) + ((y_center - half_win) * PIC_WIDTH);
	int idx_win = 0;

	retVal = 0;

	for(int x= 0; x < WINDOW_DIM; x++){
		for(int y = 0; y < WINDOW_DIM; y++){
			idx_win = idx + (x + (y * PIC_WIDTH));
			retVal += pic[idx_win];
		}
	}
}

__global__ void correlationCoefficient(int *l, int *r, int row, double *out)
{
	int x; x = threadIdx.x + (blockDim.x * blockIdx.x);
	int y; y = threadIdx.y + (blockDim.y * blockIdx.y);

	if(!(x - (WINDOW_DIM / 2) < 0) && !(x + (WINDOW_DIM / 2) >= PIC_WIDTH)){
		if(!(y - (WINDOW_DIM / 2) < 0) && !(y + (WINDOW_DIM / 2) >= PIC_HEIGHT)){
			double N = WINDOW_DIM * WINDOW_DIM;

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
			winDotProduct(l, l, x, row, 0, LdL);

			// calc (R dot R) / N
			double RdR;
			winDotProduct(r, r, x+y, row, 0, RdR);
			//winDotProduct(r, r, 40+y, 40, 0, LdR);

			// calculate correlation coefficient
			// [n(X.Y) - (X.1)(Y.1)] / [(n(X.X) - X.1)(n(Y.Y - Y.1))]
			//double top = ((N) * LdR) - (Ld1 * Rd1);
			double top = (LdR/N) - ((Ld1/N)*(Rd1/N));
			//double bot = ((N * LdL) - Ld1) * (N * (RdR - Rd1));
			double bot = sqrtf(
					(LdL/N) - ((Ld1/N)*(Ld1/N))
					) *
				sqrtf(
						(RdR/N) - ((Rd1/N)*(Rd1/N))
				     );

			__syncthreads();
			out[x + (y*PIC_WIDTH)] = (top/bot);
		}
	}
	else
	{
		out[x + (y*PIC_WIDTH)] = 0.0;
	}
}
//image utilities

PPMImage* readPPM(const char* filename, int type) {
	FILE* file = fopen(filename, "rb");
	int numBytes;

	if (!file) {
		perror("Error opening file");
		exit(0);
	}

	if(type==0) numBytes=1; //grayscale

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

	int* leftimg = (int*) malloc(sizeof(int) * PIC_WIDTH * PIC_HEIGHT);
	int* rightimg = (int*) malloc(sizeof(int) * PIC_WIDTH * PIC_HEIGHT);
	double* h_CorrCoefMtx = (double*) malloc(sizeof(double) * PIC_WIDTH * PIC_HEIGHT); 
	for(int i = 0; i < PIC_WIDTH * PIC_HEIGHT; i++){
			leftimg[i] = i;
			rightimg[i] = (i*i)%255;
			h_CorrCoefMtx[i] = 0.0;
	}

	int picSize = PIC_WIDTH * PIC_HEIGHT;

	int * d_leftimg;
	cudaMalloc(&d_leftimg, sizeof(int) * picSize);
	int * d_rightimg;
	cudaMalloc(&d_rightimg, sizeof(int) * picSize);
	double * d_CC;
	cudaMalloc(&d_CC, sizeof(double) * picSize);

	// setup the 2d matrices that will hold the result of our matrix
	// mult operations, for each combination of pixels on each pixel

	cudaMemcpy(d_leftimg, leftimg, sizeof(int) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rightimg, rightimg, sizeof(int) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_CC, h_CorrCoefMtx, sizeof(double) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyHostToDevice);

	dim3 threadCount(32,32);
	dim3 blockCount(1,1);
	correlationCoefficient<<<blockCount, threadCount>>>(d_leftimg, d_rightimg, 20, d_CC);

	cudaMemcpy(h_CorrCoefMtx, d_CC, sizeof(double) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyDeviceToHost);


	for(int i = 0; i < PIC_WIDTH; i++){
		for(int j = 0; j < PIC_HEIGHT; j++){
			printf("%f ", h_CorrCoefMtx[i + (j*PIC_WIDTH)]);
		}
		printf("\n");
	}

	cudaFree(&d_leftimg);
	cudaFree(&d_rightimg);
	cudaFree(&d_CC);
	return 0;
}
