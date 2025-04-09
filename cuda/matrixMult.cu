// Cmeated By:      Jacob Huckins & Mikey Thoreson
// Last Modified:   03/04/2025

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <jpeglib.h>

const int WINDOW_DIM = 4;
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
				__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed!= old);
	return __longlong_as_double(old);
}

__device__ static bool doubleAtomicMax(double* address, double val, int* disparityIn, int colIn, int rowIn, int offsetIn)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
	bool writeflag = false;
	assumed = old;
	old = atomicCAS(address_as_ull, assumed,
					__double_as_longlong(fmaxf(val, __longlong_as_double(assumed))));
    while (assumed != old)
		{
		disparityIn[colIn + (rowIn * PIC_WIDTH)] = offsetIn;
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmaxf(val, __longlong_as_double(assumed))));
		writeflag = 1;
		};
    //return __longlong_as_double(old);
    return writeflag;
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

__global__ void correlationCoefficient(int *r, int *l, double *corrCoefOut, int *corrCoefIdx)
{
	int col; col = threadIdx.x + (blockDim.x * blockIdx.x);
	int row; row = threadIdx.y + (blockDim.y * blockIdx.y);
	int offs; offs = threadIdx.z + (blockDim.z * blockIdx.z);

	if(col + offs <= PIC_WIDTH){
		if(!(col - (WINDOW_DIM / 2)  < 0) && !(col + (WINDOW_DIM / 2) >= PIC_WIDTH)){
			if(!(row - (WINDOW_DIM / 2)  < 0) && !(row + (WINDOW_DIM / 2) >= PIC_HEIGHT))
			{
				double N = WINDOW_DIM * WINDOW_DIM;

				// calc L dot 1
				double Ld1;
				windowSum(l, col, row, Ld1);

				// calc R dot 1
				double Rd1;
				windowSum(r, col + offs, row, Rd1);

				// calc (L dot R) / N
				double LdR;
				winDotProduct(l, r, col, row, offs, LdR);

				// calc (L dot L) / N
				double LdL;
				winDotProduct(l, l, col, row, 0, LdL);

				// calc (R dot R) / N
				double RdR;
				winDotProduct(r, r, offs+col, row, 0, RdR);
				//winDotProduct(r, r, 40+y, 40, 0, LdR);

				// calculate correlation coefficient
				// [n(X.Y) - (X.1)(Y.1)] / [(n(X.X) - X.1)(n(Y.Y - Y.1))]
				//double top = ((N) * LdR) - (Ld1 * Rd1);
				double top = (LdR/N) - ((Ld1/N)*(Rd1/N));
				double bot = sqrtf(
					(LdL/N) - ((Ld1/N)*(Ld1/N))
					) *
				sqrtf(
						(RdR/N) - ((Rd1/N)*(Rd1/N))
				     );
			double corrCoef = (top/bot);
			//int old = corrCoefIdx[col + (row * PIC_WIDTH)];
			//int assumed;
			// if this thread results in a new update to max corrCoeff:
			if(doubleAtomicMax(&corrCoefOut[col + (row * PIC_WIDTH)], corrCoef, corrCoefIdx, col, row, offs));
			{
				//wait for the threads to execute
				// if this thread's corrCoef is still equal to the maximum
				//if(labs(corrCoefOut[col + (row * PIC_WIDTH)] - corrCoef) < 0.000001){
					//loop trying to get a lock on the disparity graph so we can write our value to it
					//do{
					//	assumed = old;
					//	old = atomicCAS(&corrCoefIdx[col + (row * PIC_WIDTH)],
					//					assumed,
					//					offs);
					//}
					//while(assumed != old);
				//}
			}
		}
	}
	}
}
//image utilities

//PPMImage* readPPM(const char* filename, int type) {
//	FILE* file = fopen(filename, "rb");
//	int numBytes;
//
//	if (!file) {
//		perror("Error opening file");
//		exit(0);
//	}
//
//	if(type==0) numBytes=1; //grayscale

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
	int* h_CorrCoefIdxMtx = (int*) malloc(sizeof(int) * PIC_WIDTH * PIC_HEIGHT);

	int imgfield;
	srand(12);
	for(int x = 0; x < PIC_WIDTH; x++){
		for(int y = 0; y < PIC_HEIGHT; y++){
			imgfield = (int) rand() % 255;
			rightimg[x + (y * PIC_WIDTH)] = imgfield;
			leftimg[((x + 10) % PIC_WIDTH ) + (y * PIC_WIDTH)] = rightimg[x + (y * PIC_WIDTH)];
		}
	}

	int picSize = PIC_WIDTH * PIC_HEIGHT;

	int * d_leftimg;
	cudaMalloc(&d_leftimg, sizeof(int) * picSize);
	int * d_rightimg;
	cudaMalloc(&d_rightimg, sizeof(int) * picSize);
	double * d_CorrCoefMtx;
	cudaMalloc(&d_CorrCoefMtx, sizeof(double) * picSize);
	int * d_CorrCoefIdxMtx;
	cudaMalloc(&d_CorrCoefIdxMtx, sizeof(int) * picSize);

	// setup the 2d matrices that will hold the result of our matrix
	// mult operations, for each combination of pixels on each pixel

	cudaMemcpy(d_leftimg, leftimg, sizeof(int) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rightimg, rightimg, sizeof(int) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_CorrCoefMtx, h_CorrCoefMtx, sizeof(double) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_CorrCoefMtx, h_CorrCoefIdxMtx, sizeof(int) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyHostToDevice);

	dim3 threadCount(8, 8, 8);
	dim3 blockCount((PIC_HEIGHT/8) + 1,(PIC_WIDTH/8) + 1, (PIC_WIDTH/8) + 1);
	correlationCoefficient<<<blockCount, threadCount>>>(d_leftimg, d_rightimg, d_CorrCoefMtx, d_CorrCoefIdxMtx);
	cudaDeviceSynchronize();

	cudaMemcpy(h_CorrCoefMtx, d_CorrCoefMtx, sizeof(double) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_CorrCoefIdxMtx, d_CorrCoefIdxMtx, sizeof(int) * PIC_WIDTH * PIC_HEIGHT, cudaMemcpyDeviceToHost);


	for(int y = 0; y < PIC_WIDTH; y++){
		for(int x = 0; x < PIC_HEIGHT; x++){
			printf("%.3f  ", h_CorrCoefMtx[x + (y*PIC_WIDTH)]);
			//printf("%03d  ", leftimg[x + (y*PIC_WIDTH)]);
			//printf("%03d ", h_CorrCoefIdxMtx[i + (j * PIC_WIDTH)]);
		}
		printf("\n");
	}
	printf("\n\n\n");
	for(int y = 0; y < PIC_WIDTH; y++){
		for(int x = 0; x < PIC_HEIGHT; x++){
			//printf("%.3f  ", h_CorrCoefMtx[x + (y*PIC_WIDTH)]);
			//printf("%03d  ", rightimg[x + (y*PIC_WIDTH)]);
			printf("%03d ", h_CorrCoefIdxMtx[x + (y * PIC_WIDTH)]);
		}
		printf("\n");
	}

	cudaFree(&d_leftimg);
	cudaFree(&d_rightimg);
	cudaFree(&d_CorrCoefMtx);
	cudaFree(&d_CorrCoefIdxMtx);
	return 0;
}
