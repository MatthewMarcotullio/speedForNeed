#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <limits.h>

__global__ void stereoKernel(unsigned char* left, unsigned char* right, unsigned char* depth,
                             double maxDistance, int rows, int cols){


// compute the row and col of the pixel to be processed
int col = blockIdx.x*blockDim.x + threadIdx.x;
int row = blockIdx.y*blockDim.y + threadIdx.y;

// put your stereo matching code here
// This code should only be for one pixel
// See the video I posted on acceleration stereo on the GPU

const int windowWidth = 9;
const int halfWindow = (windowWidth -1) / 2;
double baseLine = 60.0;
double focalLength = 560.0;
double maxDisparity = 61;

unsigned char leftPixel;
unsigned char rightPixel;
int disparity;
double distance;
int sumSqDiff;
int minSumSqDiff = INT_MAX;
int diff=0;

//make sure the pixel is with the image border
if (row<halfWindow || row>rows-halfWindow || col<halfWindow || col>cols-halfWindow) return;

//compute the sumSqDiff of each shifted window
for(int k=0; k<maxDisparity;k++){
	sumSqDiff = 0;

	//compute the sume w/i the windows in each image
	for(int i = -halfWindow; i<halfWindow+1; i++){
		for(int j = -halfWindow; j<halfWindow+1; j++){
			leftPixel = left[(row+i)*cols+(col+j)];
			rightPixel = right[(row+i)*cols+(col+j-k)];
			diff = leftPixel - rightPixel;
			sumSqDiff += diff*diff;
		}
		
	}
	//compute min sumSqDiff
	if(sumSqDiff < minSumSqDiff){
		minSumSqDiff = sumSqDiff;
		disparity = k;	
	}
}
//if valid disp, compute dist and save
if (disparity > 0 || sumSqDiff < 1){
	distance = baseLine*focalLength/disparity;
	if (distance < maxDistance)
	depth[row*cols+col] = (unsigned char) (255.0*distance/maxDistance);	
	else depth[row*cols+col] = 255;
}

else{
	depth[row*cols+col]=255;
}

}
