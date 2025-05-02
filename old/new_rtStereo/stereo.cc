/*
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "imageUtils.h"
#include "matrixUtils.h"
#include "utils.h"

int main(){

const int windowWidth = 11; //must be odd
const int halfWindow = (windowWidth-1)/2;
const int searchWidth = 71; //pixels must be odd
const char* leftBW  = "leftBW.ppm";
const char* rightBW = "rightBW.ppm";
const char* depthImageName = "depth.ppm";
const char* disparityImageName = "disparity.ppm";
double pixelCorr[searchWidth];

PPMImage* leftImg; 
PPMImage* rightImg;

int cols     = 640;
int rows     = 480;
int maxColor = 255;
double baseLine = 60.0;
double focalLength = 560.0;
double maxDisparity = searchWidth;
double minDisparity = 50;
double maxDistance = baseLine * focalLength / minDisparity;
double distance;
double disparity;

//allocate memory for the output images
unsigned char* depthImage = (unsigned char*) malloc(rows*cols * sizeof(unsigned char));
unsigned char* disparityImage = (unsigned char*) malloc(rows*cols * sizeof(unsigned char));

//read images
leftImg   = readPPM(leftBW,0);
rightImg  = readPPM(rightBW,0);

    // search window vars
    unsigned char leftPixel;
    unsigned char rightPixel;
    double leftSquare;
    double rightSquare;
    double product;
    double leftMean;
    double rightMean;
    double expectedProduct;
    double expectedLeftSquare;
    double expectedRightSquare;

// put your code here to do the stereo matching
    for (int row = halfWindow; row < rows - halfWindow; row++)
    {
        for (int col = halfWindow; col < cols - halfWindow; col++)
        {

            // compute the correlation coefficient
            int index = 0;
            int pixels;
            for (int k = -searchWidth; k < 0; k++)
            {

                leftMean = 0.0;
                rightMean = 0.0;
                leftSquare = 0.0;
                rightSquare = 0.0;
                product = 0.0;
                pixels = 0;

                // compute the sums within the windows in each image
                for (int i = -halfWindow; i < halfWindow + 1; i++)
                {
                    for (int j = -halfWindow; j < halfWindow + 1; j++)
                    {

                        if (0 <= col + j && col + j < cols && 0 <= col + j + k && col + j + k < cols)
                        {
                            leftPixel = leftImg->data[(row + i) * cols + (col + j)];
                            rightPixel = rightImg->data[(row + i) * cols + (col + j)];
                            leftMean = leftMean + leftPixel;
                            rightMean = rightMean + rightPixel;
                            product = product + leftPixel * rightPixel;
                            leftSquare = leftSquare + leftPixel * leftPixel;
                            rightSquare = rightSquare + rightPixel * rightPixel;
                            pixels++;
                        }
                    }
                }

                // compute the correlation
                if (pixels > 0)
                {
                    leftMean = leftMean / pixels;
                    rightMean = rightMean / pixels;
                    expectedLeftSquare = leftSquare / pixels;
                    expectedRightSquare = rightSquare / pixels;
                    pixelCorr[index] = (expectedProduct - leftMean * rightMean) /
                                       sqrt((expectedLeftSquare - leftMean * leftMean) * (expectedRightSquare - rightMean * rightMean));
                }
                else
                {
                    pixelCorr[index] = 0.0;
                }
                index++;
            }
            // compute disparity for the window match with the max correlation
            double max = 0;
            for (int k = 0; k < searchWidth; k++)
            {
                if (pixelCorr[k] > max)
                {
                    max = pixelCorr[k];
                    disparity = searchWidth - k;
                }
            }
            // if we have a valid disparity save it to the image file. also compute the distance
            // and save it
            printf("disparity: %d \n", disparity);
            printf("minDisparity: %d \n", minDisparity);
            if (1)
            {
                //distance = baseLine * focalLength / disparity;
                distance = 60 * 560 / 620;
                maxDistance = 60 * 560 / 50;
                printf("baseLine: %d \n focalLength: %d \n", baseLine, focalLength);
                unsigned char pixelValue = 255 * distance / maxDistance;
	            printf("Depth image pixel value: %c \n", pixelValue);
                depthImage[row * cols + col] = pixelValue;
                disparityImage[row * cols + col] = (unsigned char) (255 * minDisparity / disparity);
		        //depthImage[row * cols + col] = (unsigned char)(255 * distance / maxDistance);
            	
	        }
            else
            {
                printf("invalid disparity found...\n");
                depthImage[row * cols + col] = 255;
                disparityImage[row * cols + col] = 255;
            }
        }
    }
    // write the disparity and depth images to files
    writePPM(depthImageName, cols, rows, maxColor, 0, depthImage);
    writePPM(disparityImageName, cols, rows, maxColor, 0, disparityImage);



return 0;
}
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "imageUtils.h"
#include "matrixUtils.h"
#include "utils.h"

int main()
{

    const int windowWidth = 11;
    const int halfWindow = (windowWidth - 1) / 2;
    const int searchWidth = 71;
    const char *leftBW = "leftBW.ppm";
    const char *rightBW = "rightBW.ppm";
    const char *depthImageName = "depth.ppm";
    const char *disparityImageName = "disparity.ppm";
    double pixelCorr[searchWidth];

    PPMImage *leftImg;
    PPMImage *rightImg;

    int cols = 640;
    int rows = 480;
    int maxColor = 255;
    double baseLine = 60.0;
    double focalLength = 560.0;
    double maxDisparity = searchWidth;
    double minDisparity = 50;
    double disparity;
    double maxDistance = baseLine * focalLength / minDisparity;
    double distance;

    unsigned char *depthImage = (unsigned char *)malloc(rows * cols * sizeof(unsigned char));
    unsigned char *disparityImage = (unsigned char *)malloc(rows * cols * sizeof(unsigned char));

    leftImg = readPPM(leftBW, 0);
    rightImg = readPPM(rightBW, 0);

    // search window vars
    unsigned char leftPixel;
    unsigned char rightPixel;
    double leftSquare;
    double rightSquare;
    double product;
    double leftMean;
    double rightMean;
    double expectedProduct;
    double expectedLeftSquare;
    double expectedRightSquare;

    // search the images
    for (int row = halfWindow; row < rows - halfWindow; row++)
    {
        for (int col = halfWindow; col < cols - halfWindow; col++)
        {

            // compute the correlation coefficient
            int index = 0;
            int pixels;
            for (int k = -searchWidth; k < 0; k++)
            {

                leftMean = 0.0;
                rightMean = 0.0;
                leftSquare = 0.0;
                rightSquare = 0.0;
                product = 0.0;
                pixels = 0;

                // compute the sums within the windows in each image
                for (int i = -halfWindow; i < halfWindow + 1; i++)
                {
                    for (int j = -halfWindow; j < halfWindow + 1; j++)
                    {

                        if (0 <= col + j && col + j < cols && 0 <= col + j + k && col + j + k < cols)
                        {
                            leftPixel = leftImg->data[(row + i) * cols + (col + j)];
                            rightPixel = rightImg->data[(row + i) * cols + (col + k + j)];
                            leftMean = leftMean + leftPixel;
                            rightMean = rightMean + rightPixel;
                            product = product + leftPixel * rightPixel;
                            leftSquare = leftSquare + leftPixel * leftPixel;
                            rightSquare = rightSquare + rightPixel * rightPixel;
                            pixels++;
                        }
                    }
                }

                // compute the correlation
                if (pixels > 0)
                {
                    leftMean = leftMean / pixels;
                    rightMean = rightMean / pixels;
                    expectedLeftSquare = leftSquare / pixels;
                    expectedRightSquare = rightSquare / pixels;
                    expectedProduct = product / pixels;
                    pixelCorr[index] = (expectedProduct - leftMean * rightMean) /
                                       sqrt((expectedLeftSquare - leftMean * leftMean) * (expectedRightSquare - rightMean * rightMean));
                }
                else
                {
                    pixelCorr[index] = 0.0;
                }
                index++;
            }
            // compute disparity for the window match with the max correlation
            double max = 0;
            for (int k = 0; k < searchWidth; k++)
            {
                if (pixelCorr[k] > max)
                {
                    max = pixelCorr[k];
                    disparity = searchWidth - k;
                }
            }
            // if we have a valid disparity save it to the image file. also compute the distance
            // and save it
            if (disparity > minDisparity)
            {
                distance = baseLine * focalLength / disparity;
                depthImage[row * cols + col] = (unsigned char)(255 * distance / maxDistance);
                disparityImage[row * cols + col] = (unsigned char)(255 * disparity / maxDisparity);
            }
            else
            {
                depthImage[row * cols + col] = 255;
                disparityImage[row * cols + col] = 255;
            }
        }
    }
    // write the disparity and depth images to files
    writePPM(depthImageName, cols, rows, maxColor, 0, depthImage);
    writePPM(disparityImageName, cols, rows, maxColor, 0, disparityImage);

    return 0;
}