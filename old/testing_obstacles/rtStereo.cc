#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "stereoDepth.h"
#include <time.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {


int fps = 10; // in frames per sec
int frameDelay = 1000/(10*fps); // in millisec 
double maxDistance = 5000.0; // mm
int rows  = 480;
int cols  = 640;
Mat depthImage = Mat::zeros(rows,cols, CV_8UC1);

clock_t start;
clock_t stop;
double elapsedTime = 0.0;

bool loop=true;


//Read rectification lookup tables
Mat map1x,map1y,map2x,map2y;
FileStorage fs("lookupTables.xml",FileStorage::READ);
fs["Map1x"]>>map1x;
fs["Map1y"]>>map1y;
fs["Map2x"]>>map2x;
fs["Map2y"]>>map2y;
fs.release();

if( map1x.empty()) cout << "Empty 1x lookup table"<<endl;
if( map1y.empty()) cout << "Empty 1y lookup table"<<endl;
if( map2x.empty()) cout << "Empty 2x lookup table"<<endl;
if( map2y.empty()) cout << "Empty 2y lookup table"<<endl;


float offset = 7.0;
   for (int y=0; y<rows;y++){
      for(int x=0;x<cols;x++){
  
        if(y+offset<0)
            map2y.at<float>(y,x) = 0.0;
        else if(y+offset>rows)
            map2y.at<float>(y,x) = (float)rows;
        else
            map2y.at<float>(y,x) = (float)y + offset;
  
      }
 }


// GStreamer pipeline for Jetson Nano with IMX219-83 cameras
 string left_cam_pipeline  = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, framerate="+to_string(fps)+"/1 ! nvvidconv flip-method=rotate-180 ! video/x-raw, format=GRAY8 !  appsink drop=true sync=true max-buffers=1";


 string right_cam_pipeline = "nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, framerate="+to_string(fps)+"/1 ! nvvidconv flip-method=rotate-180 ! video/x-raw, format=GRAY8 !  appsink drop=true sync=true max-buffers=1";
    
// Open both cameras
    VideoCapture capL(left_cam_pipeline, CAP_GSTREAMER);
    VideoCapture capR(right_cam_pipeline,CAP_GSTREAMER);


    if (!capL.isOpened() || !capR.isOpened()) {
        cerr << "Error: Could not open stereo cameras." << endl;
        return -1;
    }


    Mat leftFrame, rightFrame;

    cout << " width \n" << capL.get(CAP_PROP_FRAME_WIDTH)<<endl;
    cout << " height  \n" << capL.get(CAP_PROP_FRAME_HEIGHT)<<endl;
    cout << " format \n" << capL.get(CAP_PROP_FORMAT)<<endl;
    cout << " fps \n" << capL.get(CAP_PROP_FPS)<<endl;

   start = clock();
   while((clock() - start) < 5 * CLOCKS_PER_SEC){}
    
    // Real-time loop for capturing frames
    while (loop) {

        // Capture frames from both cameras
        capL >> leftFrame;
        capR >> rightFrame;

        if (leftFrame.empty() || rightFrame.empty()) {
            cerr << "Error: Empty frame detected!" << endl;
            break;
        }



      // Apply rectification
      Mat rectifiedLeft, rectifiedRight, both;
      remap(leftFrame, rectifiedLeft, map1x, map1y, INTER_LINEAR);
      remap(rightFrame, rectifiedRight, map2x, map2y, INTER_LINEAR);


	//start = clock();
      // Compute depth image using GPU
      stereoDepth(&rectifiedLeft, &rectifiedRight, &depthImage, maxDistance, rows, cols);
	//stop = clock();
	//elapsedTime = (double) (stop-start)/CLOCKS_PER_SEC;
	//printf("Elapsed time:  %f \n", elapsedTime);
	
     // Smooth the depth image
     Mat medianFiltered;
     medianBlur(depthImage, medianFiltered, 3);

      // display depth map
      imshow("Depth", medianFiltered);
      hconcat(rectifiedLeft, rectifiedRight,both);
      imshow("Left and Right",both);
  



	// build obstacle map
	Mat obstacles;
	obstacles = medianFiltered;

	double threshold_dist = 200;
	
	 for (int i = 0; i < rows; ++i) {
	         for (int j = 0; j < cols; ++j) {
     			 uchar curr_pixel = obstacles.at<uchar>(i,j);
			if(curr_pixel > threshold_dist) curr_pixel = 255;
			else curr_pixel = 0;
			obstacles.at<uchar>(i,j) = curr_pixel;
		 }
	 }
	imshow("obstacles", obstacles);	
	// end obstacle map
	
	//detection
	int left_window = cols/3;
	int center_window = 2*cols/3;
	int right_window = cols;

	//add values in center window
	int center_sum = 0;
	for(int i=(left_window + 1); i < right_window+1;i++){
		for(int j = 0; j < rows; ++j){
		 uchar curr_pixel = obstacles.at<uchar>(i,j);
		 center_sum+= curr_pixel;}
	}


	//----
	int obstacle_sum_threshold = 510;
	bool is_clear = true;
	if(center_sum > obstacle_sum_threshold){
		is_clear = false;}
	
	// STEER COMMANDS GO HERE



      // pause
      char key = waitKey(frameDelay);
      if (key == 'q' || key == 27) break;
    }

    // Release resources
    capL.release();
    capR.release();
    destroyAllWindows();

    return 0;
}

