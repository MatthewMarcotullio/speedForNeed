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

void vectorPrint(float* u , int rows){
  for(int i=0;i<rows;i++){
    printf( "%7.1f \n",u[i]);
  }
  printf("\n");

}

void vectorScale(float* u, int rows, float alpha , float* v){
  for(int i=0; i<rows; i++){
    v[i]=alpha*u[i];
  }
}

float vectorDotProduct(float* u, float* v, int u_rows, int v_rows){
  if(u_rows == v_rows){
    float dot_product = 0;
    for(int i=0; i<u_rows; i++){
      dot_product += u[i]*v[i];
  }
  return dot_product;
  }
  else{
    printf("vectorDotProduct ERROR: Both vectors must have the same number of rows\n");
  }
  return 1;
}


void vectorSubtract(float* u, float* v, int u_rows, int v_rows, float* output_vect){
  if(u_rows == v_rows){
    float* output_vect = (float*)malloc(u_rows * sizeof(float));
    for(int i=0; i<u_rows; i++){
      float difference = u[i] - v[i];
      //cout << u[i] << " - " << v[i] << " = " << difference << endl;
      output_vect[i] = difference;
  }
  }
  else{
    printf("vectorSubtract ERROR: Both vectors must have the same number of rows\n");
  }
}

float vectorMag(float* u, int rows){
  float sum=0;
  float magnitude=0;
  for(int i=0; i<rows; i++){
    sum += u[i]*u[i];
  }
  magnitude = sqrt(sum);
  return magnitude;
}

float* vectorNorm(float* u, int rows){
  float* normalized_vect = (float*)malloc(rows * sizeof(float));
  float mag = vectorMag(u, rows);
  for(int i=0; i<rows; i++){ 
    float quotient = u[i]/mag;
    normalized_vect[i] = quotient;
  }
  return normalized_vect;
}
