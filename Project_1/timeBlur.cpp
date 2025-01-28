/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Example of how to time an image processing task.

  Program takes a path to an image on the command line
*/

#include <cstdio> // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings
#include <cmath>
#include "filters.h"
#include <chrono>
#include <ctime>
#include "opencv2/opencv.hpp"

// prototypes for the functions to test
//int blur5x5_1( cv::Mat &src, cv::Mat &dst );
//int blur5x5_2( cv::Mat &src, cv::Mat &dst );

// returns a double which gives time in seconds
// https://stackoverflow.com/questions/997946/how-to-get-current-time-and-date-in-c
double getTime() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() / 1e9;
}
  

// argc is # of command line parameters (including program name), argv is the array of strings
// This executable is expecting the name of an image on the command line.

int main(int argc, char *argv[]) {  // main function, execution starts here
  cv::Mat src; // define a Mat data type (matrix/image), allocates a header, image data is null
  cv::Mat dst; // cv::Mat to hold the output of the process
  std::string image_path = cv::samples::findFile("C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_1\\cathedral.jpeg");

  // read the image
  std::cout << "Opening image..." << std::endl;
  cv::Mat image;
  src = cv::imread(image_path, cv::IMREAD_COLOR); // allocating the image data
  dst.create(src.size(), src.type());

  // test if the read was successful
  if(src.data == NULL) {  // src.data is the reference to the image data
    printf("Unable to read image");
    exit(-1);
  }

  const int Ntimes = 10;
	
  //////////////////////////////
  // set up the timing for version 1
  double startTime = getTime();

  // execute the file on the original image a couple of times
  for(int i=0;i<Ntimes;i++) {
    blur5x5_1( src, dst );
  }

  // end the timing
  double endTime = getTime();

  // compute the time per image
  double difference = (endTime - startTime) / Ntimes;

  // print the results
  printf("Time per image (1): %.4lf seconds\n", difference );

  //////////////////////////////
  // set up the timing for version 2
  startTime = getTime();

  // execute the file on the original image a couple of times
  for(int i=0;i<Ntimes;i++) {
    blur5x5_2( src, dst );
  }

  // end the timing
  endTime = getTime();

  // compute the time per image
  difference = (endTime - startTime) / Ntimes;

  // print the results
  printf("Time per image (2): %.4lf seconds\n", difference );
  
  // terminate the program
  printf("Terminating\n");

  return(0);
}
