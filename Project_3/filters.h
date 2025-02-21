/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  Include file for filters.cpp, filter functions
*/

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

// prototypes required
int greyscale(cv::Mat &src, cv::Mat &dst);

int blur5x5_1( cv::Mat &src, cv::Mat &dst );
int blur5x5_2( cv::Mat &src, cv::Mat &dst );
int vert_blur( cv::Mat &src, cv::Mat &dst );
int horiz_blur( cv::Mat &src, cv::Mat &dst );

// project 3 2D object recognition
int threshold( cv::Mat &src, cv::Mat &dst );
int morphological( cv::Mat &src, cv::Mat &dst );
int connectedComponentAnaylsis( cv::Mat &src, cv::Mat &dst );
int computeFeaturesForRegions();
int evaluatePerformance();
int depthAnything( cv::Mat &src, cv::Mat &dst );

#endif // FILTERS_H
