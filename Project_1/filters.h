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

int sepia(cv::Mat &src, cv::Mat &dst);

int blur5x5_1( cv::Mat &src, cv::Mat &dst );
int blur5x5_2( cv::Mat &src, cv::Mat &dst );
int vert_blur( cv::Mat &src, cv::Mat &dst );
int horiz_blur( cv::Mat &src, cv::Mat &dst );

int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

// prototypes extra
int keepOneColor( cv::Mat &src, cv::Mat &dst );
int vignetting( cv::Mat &src, cv::Mat &dst, int levels );

#endif // FILTERS_H
