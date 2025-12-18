#ifndef TEXTURE_H
#define TEXTURE_H

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

float sumOfSquaredDifference(std::vector<float>& vec1, std::vector<float>& vec2);

void computeTextureHistogram(cv::Mat &src, cv::Mat &dst);

int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

#endif