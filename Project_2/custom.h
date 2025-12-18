#ifndef CUSTOM_H
#define CUSTOM_H

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "histogram_matcher.h"

void retrieveImagesWithFaces(const std::vector<std::string> &imagePaths, std::vector<std::string> &resultPaths);
float sumOfSquaredDifference(std::vector<float>& vec1, std::vector<float>& vec2);
void runFigure10Task(std::vector<char *> filenames);
double sumOfSquaredDifference(const cv::Mat &descriptor1, const cv::Mat &descriptor2);
void doSIFT();

#endif