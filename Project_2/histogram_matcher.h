#ifndef HISTOGRAM_MATCHER_H
#define HISTOGRAM_MATCHER_H

#include <vector>
#include <iostream>

double histogramIntersection(const cv::Mat &histA, const cv::Mat &histB);
void computeRGChromaticityHistogram(const cv::Mat &image, cv::Mat &hist, int r_bins, int g_bins);
void computeRGBChromaticityHistogram(const cv::Mat &image, cv::Mat &hist, int r_bins, int g_bins, int b_bins);
void computeSingleHistogramMatch();
void computeMultiHistogramMatch();
cv::Mat getCenter(const cv::Mat &image, int size);
double computeWeightedDistance(const cv::Mat &wholeHistT, const cv::Mat &centerHistT, const cv::Mat &wholeHist, const cv::Mat &centerHist, double wholeWeight, double centerWeight);

#endif