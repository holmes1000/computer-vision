/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  This program takes a target image and a single color histogram to generate 
  the feature vector (uses histogram intersection as distance metric), 
  and identifies the top N matches.
*/

#include <iostream>
#include <vector>
#include "csv_util.h"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "histogram_matcher.h"
#include <filesystem>

/*
  Function to computer the RG chromaticity histogram
  Arguments:
  const cv::Mat& image
  cv::Mat& hist
  int r_bins = 16
  int g_bins = 16
*/
void computeRGChromaticityHistogram(const cv::Mat& image, cv::Mat& hist, int r_bins = 16, int g_bins = 16)
{
    // Initialize the chromaticity image
    cv::Mat chromaticity(image.size(), CV_32FC2);

    // Convert the image to RG chromaticity space
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            float r = pixel[2];
            float g = pixel[1];
            float b = pixel[0];
            float sum = r + g + b;
            if (sum > 0) {
                chromaticity.at<cv::Vec2f>(i, j)[0] = r / sum; // r chromaticity
                chromaticity.at<cv::Vec2f>(i, j)[1] = g / sum; // g chromaticity
            } else {
                chromaticity.at<cv::Vec2f>(i, j) = cv::Vec2f(0, 0);
            }
        }
    }

    // Set the ranges and bins for the histogram
    int histSize[] = {r_bins, g_bins};
    float r_ranges[] = {0, 1};  // r chromaticity range
    float g_ranges[] = {0, 1};  // g chromaticity range
    const float* ranges[] = {r_ranges, g_ranges};
    int channels[] = {0, 1};

    // Compute the histogram
    // std::cout << "Running calcHist" << std::endl;
    cv::calcHist(&chromaticity, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);

    // Normalize the histogram to have a sum of 1
    // std::cout << "Normalizing" << std::endl;
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
}

/*
  Function to computer the RGB chromaticity histogram
  Arguments:
  const cv::Mat& image
  cv::Mat& hist
  int r_bins = 16
  int g_bins = 16
  int b_bins = 16
*/
void computeRGBChromaticityHistogram(const cv::Mat& image, cv::Mat& hist, int r_bins = 16, int g_bins = 16, int b_bins = 16)
{
    // Initialize the chromaticity image
    cv::Mat chromaticity(image.size(), CV_32FC3);

    // Convert the image to RG chromaticity space
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            float r = pixel[2];
            float g = pixel[1];
            float b = pixel[0];
            float sum = r + g + b;
            if (sum > 0) {
                chromaticity.at<cv::Vec3f>(i, j)[0] = r / sum; // r chromaticity
                chromaticity.at<cv::Vec3f>(i, j)[1] = g / sum; // g chromaticity
                chromaticity.at<cv::Vec3f>(i, j)[2] = b / sum; // b chromaticity
            } else {
                chromaticity.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0);
            }
        }
    }

    // Set the ranges and bins for the histogram
    int histSize[] = {r_bins, g_bins, b_bins};
    float r_ranges[] = {0, 1};  // r chromaticity range
    float g_ranges[] = {0, 1};  // g chromaticity range
    float b_ranges[] = {0, 1};  // b chromaticity range
    const float* ranges[] = {r_ranges, g_ranges, b_ranges};
    int channels[] = {0, 1, 2};

    // Compute the histogram
    // std::cout << "Running calcHist" << std::endl;
    cv::calcHist(&chromaticity, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);

    // Normalize the histogram to have a sum of 1
    // std::cout << "Normalizing" << std::endl;
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
}

double histogramIntersection(const cv::Mat& histA, const cv::Mat& histB) {
    double intersection = 0.0;
    for (int i = 0; i < histA.rows; ++i) {
        for (int j = 0; j < histA.cols; ++j) {
            intersection += std::min(histA.at<float>(i, j), histB.at<float>(i, j));
        }
    }
    return intersection;
}

/*
  Function to return the center of an image of a certain size
  Arguments:
  const cv::Mat &image
  int size
*/
cv::Mat getCenter(const cv::Mat &image, int size) 
{
  int x = (image.cols - size) / 2;
  int y = (image.rows - size) / 2;
  return image(cv::Rect(x, y, size, size));
}

/*
  Function to compute the weighted distance between two feature vectors
*/
double computeWeightedDistance(const cv::Mat &wholeHistT, const cv::Mat &centerHistT, const cv::Mat &wholeHist, const cv::Mat &centerHist, double wholeWeight = 1, double centerWeight = 1) {
    double wholeDistance = histogramIntersection(wholeHistT, wholeHist);
    double centerDistance = histogramIntersection(centerHistT, centerHist);
    return (wholeWeight * wholeDistance) + (centerWeight * centerDistance);
}

/*
  Function to run the single histogram matcher
*/
void computeSingleHistogramMatch()
{
  std::cout << "Running the single histogram matcher" << std::endl;
  // Init file locations
  std::string mainDir = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_2\\olympus";
  std::string targetImageString = "pic.0164.jpg";
  std::string imagePath = mainDir+"\\"+targetImageString;
  int topN = 4;   // Show the top 4 matches because the first match should be itself

  // Init the vectors
  std::vector<std::pair<double, std::string>> matches;

  cv::Mat targetImage = cv::imread(imagePath);
  std::cout << "Reading: " << imagePath << std::endl;
  if (targetImage.empty()) {
    std::cerr << "Error -> Could not open or find the query image" << std::endl;
    exit;
  }

  cv::Mat targetHist;
  computeRGBChromaticityHistogram(targetImage, targetHist);
  std::cout << "Done computing RG chromaticity for target" << std::endl;

  // Compute chromaticity histogram for all images in the directory
  // https://stackoverflow.com/questions/69293016/not-understanding-stdfilesystemdirectory-iterator
  for (const auto &entry : std::filesystem::directory_iterator(mainDir)) {
    std::string imagePath = entry.path().string();
    // std::cout << "imagePath: " << imagePath << std::endl;
    std::string imageFilename = entry.path().filename().string();
    // std::cout << "imageFilename: " << imageFilename << std::endl;
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
      std::cerr << "Error -> Could not open or find the image: " << imagePath << std::endl;
      continue;
    }
    cv::Mat hist;
    computeRGBChromaticityHistogram(image, hist);
    double intersection = histogramIntersection(targetHist, hist);
    matches.push_back({intersection, imageFilename});
    std::cout << "Processing: " << imageFilename << " Intersection: " << intersection << std::endl;
  }

  // Sort by intersection values (descending)
  std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
      return a.first > b.first;
  });

  // Print top N matches
  std::cout << "Top " << topN << " matches for " << targetImageString << ":\n";
  for (int i = 0; i < std::min(topN, static_cast<int>(matches.size())); ++i) {
    std::cout << i + 1 << ". " << matches[i].second << " (Intersection: " << matches[i].first << ")" << std::endl;
  }
}

/*
  Function to run the multi histogram matcher
*/
void computeMultiHistogramMatch()
{
  std::cout << "Running the MULTI histogram matcher" << std::endl;
  // Init file locations
  std::string mainDir = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_2\\olympus";
  std::string targetImageString = "pic.0274.jpg";
  std::string imagePath = mainDir+"\\"+targetImageString;
  int topN = 4;   // Show the top 4 matches because the first match should be itself

  // Init the vectors
  std::vector<std::pair<double, std::string>> matches;

  // Target image is the whole image histogram
  cv::Mat targetImage = cv::imread(imagePath);
  std::cout << "Reading: " << imagePath << std::endl;
  if (targetImage.empty()) {
    std::cerr << "Error -> Could not open or find the query image" << std::endl;
    exit;
  }

  cv::Mat targetHist, centerT, targetCenterHist;
  computeRGChromaticityHistogram(targetImage, targetHist);
  std::cout << "Done computing RG chromaticity for whole image target" << std::endl;

  // Then compute for partial image
  centerT = getCenter(targetImage, 8);
  computeRGChromaticityHistogram(centerT, targetCenterHist);

  // Compute CENTER OF IMAGE histogram for all images in the directory
  // https://stackoverflow.com/questions/69293016/not-understanding-stdfilesystemdirectory-iterator
  for (const auto &entry : std::filesystem::directory_iterator(mainDir)) {
    std::string imagePath = entry.path().string();
    // std::cout << "imagePath: " << imagePath << std::endl;
    std::string imageFilename = entry.path().filename().string();
    // std::cout << "imageFilename: " << imageFilename << std::endl;
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
      std::cerr << "Error -> Could not open or find the image: " << imagePath << std::endl;
      continue;
    }

    // Do the center of the image chromaticity
    cv::Mat hist, center, centerHist;
    int size = 8;
    center = getCenter(image, size);

    // Compute the histogram on the whole
    computeRGChromaticityHistogram(image, hist);

    // Compute on the center
    computeRGChromaticityHistogram(center, centerHist);

    //double intersection = histogramIntersection(targetHist, hist);
    // Do a distance metric instead of interection
    double dist = computeWeightedDistance(targetHist, targetCenterHist, hist, centerHist);
    matches.push_back({dist, imageFilename});
    std::cout << "Processing: " << imageFilename << " Distance: " << dist << std::endl;
  }

  // Sort by intersection values (descending)
  std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
      return a.first > b.first;
  });

  // Print top N matches
  std::cout << "Top " << topN << " matches for " << targetImageString << ":\n";
  for (int i = 0; i < std::min(topN, static_cast<int>(matches.size())); ++i) {
    std::cout << i + 1 << ". " << matches[i].second << " (Distance: " << matches[i].first << ")" << std::endl;
  }
}


// int main()
// {
  
//   //computeSingleHistogramMatch();
//   computeMultiHistogramMatch();
//   // Exit
//   return 0;
// }