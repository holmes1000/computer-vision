/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  Use a whole image color histogram and a whole image texture histogram 
  as the feature vector. Choose a texture metric of your choice for this task. 
  Design a distance metric that weights the two types of histograms equally.
*/

#include <iostream>
#include <vector>
#include "csv_util.h"
#include "feature_vector_writer.h"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "texture.h"
#include <filesystem>
#include "histogram_matcher.h"

/*
  Gets the 3x3 Sobel filter horizontal (X) result
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the 3x3 Sobel filter horizontal (X) result
 */
int sobelX3x3( cv::Mat &src, cv::Mat &dst ) {
    // std::cout << "Running Sobel X filter" << std::endl;
    // std::cout << "src channels: " << src.channels() << std::endl;
    if (src.channels() != 3) {
        std::cerr << "Error -> Source image must be a 3-channel image." << std::endl;
        return -1;
    }

    // Define the 3x3 Sobel X kernel
    int kernel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    cv::Mat sobelx = cv::Mat::zeros(src.size(), CV_16SC3);
    // std::cout << "Init w zeros" << std::endl;
    
    // Apply the kernel
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            cv::Vec3s sum = {0, 0, 0};
            // Apply the kernel to the neighbors the current pixel
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    for (int i = 0; i < 3; i++) {     // Loop for each color
                        sum[i] += static_cast<short>(src.at<cv::Vec3b>(y + ky, x + kx)[i] * kernel_x[ky + 1][kx + 1]);
                    }
                }
            }
            sobelx.at<cv::Vec3s>(y, x) = sum;
        }
    }

    cv::convertScaleAbs(sobelx, dst);

    return 0;
}

/*
  Gets the 3x3 Sobel filter vertical (Y) result
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the 3x3 Sobel filter vertical (Y) result
 */
int sobelY3x3( cv::Mat &src, cv::Mat &dst ) {
    // std::cout << "Running Sobel Y filter" << std::endl;
    // Define the 3x3 Sobel Y kernel
    int kernel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    //std::cout << "src type: " << src.type() << std::endl;
    cv::Mat sobely = cv::Mat::zeros(src.size(), CV_16SC3);
    
    // Apply the kernel
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            cv::Vec3s sum = {0, 0, 0};
            // Apply the kernel to the neighbors the current pixel
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    for (int i = 0; i < 3; i++) {     // Loop for each color
                        sum[i] += static_cast<short>(src.at<cv::Vec3b>(y + ky, x + kx)[i] * kernel_y[ky + 1][kx + 1]);
                    }
                }
            }
            sobely.at<cv::Vec3s>(y, x) = sum;
        }
    }

    cv::convertScaleAbs(sobely, dst);

    return 0;
}

/*
  Calculates the magnitude of the gradients from the x and y directional gradients
  Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
  Arguments:
  cv::Mat &sx  - a source image representing x-direction gradient (CV_16SC3)
  cv::Mat &sy  - a source image representing y-direction gradient (CV_16SC3)
  cv::Mat &dst - a destination image to store the magnitudes of the gradients
 */
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ) {
    // std::cout << "Calculating magnitude of the gradients" << std::endl;
    //std::cout << "sx size " << sx.size() << std::endl;
    CV_Assert(sx.type() == CV_16SC3);
    CV_Assert(sy.type() == CV_16SC3);
    cv::Mat mag_matrix(sx.size(), CV_16SC3);
    // Calculate the magnitude using the Euclidean distance formula
    for (int y = 0; y < sx.rows; ++y) {
        for (int x = 0; x < sx.cols; ++x) {
            // Access the gradient values at (x, y)
            cv::Vec3s gx = sx.at<cv::Vec3s>(y, x);
            cv::Vec3s gy = sy.at<cv::Vec3s>(y, x);
            cv::Vec3s mag; // 3-channel unsigned char for display

            // Calculate the magnitude for each color channel
            for (int i = 0; i < 3; ++i) {
                mag[i] = std::sqrt(static_cast<float>(gx[i] * gx[i] + gy[i] * gy[i]));
            }

            // Assign the computed magnitude to the output image
            mag_matrix.at<cv::Vec3s>(y, x) = mag;
        }
    }
    mag_matrix.convertTo(dst, CV_8UC3);
    return 0;
}

void computeTextureHistogram(cv::Mat &src, cv::Mat &dst)
{
    cv::Mat sobelX, sobelY, grad_mag;
    
    // Compute gradients using Sobel X and Y
    sobelX3x3(src, sobelX);
    sobelY3x3(src, sobelY);
    // std::cout << "Sobels are calculated" << std::endl;

    // Convert sobelX and sobelY to CV_16SC3 !!
    sobelX.convertTo(sobelX, CV_16SC3);
    sobelY.convertTo(sobelY, CV_16SC3);

    // Compute gradient magnitude
    magnitude(sobelX, sobelY, grad_mag);
    // std::cout << "Magnitude is calculated" << std::endl;
    
    // Compute histogram of gradient magnitudes
    std::vector<cv::Mat> hist(3);
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    for (int c = 0; c < 3; c++) {
        cv::Mat channel(grad_mag.rows, grad_mag.cols, CV_8U);   // MAKE SURE TO SET THIS
        for (int y = 0; y < grad_mag.rows; y++) {
            for (int x = 0; x < grad_mag.cols; x++) {
                channel.at<uchar>(y, x) = grad_mag.at<cv::Vec3b>(y, x)[c];
            }
        }
        cv::calcHist(&channel, 1, 0, cv::Mat(), hist[c], 1, &histSize, &histRange, true, false);
        cv::normalize(hist[c], hist[c], 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    }

    cv::hconcat(hist, dst);
}

/*
  Function to run the whole image texture histogram and whole image color histogram as the feature vector
*/
void computeTextureHistogramMatcher()
{
  std::cout << "Running the Texture histogram matcher" << std::endl;
  // Init file locations
  std::string mainDir = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_2\\olympus";
  std::string targetImageString = "pic.0535.jpg";
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

  cv::Mat targetHist, targetTextureHist;
  computeRGChromaticityHistogram(targetImage, targetHist, 16, 16);
  std::cout << "Done computing RG chromaticity for whole image target" << std::endl;
  computeTextureHistogram(targetImage, targetTextureHist);
  std::cout << "Done computing texture histogram" << std::endl;


  // Compute histogram for all images in the directory
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

    // Compute on the textureHist on the whole image
    cv::Mat compareHist, compareTextureHist;
    computeRGChromaticityHistogram(image, compareHist, 16, 16);
    computeTextureHistogram(targetImage, compareTextureHist);

    // Do a distance metric instead of interection
    double dist = computeWeightedDistance(targetHist, targetTextureHist, compareHist, compareTextureHist, 1, 1);
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

int main()
{
    computeTextureHistogramMatcher();
    // Exit
    return 0;
}