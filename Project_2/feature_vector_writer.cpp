/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  This program is given a directory of images and feature set and it writes the feature vector for each image to a file
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "csv_util.h"

/*
  Gets the feature vector for an image. 
  Extract them using a 7x7 square in the middle of the image as a feature vector.
  Arguments:
  std::string &imagePath  - where the image is in the directory
*/
std::vector<float> getFeatures(std::string &imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Error -> Could not load image: " << imagePath << std::endl;
        return {};
    }

    int centerX = image.cols / 2;
    int centerY = image.rows / 2;
    int halfSize = 3;  // Half of the 7x7 square size is 3

    // Extract the 7x7 square in the middle of the image
    cv::Rect region(centerX - halfSize, centerY - halfSize, 7, 7);
    cv::Mat centerSquare = image(region);

    // Flatten the 7x7 square into a feature vector
    std::vector<float> features;
    for (int i = 0; i < centerSquare.rows; ++i) {
        for (int j = 0; j < centerSquare.cols; ++j) {
            for (int c = 0; c < centerSquare.channels(); ++c) {
                features.push_back(centerSquare.at<cv::Vec3b>(i, j)[c]);
            }
        }
    }

    return features;
}

int main()
{
  std::cout << "Running the feature vector writer" << std::endl;
  // Init the directories
  std::string mainDir = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_2\\olympus";
  std::string image_dir = mainDir;
  std::cout << "Image directory is located at: " << image_dir << std::endl;
  std::string output_csv = "feature_vectors.csv";
  std::cout << "Feature vectors will be written to: " << output_csv << std::endl;   // Note: this will be under build/Debug

  // Make sure directory exists
  if (!std::filesystem::exists(image_dir) || !std::filesystem::is_directory(image_dir)) {
    std::cerr << "ERROR -> Directory not found: " << image_dir << std::endl;
    return 1;
  }

  // Create the feature vectors for each image in the olympus directory
  // https://stackoverflow.com/questions/69293016/not-understanding-stdfilesystemdirectory-iterator
  bool reset_file = true;
  for (const auto &entry : std::filesystem::directory_iterator(image_dir)) {
        std::string imagePath = entry.path().string();  // Get the image path
        std::string imageFilename = entry.path().filename().string();   // Get the filename
        std::cout << "Processing an image : " << imageFilename << std::endl;
        std::vector<float> features = getFeatures(imagePath);  // Extract features
        append_image_data_csv(const_cast<char*>(output_csv.c_str()), const_cast<char*>(imageFilename.c_str()), features, reset_file);  // Write features to file
        reset_file = false;  // Reset file only for the first image
  }

  // Exit
  return 0;
}