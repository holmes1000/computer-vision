/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  This program is given a target image, the feature set, and the feature vector file. 
  It then computes the features for the target image, reads the feature vector file, and identifies the top N matches.
*/

#include <iostream>
#include <vector>
#include "csv_util.h"
#include "feature_vector_writer.h"
#include <algorithm>
#include <opencv2/opencv.hpp>

/*
  Function to compute the sum of square difference
  Make sure that comparing the image with itself results in a distance of 0.
  Arguments:
  std::vector<float>& vec1 - first vector
  std::vector<float>& vec2 - second vector
*/
float sumOfSquaredDifference(std::vector<float>& vec1, std::vector<float>& vec2) {
  float ssd = 0.0f;
    for (size_t i = 0; i < vec1.size(); ++i) {
        float diff = vec1[i] - vec2[i];
        ssd += diff * diff;
    }
    return ssd;
}

int main()
{
  std::cout << "Running the baseline matcher" << std::endl;
  // Init file locations
  std::string targetImage = "pic.1016.jpg";
  std::string output_csv = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_2\\build\\Debug\\feature_vectors.csv";
  int topN = 4;   // Show the top 4 matches because the first match should be itself

  // Init the vectors
  std::vector<char *> filenames;
  std::vector<std::vector<float>> featureVectors;
  bool echo_file = true;

  // Retrieve the target image features from the CSV (the first column is the imagePath)
  std::cout << "Reading feature vectors from CSV: " << output_csv << std::endl;
  // this function returns the filenames as a std::vector of character arrays, and the remaining data as a 2D std::vector<float>.
  if (read_image_data_csv((char*)output_csv.c_str(), filenames, featureVectors, echo_file) != 0) {
    std::cerr << "Error -> Could not read feature vectors from CSV." << std::endl;
    echo_file = false;
    return 1;
  }

  // List the readings to make sure stuff got read
  if (filenames.empty()) {
    std::cerr << "Error -> No files were actually read in" << std::endl;
    return 1;
  }
  else {
    std::cout << "Filenames read from CSV:\n";
    for (size_t i = 0; i < filenames.size(); ++i) {
      std::cout << filenames[i] << std::endl;
    }
  }
  
  // Allocate features for the target image
  std::vector<float> targetFeatures;
  for (size_t i = 0; i < filenames.size(); ++i) {
    if (std::string(filenames[i]) == targetImage) {
      targetFeatures = featureVectors[i];
      break;
    }
  }

  // Check if targetFeatures is empty
  if (targetFeatures.empty()) {
    std::cerr << "Error -> Features not found in the CSV for the target image." << std::endl;
    return 1;
  }

  // Compute sum of squared difference (SSD) for all images vs the target
  std::vector<std::tuple<std::string, float>> distances;
  for (size_t i = 0; i < filenames.size(); ++i) {
    float ssd = sumOfSquaredDifference(targetFeatures, featureVectors[i]);
    distances.emplace_back(filenames[i], ssd);
  }

  // Sort by SSD (smallest to largest)
  // Remember, the smaller a distance, the more similar two images will be.
  std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) { return std::get<1>(a) < std::get<1>(b); });

  // Print top N matches
  std::cout << "Top " << topN << " matches for " << targetImage << ":\n";
  for (int i = 0; i < std::min(topN, static_cast<int>(distances.size())); ++i) {
      std::cout << i + 1 << ". " << std::get<0>(distances[i]) << " (SSD: " << std::get<1>(distances[i]) << ")" << std::endl;
  }

  // Clean up allocated memory from filenames
  for (char* filename : filenames) {
    delete[] filename;
  }

  // Exit
  return 0;
}