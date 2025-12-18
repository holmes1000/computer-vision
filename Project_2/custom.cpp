/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  This program uses face detection and SIFT to do CBIR
*/

#include <iostream>
#include <vector>
#include "csv_util.h"
#include "feature_vector_writer.h"
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "faceDetect.h"
#include "custom.h"

/*
  Function to return images paths of images that contains faces
  Uses an upper and lower threshold
  Arguments:
  std::vector<std::string> &imagePaths
  std::vector<std::string> &resultPaths
*/
void retrieveImagesWithFaces(const std::vector<std::string> &imagePaths, std::vector<std::string> &resultPaths) 
{
  std::string mainDir = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_2\\olympus";
  int threshold_lower = 40;
  int threshold_upper = 100;
  for (const auto &imagePath : imagePaths) {
        cv::Mat img = cv::imread(mainDir+"\\"+imagePath);
        if (img.empty()) {
            std::cerr << "Error loading image: " << imagePath << std::endl;
            continue;
        }

        cv::Mat grey;
        cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        if (detectFaces(grey, faces) == 0 && !faces.empty()) {
          // Add in a threshold so there's less false detections
          bool isValid = false;
          for (const auto &face : faces) 
          {
            if ((face.width >= threshold_lower && face.height >= threshold_lower)
            && (face.width <= threshold_upper && face.height <= threshold_upper))
            {
              // std::cout << "Face width: " << face.width << std::endl;
              isValid = true;
              break;
            }
          }
          if (isValid) {resultPaths.push_back(imagePath);}
        }
  }
}

/*
  Function to compute the sum of square difference
  Make sure that comparing the image with itself results in a distance of 0.
  Arguments:
  std::vector<float>& vec1 - first vector
  std::vector<float>& vec2 - second vector
*/
float sumOfSquaredDifference(std::vector<float> &vec1, std::vector<float> &vec2) {
  float ssd = 0.0f;
    for (size_t i = 0; i < vec1.size(); ++i) {
        float diff = vec1[i] - vec2[i];
        ssd += diff * diff;
    }
    return ssd;
}

/*
  Function to compute the sum of square difference with CV mat images
  Make sure that comparing the image with itself results in a distance of 0.
  Arguments:
  const cv::Mat &descriptor1
  const cv::Mat &descriptor2
*/
double sumOfSquaredDifference(const cv::Mat &descriptor1, const cv::Mat &descriptor2) {
    double ssd = 0.0;
    for (int i = 0; i < descriptor1.cols; ++i) {
        double diff = descriptor1.at<float>(0, i) - descriptor2.at<float>(0, i);
        ssd += diff * diff;
    }
    return ssd;
}

/*
  Function to return all the images containing a face using retrieveImagesWithFaces()
  Use this for simplification and separation of tasks
  Arguments:
  std::vector<char *> filenames
*/
void runFigure10Task(std::vector<char *> filenames) {
  std::cout << "Running Figure 10 task" << std::endl;
  // Retrieve images containing faces
  std::vector<std::string> imagePaths(filenames.begin(), filenames.end());
  std::vector<std::string> resultPaths;
  retrieveImagesWithFaces(imagePaths, resultPaths);

  // Print paths of images containing faces
  std::cout << "Images containing faces:" << std::endl;
  for (const auto &path : resultPaths) {
    std::cout << path << std::endl;
  }
  std::cout << "Amount of images containing faces" << resultPaths.size() << std::endl;
}

void doSIFT()
{
  // Init the dir
  std::string mainDir = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_2\\olympus_small_bananas";
  // Load target image
  std::string targetImage = "pic.0343.jpg";
  cv::Mat targetImg = imread(mainDir+"\\"+targetImage, cv::IMREAD_GRAYSCALE);
  if (targetImg.empty()) {
      std::cerr << "Error -> Could not open or find the target image" << std::endl;
      exit;
  }
  // Detect keypoints and descriptors using SIFT for target image
  // https://stackoverflow.com/questions/22722772/how-to-use-sift-in-opencv
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

  // Create pointer for SIFT feature detector
  std::vector<cv::KeyPoint> targetKeypoints;
  cv::Mat targetDescriptors;
  sift->detectAndCompute(targetImg, cv::noArray(), targetKeypoints, targetDescriptors);
  std::vector<std::pair<std::string, double>> matchScores;

  // Iterate through each image in the database
  for (const auto &entry : std::filesystem::directory_iterator(mainDir)) {
      // Load database image
      cv::Mat dbImg = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
      if (dbImg.empty()) {
          std::cout << "Could not open or find the database image: " << entry.path() << std::endl;
          continue;
      }
      // Detect keypoints and descriptors using SIFT for database image
      std::vector<cv::KeyPoint> dbKeypoints;
      cv::Mat dbDescriptors;
      sift->detectAndCompute(dbImg, cv::noArray(), dbKeypoints, dbDescriptors);
      // Calculate SSD between descriptors
      std::vector<cv::DMatch> matches;
      for (int i = 0; i < targetDescriptors.rows; ++i) {
          double bestSSD = DBL_MAX;
          int bestMatchIdx = -1;
          for (int j = 0; j < dbDescriptors.rows; ++j) {
              double ssd = sumOfSquaredDifference(targetDescriptors.row(i), dbDescriptors.row(j));
              if (ssd < bestSSD) {
                  bestSSD = ssd;
                  bestMatchIdx = j;
              }
          }
          matches.push_back(cv::DMatch(i, bestMatchIdx, bestSSD));
      }
      // Calculate match score (sum of distances of good matches)
      double matchScore = 0;
      for (const auto& match : matches) {
          matchScore += match.distance;
      }
      // Store match score with image path
      matchScores.push_back({entry.path().string(), matchScore});
  }
  // Sort matches by score (ascending order)
  std::sort(matchScores.begin(), matchScores.end(), [](const auto& a, const auto& b) {
      return a.second < b.second;
  });
  // List top matches
  std::cout << "Top matches:" << std::endl;
  for (const auto &match : matchScores) {
      std::cout << "Image: " << match.first << " Score: " << match.second << std::endl;
  }
}

int main()
{
  std::cout << "Running the custom CBIR task" << std::endl;
  // Init file locations
  std::string mainDir = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_2\\olympus";
  std::string targetImage = "pic.0343.jpg";
  std::string output_csv = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_2\\ResNet18_olym.csv";
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
  
  // -------------- RUNNING THE TASKS --------------------
  //runFigure10Task(filenames);
  //doSIFT();


  // Clean up allocated memory from filenames
  for (char* filename : filenames) {
    delete[] filename;
  }

  // Exit
  return 0;
}