/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  Include file for calibration.cpp, calibration functions
*/

#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <opencv2/opencv.hpp>

// Filters
int greyscale( cv::Mat &src, cv::Mat &dst );

// Detecting and extracting corners
int detectCorners( cv::Mat &src, cv::Mat &dst, std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::vector<cv::Vec3f>> &point_list, bool saveCorners );
std::vector<cv::Vec3f> get3DChessboardCorners( int rows, int cols, float squareSize );
int printCorners(std::vector<cv::Point2f> corner_set);
int detectArucoMarkers( cv::Mat &src, cv::Mat &dst );

// Save images / list of found corners
int saveCalibrationImage( cv::Mat &src, std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::vector<cv::Vec3f>> &point_list, std::string img_name);
int saveCurrentFrame( cv::Mat &src, std::string img_name );

int saveCornerSet( std::vector<cv::Point2f> &corner_set, std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::vector<cv::Vec3f>> &point_list, const std::string &filename );
int loadCornerSet( std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::vector<cv::Vec3f>> &point_list, const std::string &filename );

// Do the calibration
int runCameraCalibration( std::vector<std::vector<cv::Vec3f>> &point_list, 
  std::vector<std::vector<cv::Point2f>> &corner_list,
  cv::Size image_size, 
  cv::Mat &camera_matrix, 
  cv::Mat &distortion_coeffs, 
  cv::Mat &rotations, 
  cv::Mat &translations ) ;

// Calculate current camera pose
int calcCameraPose( cv::Mat &src, cv::Mat &camera_matrix, cv::Mat &distortion_coeffs, cv::Mat &rotationVec, cv::Mat &translationVec );

// Project outside corners or 3D axes
int projectOutsideCorners( cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs, const cv::Mat &rotationVec, const cv::Mat &translationVec );

// Create virtual object
int createVirtualObject( cv::Mat &src, cv::Mat &dst, cv::Mat &camera_matrix, cv::Mat &distortion_coeffs, cv::Mat &rotationVec, cv::Mat &translationVec );

// Detect robust features
int detectHarrisCorners( cv::Mat &src, cv::Mat &dst );
int detectOrbFeatures( cv::Mat &src, cv::Mat &dst );
int detectSiftFeatures( cv::Mat &src, cv::Mat &dst );

// Extension: Undistort an image
static double computeReprojectionErrors( const std::vector<std::vector<cv::Point3f> >& objectPoints,
  const std::vector<std::vector<cv::Point2f> >& imagePoints,
  const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
  const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs,
  std::vector<float>& perViewErrors, bool fisheye);

// Extension: Figure out how to get the 3D points on an object in the scene if the scene also has a calibration target. Show the 3-D point cloud.

// Extension: Hide the target when detected
int hideTarget( cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs, cv::Mat &rotationVec, cv::Mat &translationVec, cv::Mat &image);

// DepthAnything
int depthAnything( cv::Mat &src, cv::Mat &dst );
int depthAnythingPtCloud( cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs );

#endif // CALIBRATION_H
