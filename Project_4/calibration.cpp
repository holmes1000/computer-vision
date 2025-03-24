/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  Functions for calibrating on a live video stream
*/

#include "calibration.h"
#include <cmath>
#include <corecrt_math_defines.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <vector>
#include "DA2Network.hpp"

/*
  An alternative greyscale function to cv::cvtColor(new_frame, new_frame, cv::COLOR_BGR2GRAY);
  Greyscale formula = 0.299R + 0.587G + 0.114B
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the greyscale result
 */
int greyscale( cv::Mat &src, cv::Mat &dst ) {
    // std::cout << "Running greyscale filter" << std::endl;
    std::vector<cv::Mat> channels(3);
    cv::split(src, channels); // Split the source image into 3 channels

    dst = 0.299*channels[2] + 0.587*channels[1] + 0.114*channels[0];
    
    return 0;
}

/*
  Detect corners
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the result
  */
int detectCorners( cv::Mat &src, cv::Mat &dst, std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::vector<cv::Vec3f>> &point_list, bool saveCorners) {

    // Convert image to greyscale
    cv::Mat greyFrame;
    greyscale(src, greyFrame);

    // Find the internal corners of the chessboard
    cv::Size patternSize(9, 6);  // Using a 10x7 board = 9x6 internal
    std::vector<cv::Point2f> corner_set;
    bool cornersFound = cv::findChessboardCorners(greyFrame, patternSize, corner_set);

    // Draw if corners found
    if (cornersFound) {
        // Refine the corner locations
        cornerSubPix(greyFrame, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

        // Draw the corners
        drawChessboardCorners(dst, patternSize, cv::Mat(corner_set), cornersFound);
    }

    printCorners(corner_set);

    // If saveCorners is true, save them to a txt file (this is called within saveCalibrationImage)
    if (saveCorners) {
        saveCornerSet(corner_set, corner_list, point_list, "corners.txt");
    }

    return 0;
}


/*
  Function to generate 3D world coordinates for the chessboard corners
  */
std::vector<cv::Vec3f> get3DChessboardCorners( int rows, int cols, float squareSize ) {
    std::vector<cv::Vec3f> corners;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            corners.push_back(cv::Vec3f(j * squareSize, -i * squareSize, 0));
        }
    }
    return corners;

}

/*
  Print corners
  https://stackoverflow.com/questions/28892239/how-do-i-access-the-vector-value-std-vector-cv-point2f-pto
  */
int printCorners(std::vector<cv::Point2f> corner_set) {
    int cornersSize = corner_set.size();
    std::cout << "Found " << cornersSize << " corners" << std::endl; 
    for(int k=0; k<cornersSize; k++){          //goes through all cv::Point2f in the vector
        float x = corner_set[k].x;
        float y = corner_set[k].y;
        // std::cout << "Corner " << k << " is located at: " << x << "," << y << std::endl; 
    }
    return 0;
}

/*
  Saves the calibration image to a file in the images folder
  */
int saveCalibrationImage( cv::Mat &src, std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::vector<cv::Vec3f>> &point_list, std::string img_name ) {
    detectCorners(src, src, corner_list, point_list, true);
    std::string folderPath = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_4\\images";
    std::string filePath = folderPath + "/" + img_name;
    cv::imwrite(filePath, src);
    std::cout << "Image saved at: " << filePath << std::endl;
    return 0;
}

/*
  Saves a background image to a file in the background_images folder
  */
int saveCurrentFrame( cv::Mat &src, std::string img_name ) {
    std::string folderPath = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_4\\background_images";
    std::string filePath = folderPath + "/" + img_name;
    cv::imwrite(filePath, src);
    std::cout << "Image saved at: " << filePath << std::endl;
    return 0;
}

/*
  Saves the corner set to a file
  Arguments:
  std::vector<cv::Point2f> &corner_set - a list of 2D points that represents the corners on the image plane
  std::vector<std::vector<cv::Point2f>>& corner_list
  std::vector<std::vector<cv::Vec3f>> &point_list - a list of std::vector<cv::Vec3f> point_set -> a list of 3D points that represents the corners in world coordinates
  const std::string &filename
  */
int saveCornerSet( std::vector<cv::Point2f> &corner_set, std::vector<std::vector<cv::Point2f>>& corner_list, std::vector<std::vector<cv::Vec3f>> &point_list, const std::string &filename ) {
    std::string folderPath = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_4\\" + filename;
    std::ofstream file(folderPath, std::ios::app); // Append mode
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << folderPath << std::endl;
        return -1;
    }

    // Generate the 3D world points
    std::vector<cv::Vec3f> point_set = get3DChessboardCorners(6, 9, 1.0);

    int cornersSize = corner_set.size();

    // Save the corner set and point set to the respective lists
    corner_list.push_back(corner_set);
    point_list.push_back(point_set);

    // Write corner set to a file
    for (int k=0; k<cornersSize; k++) {
        file << corner_set[k].x << " " << corner_set[k].y << "\n";
    }
    
    file.close();
    std::cout << "Corner set saved to " << filename << std::endl;
    return 0;
}

/*
  Loads the corner set
  Arguments:
  std::vector<cv::Point2f> &corner_set - a list of 2D points that represents the corners on the image plane
  std::vector<std::vector<cv::Point2f>>& corner_list
  std::vector<std::vector<cv::Vec3f>> &point_list - a list of std::vector<cv::Vec3f> point_set -> a list of 3D points that represents the corners in world coordinates
  const std::string &filename
  */
int loadCornerSet(std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::vector<cv::Vec3f>> &point_list, const std::string &filename) {
    std::string folderPath = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_4\\" + filename;
    std::ifstream file(folderPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }

    std::vector<cv::Point2f> corner_set;
    float x, y;
    int num_corners = 54; // Number of corners in each image

    while (file >> x >> y) {
        corner_set.push_back(cv::Point2f(x, y));
        if (corner_set.size() == num_corners) {
            corner_list.push_back(corner_set);
            corner_set.clear();
        }
    }

    file.close();

    // Generate the 3D world points (assuming a 6x9 chessboard)
    std::vector<cv::Vec3f> point_set = get3DChessboardCorners(6, 9, 1.0);

    // Save the point set for each calibration image
    for (int i = 0; i < corner_list.size(); ++i) {
        point_list.push_back(point_set);
    }

    std::cout << "Corner set loaded from " << filename << std::endl;
    return 0;
}

/*
  Calibrate the camera
  */
int runCameraCalibration(std::vector<std::vector<cv::Vec3f>> &point_list, 
    std::vector<std::vector<cv::Point2f>> &corner_list,
    cv::Size image_size, 
    cv::Mat &camera_matrix, 
    cv::Mat &distortion_coeffs, 
    cv::Mat &rotations, 
    cv::Mat &translations) {
    // If the user has selected enough calibration frames--require the user to select at least 5--then let the user run a calibration

    // The parameters to the function are the point_list, corner_list (definitions above), the size of the calibration images, the camera_matrix, the distortion_coefficients, the rotations, and the translations
    // cv::calibrateCamera();

    int flags = 0;

    // Initialize output variables for camera matrix and distortion coefficients
    camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    distortion_coeffs = cv::Mat::zeros(8, 1, CV_64F);

    // Print the initial camera matrix and distortion coefficients
    std::cout << "Initial Camera Matrix: " << camera_matrix << std::endl;
    std::cout << "Initial Distortion Coefficients: " << distortion_coeffs << std::endl;
    
    // Run the camera calibration
    double rms = cv::calibrateCamera(point_list, corner_list, image_size, camera_matrix, distortion_coeffs, rotations, translations, flags);

    // Output the calibration result
    std::cout << "Re-projection error reported by cv::calibrateCamera: " << rms << std::endl;
    std::cout << "Camera matrix: " << camera_matrix << std::endl;
    std::cout << "Distortion coefficients: " << distortion_coeffs << std::endl;

    // Save camera matrix
    std::ofstream camera_file("C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_4\\camera_matrix.txt");
    camera_file << "Camera Matrix:\n" << camera_matrix << "\n";

    // Save distortion coefficients
    std::ofstream distortion_file("C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_4\\distortion_coefficients.txt");
    distortion_file << "Distortion Coefficients:\n" << distortion_coeffs << "\n";

    return 0;
}

/*
  Calculate the camera pose from camera calibration parameters and target
  */
int calcCameraPose( cv::Mat &src, cv::Mat &camera_matrix, cv::Mat &distortion_coeffs, cv::Mat &rotationVec, cv::Mat &translationVec ) {

    // Convert image to greyscale
    cv::Mat greyFrame;
    greyscale(src, greyFrame);

    // Find the internal corners of the chessboard
    cv::Size patternSize(9, 6);  // Using a 10x7 board = 9x6 internal
    std::vector<cv::Point2f> corner_set;

    // Detect target
    bool cornersFound = cv::findChessboardCorners(greyFrame, patternSize, corner_set,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

    if (!cornersFound) {
        std::cerr << "Target not detected." << std::endl;
        return -1;
    }

    // Generate the 3D world points
    std::vector<cv::Vec3f> point_set = get3DChessboardCorners(6, 9, 1.0);

    // If target detected, use solvePNP() to get the rotation and translation vectors
    cv::solvePnP(point_set, corner_set, camera_matrix, distortion_coeffs, rotationVec, translationVec);

    // Print out the rotation and translation data in real time
    std::cout << "Rotation Vector: " << rotationVec << std::endl;
    std::cout << "Translation Vector: " << translationVec << std::endl;

    return 0;
}

/*
  Given the pose estimated in the prior step, have your program use the projectPoints() function to project the 3D points 
  corresponding to at least four corners of the target onto the image plane in real time as the target or camera moves around
  */
int projectOutsideCorners( cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs, const cv::Mat &rotationVec, const cv::Mat &translationVec ) {
    // std::cout << "Rotation vector size: " << rotationVec.size() << std::endl;
    // std::cout << "Rotation Vector: " << rotationVec << std::endl;
    // CV_Assert(rotationVec.size() == cv::Size(3, 1) || rotationVec.size() == cv::Size(1, 3));
    // CV_Assert(translationVec.size() == cv::Size(3, 1) || translationVec.size() == cv::Size(1, 3));
    
    // Define the 3D points of the corners of the target
    std::vector<cv::Point3f> point_set;

    point_set.push_back(cv::Point3f(0, 0, 0)); // Corner 1
    point_set.push_back(cv::Point3f(8, 0, 0)); // Corner 2
    point_set.push_back(cv::Point3f(8, -5, 0)); // Corner 3
    point_set.push_back(cv::Point3f(0, -5, 0)); // Corner 43

    // Project the 3D points to the image plane
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(point_set, rotationVec, translationVec, camera_matrix, distortion_coeffs, image_points);

    // Draw the projected points
    for (size_t i = 0; i < image_points.size(); ++i) {
        cv::circle(dst, image_points[i], 5, cv::Scalar(0, 0, 255), -1);
    }

    return 0;
}

/*
  Construct a virtual object in 3D world space made out of lines that floats above the board.
  Then project that virtual object to the image and draw the lines in the image.
  */
int createVirtualObject( cv::Mat &src, cv::Mat &dst, cv::Mat &camera_matrix, cv::Mat &distortion_coeffs, cv::Mat &rotationVec, cv::Mat &translationVec ) {

    // Define the 3D points of the virtual object
    std::vector<cv::Point3f> point_set;
    float radius = 4.0;
    float height = 8.0;
    float offset = 2.0;

    // Define the vertices
    point_set.push_back(cv::Point3f(0, 0, offset));      // Vertex 0: top tip
    point_set.push_back(cv::Point3f(-radius, -radius, height)); // Vertex 1
    point_set.push_back(cv::Point3f(radius, -radius, height));  // Vertex 2
    point_set.push_back(cv::Point3f(radius, radius, height));   // Vertex 3
    point_set.push_back(cv::Point3f(-radius, radius, height));  // Vertex 4
    point_set.push_back(cv::Point3f(0, 0, height * 2 + offset));
    point_set.push_back(cv::Point3f(-radius, -radius, height)); // Vertex 1
    point_set.push_back(cv::Point3f(radius, -radius, height));  // Vertex 2
    point_set.push_back(cv::Point3f(radius, radius, height));   // Vertex 3
    point_set.push_back(cv::Point3f(-radius, radius, height));  // Vertex 4

    // Define the edges
    std::vector<std::pair<int, int>> edges;
    edges.push_back(std::make_pair(0, 1));
    edges.push_back(std::make_pair(0, 2));
    edges.push_back(std::make_pair(0, 3));
    edges.push_back(std::make_pair(0, 4));
    edges.push_back(std::make_pair(1, 2));
    edges.push_back(std::make_pair(2, 3));
    edges.push_back(std::make_pair(3, 4));
    edges.push_back(std::make_pair(4, 1));

    edges.push_back(std::make_pair(5, 6));
    edges.push_back(std::make_pair(5, 7));
    edges.push_back(std::make_pair(5, 8));
    edges.push_back(std::make_pair(5, 9));
    edges.push_back(std::make_pair(6, 7));
    edges.push_back(std::make_pair(7, 8));
    edges.push_back(std::make_pair(8, 9));
    edges.push_back(std::make_pair(9, 6));

    // Project the 3D points to the image plane
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(point_set, rotationVec, translationVec, camera_matrix, distortion_coeffs, image_points);

    // Draw the lines in the image
    for (const auto &edge : edges) {
        cv::line(dst, image_points[edge.first], image_points[edge.second], cv::Scalar(0, 0, 255), 2);
    }

    return 0;
}

/*
  Detect Harris corners
  https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
  https://datahacker.rs/opencv-harris-corner-detector-part2/
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the result
  */
int detectHarrisCorners( cv::Mat &src, cv::Mat &dst ) {

    // Convert image to greyscale
    cv::Mat greyFrame;
    // greyscale(src, greyFrame);
    cvtColor(src, greyFrame, cv::COLOR_BGR2GRAY);

    // Detecting corners
    cv::Mat output, output_norm, output_norm_scaled;
    output = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::cornerHarris(greyFrame, output, 2, 3, 0.04);

    // Normalizing
    cv::normalize(output, output_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(output_norm, output_norm_scaled);

    // Drawing a circle around corners
    for(int j = 0; j < output_norm.rows ; j++){
        for(int i = 0; i < output_norm.cols; i++){
            if((int) output_norm.at<float>(j,i) > 100){
               cv::circle(dst, cv::Point(i,j), 2,  cv::Scalar(0,0,255), 2, 8, 0 );
            }
        }
    }

    return 0;
}

/*
  Detect Orb features (Oriented FAST and Rotated BRIEF) -> SURF isn't available
  https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html
  https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html
  https://stackoverflow.com/questions/33670222/how-to-use-surf-and-sift-detector-in-opencv-for-python
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the result
  */
int detectOrbFeatures( cv::Mat &src, cv::Mat &dst ) {

    // Convert image to greyscale
    cv::Mat greyFrame;
    greyscale(src, greyFrame);

    // Detect keypoints using SURF detector
    int minHessian = 400;

    // Create pointer for ORB feature detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(minHessian);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat targetDescriptors;

    orb->detectAndCompute( src, cv::noArray(), keypoints, targetDescriptors);

    //-- Draw keypoints
    drawKeypoints( src, keypoints, dst );
    return 0;
}


/*
  Detect SIFT features (Scale Invariant Feature Transform)
  https://stackoverflow.com/questions/22722772/how-to-use-sift-in-opencv
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the result
  */
int detectSiftFeatures( cv::Mat &src, cv::Mat &dst ) {

    // Create pointer for SIFT feature detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat targetDescriptors;

    sift->detectAndCompute(src, cv::noArray(), keypoints, targetDescriptors);

    //-- Draw keypoints
    drawKeypoints( src, keypoints, dst );

    return 0;
}


/*
  Function to return the average re-projection error
  https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
  Arguments:
  */
static double computeReprojectionErrors( const std::vector<std::vector<cv::Point3f> >& point_set,
    const std::vector<std::vector<cv::Point2f> >& imagePoints,
    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
    const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs,
    std::vector<float>& perViewErrors, bool fisheye)
{
    std::vector<cv::Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(point_set.size());

    for(size_t i = 0; i < point_set.size(); ++i ) {
        if (fisheye)
        {
            cv::fisheye::projectPoints(point_set[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix, distCoeffs);
        }
        else
        {
            cv::projectPoints(point_set[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
        }

        err = norm(imagePoints[i], imagePoints2, cv::NORM_L2);

        size_t n = point_set[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);
}

/*
  Function to cover the chessboard target with a bounding box
  Arguments:
  */
int hideTarget( cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs, cv::Mat &rotationVec, cv::Mat &translationVec, cv::Mat &image ) {

    // Define the 3D points of the corners of the target
    std::vector<cv::Point3f> point_set;
    float offset = 1.0;

    point_set.push_back(cv::Point3f(-offset, offset, 0)); // Corner 1
    point_set.push_back(cv::Point3f(8+offset, offset, 0)); // Corner 2
    point_set.push_back(cv::Point3f(8+offset,-5-offset, 0)); // Corner 3
    point_set.push_back(cv::Point3f(-offset, -5-offset, 0)); // Corner 4
    
    // Project the 3D points to the image plane
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(point_set, rotationVec, translationVec, camera_matrix, distortion_coeffs, projected_points);
    
    if (image.empty())
    {
        // Draw the projected points
        for (size_t i = 0; i < projected_points.size(); ++i) {
            cv::circle(dst, projected_points[i], 5, cv::Scalar(0, 0, 255), -1);
        }

        // Convert the projected points to a polygon (rectange doesn't work because it doesn't rotate)
        std::vector<cv::Point> polygon;
        for (const auto& pt : projected_points) {
            polygon.push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
        }

        // Draw a red polygon over the target
        cv::polylines(dst, polygon, true, cv::Scalar(0, 0, 255), 2);

        // Fill the polygon
        std::vector<std::vector<cv::Point>> polygons{polygon};
        cv::fillPoly(dst, polygons, cv::Scalar(0, 0, 255));
    }
    else 
    {
        // Convert the projected points to a polygon (rectange doesn't work because it doesn't rotate)
        std::vector<cv::Point> polygon;
        for (const auto& pt : projected_points) {
            polygon.push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
        }

        // Create a mask
        cv::Mat mask(dst.size(), CV_8UC1, cv::Scalar(0));

        // Fill the polygon on the mask
        std::vector<std::vector<cv::Point>> polygons{polygon};
        cv::fillPoly(mask, polygons, cv::Scalar(255));

        // Determine the bounding box of the polygon
        cv::Rect bounding_box = cv::boundingRect(polygon);

        // Make sure the bounding box is within the bounds of the source image
        bounding_box &= cv::Rect(0, 0, src.cols, src.rows);

        // Extract the patch from the background image
        cv::Mat patch = image(bounding_box);

        // Overlay the patch onto the dst image, using the mask
        cv::Mat background_patch = dst(bounding_box);
        patch.copyTo(background_patch, mask(bounding_box));
    }
    

    return 0;
}

/*
  A function to run DepthAnythingV2 model
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the depthanything result
 */
int depthAnything( cv::Mat &src, cv::Mat &dst ) {
    // Init vars
    const float reduction = 0.5;

    // make a DANetwork object
    DA2Network da_net( "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_4\\model_fp16.onnx" );

    // for speed purposes, reduce the size of the input frame by half
    cv::Mat src_small;
    cv::resize( src, src_small, cv::Size(), reduction, reduction );
    // printf("Expected size: %d %d\n", src.cols, src.rows);
    float scale_factor = 256.0 / (src_small.rows*reduction);
    // printf("Using scale factor %.2f\n", scale_factor);

    // set the network input
    da_net.set_input( src_small, scale_factor );

    // run the network
    cv::Mat dst_small;
    da_net.run_network( dst_small, src.size() );

    // upscale
    cv::resize(dst_small, dst, src.size(), 0, 0, cv::INTER_LINEAR);

    // apply a color map to the depth output to get a good visualization
    cv::applyColorMap(dst, dst, cv::COLORMAP_INFERNO );
    return 0;
}

/*
  A function to run DepthAnythingV2 model
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the depthanything result
 */
int depthAnythingPtCloud( cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &distortion_coeffs ) {

    float fx = static_cast<float>(camera_matrix.at<double>(0, 0));
    float fy = static_cast<float>(camera_matrix.at<double>(1, 1));
    float cx = static_cast<float>(camera_matrix.at<double>(0, 2));
    float cy = static_cast<float>(camera_matrix.at<double>(1, 2));

    // Convert the color-mapped depth image to grayscale
    cv::Mat depth_map;
    cv::cvtColor(src, depth_map, cv::COLOR_BGR2GRAY);

    // Convert grayscale values to floating-point depth values
    cv::Mat depth;
    depth_map.convertTo(depth, CV_32F, 1.0 / 255.0);

    // Initialize the 3D point cloud
    std::vector<cv::Point3f> point_cloud;
    float threshold_distance = 0.5;

    // Iterate through the depth map to generate the point cloud and visualize it
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            float z = depth.at<float>(y, x); // Depth value
            if (z > 0 && z <= threshold_distance) { // Apply threshold for distance
                // Back-project to 3D space
                float X = (x - cx) * z / fx;
                float Y = (y - cy) * z / fy;
                float Z = z;

                // Add the 3D point to the point cloud
                point_cloud.emplace_back(X, Y, Z);

                // Visualize the point on dst
                int intensity = static_cast<int>((Z / threshold_distance) * 255.0); // Map Z to intensity
                intensity = std::min(255, std::max(0, intensity)); // Intensity to [0, 255]
                cv::Scalar color(0, intensity, 255 - intensity); // Gradient (red to yellow)

                // Draw the point on dst
                cv::circle(dst, cv::Point(x, y), 1, color, -1);
            }
        }
    }

    return 0;
}
