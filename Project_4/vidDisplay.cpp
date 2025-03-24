/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  Capture live video from a camera and conduct camera calibration
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "calibration.h"

int main()
{
    cv::VideoCapture *capdev;

    // open the video device
    // https://answers.opencv.org/question/215586/why-does-videocapture1-take-so-long/ This speeds up opening the video device
    capdev = new cv::VideoCapture(1, cv::CAP_DSHOW);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    // Create the windows
    cv::namedWindow("Video", 1); // identifies a window
    cv::namedWindow("Depth", 1);
    // cv::namedWindow("Harris Corners", 1);

    cv::Mat frame, depthFrame;
    int img_count = 0;
    bool imageSaved = false;
    bool replaceImageSaved = false;
    char prev_key = -1;
    bool calibrated = false;

    // Initialize the lists for storing corner sets and point sets
    std::vector<cv::Vec3f> point_set;
    std::vector<std::vector<cv::Vec3f>> point_list;
    std::vector<std::vector<cv::Point2f>> corner_list;

    // Initialize camera matrix, distortion coefficients, rotation vectors, and translation vectors
    cv::Mat camera_matrix, distortion_coeffs;
    cv::Mat rotationVec, translationVec;

    cv::Mat background_img;
    bool background_img_loaded = false;

    for (;;)
    {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty())
        {
            printf("frame is empty\n");
            break;
        }

        // Apply modifications based on the last key pressed 
        cv::Mat src_frame = frame.clone();
        cv::Mat resultFrame = frame.clone();

        // Get the camera_matrix and distortion
        // runCameraCalibration();
        // Get the rotation and translation vectors
        // calcCameraPose();

        switch (prev_key) 
        {
            // Quit
            case 'q':
                break;
            // Detect corners
            case 'c':
                detectCorners(src_frame, resultFrame, corner_list, point_list, false);
                break;
            case 's':
                // detectCorners(src_frame, resultFrame);
                if (!imageSaved) {
                    std::string img_name = "image_" + std::to_string(img_count) + ".png";
                    saveCalibrationImage(src_frame, corner_list, point_list, img_name);
                    img_count += 1;
                    imageSaved = true;
                }
                break;
            case 'n':
                if (!replaceImageSaved) {
                    std::string img_name = "background.png";
                    saveCurrentFrame( src_frame, img_name );
                    replaceImageSaved = true;
                }
            // Load the corner set from a txt file and calibrate
            case '1':
                if (!calibrated) {
                    loadCornerSet(corner_list, point_list, "corners.txt");

                    // Get the camera_matrix and distortion
                    runCameraCalibration(point_list, corner_list, frame.size(), camera_matrix, distortion_coeffs, rotationVec, translationVec);

                    calibrated = true;
                }
                break;
            case '2':
                // Get the rotation and translation vectors in real-time
                calcCameraPose(src_frame, camera_matrix, distortion_coeffs, rotationVec, translationVec);
                break;
            case '3':
                // Get the rotation and translation vectors in real-time
                calcCameraPose(src_frame, camera_matrix, distortion_coeffs, rotationVec, translationVec);
                if (!rotationVec.empty() || !translationVec.empty())
                {
                    // Project the points
                    projectOutsideCorners(src_frame, resultFrame, camera_matrix, distortion_coeffs, rotationVec, translationVec);
                }
                break;
            case '4':
                // Get the rotation and translation vectors in real-time
                calcCameraPose(src_frame, camera_matrix, distortion_coeffs, rotationVec, translationVec);
                // Hide the target
                hideTarget(src_frame, resultFrame, camera_matrix, distortion_coeffs, rotationVec, translationVec, background_img);
                break;
            case '5':
                // Get the rotation and translation vectors in real-time
                calcCameraPose(src_frame, camera_matrix, distortion_coeffs, rotationVec, translationVec);
                // Create the virtual object
                createVirtualObject(src_frame, resultFrame, camera_matrix, distortion_coeffs, rotationVec, translationVec);
                break;
            // Hide target on a pre-recorded video
            case '6':
                if (!background_img_loaded)
                {
                    background_img = cv::imread("C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_4\\background_images\\background.png");
                    background_img_loaded = true;
                }
                // Get the rotation and translation vectors in real-time
                calcCameraPose(src_frame, camera_matrix, distortion_coeffs, rotationVec, translationVec);
                // Hide the target
                hideTarget(src_frame, resultFrame, camera_matrix, distortion_coeffs, rotationVec, translationVec, background_img);
                break;
            case '7':
                depthAnything(src_frame, depthFrame);
                depthAnythingPtCloud(depthFrame, resultFrame, camera_matrix, distortion_coeffs);
                break;
            case 'h':
                detectHarrisCorners(src_frame, resultFrame);
                break;
            case 'o':
                detectOrbFeatures(src_frame, resultFrame);
                break;
            case 'f':
                detectSiftFeatures(src_frame, resultFrame);
                break;
            // Undistort the feed
            case 'd':
            {
                // cv::Mat frame, undistorted_frame;
                // cv::Mat map1, map2;
                // cv::initUndistortRectifyMap(camera_matrix, distortion_coeffs, cv::Mat(),
                //                             cv::getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, src_frame.size(), 1, src_frame.size(), 0),
                //                             src_frame.size(), CV_32FC1, map1, map2);
                // cv::remap(src_frame, resultFrame, map1, map2, cv::INTER_LINEAR);
                cv::undistort(src_frame, resultFrame, camera_matrix, distortion_coeffs);
            }
            break;
            default:
            {
                imageSaved = false;
                calibrated = false;
                replaceImageSaved = false;
                break;
            }
        }
        
        // Look for new key press
        char key = cv::waitKey(10);
        if (key != -1) { 
            std::cout << "New key press detected: " << key << std::endl;
            prev_key = key; 
        }

        // Make sure to display the new frames and the OG
        cv::imshow("Video", resultFrame);
        if (!depthFrame.empty())
        {
            cv::imshow("Depth", depthFrame);
        }
        // cv::imshow("Harris Corners", harrisFrame);

        // Look for quit
        if (prev_key == 'q') {
            std::cout << "q key detected...quitting " << std::endl;
            break;
        }

    }

    delete capdev;
    return (0);
}