/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  Capture live video from a camera and add effects to it
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters.h"
// #include <C://onnxruntime-win-x64-1.17.1//include//onnxruntime_cxx_api.h>
// #include <C://onnxruntime-win-x64-1.17.1//include//onnxruntime_c_api.h>
// #include <C://onnxruntime-win-x64-1.17.1//include//cpu_provider_factory.h>

int main()
{
    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(1);
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
    // cv::namedWindow("Thresholded Video", cv::WINDOW_AUTOSIZE); // Create another window for the thresholded video
    // cv::namedWindow("Morphological Video", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Region Map", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("DepthAnythingV2", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("NeuFlowV2", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    int img_count = 0;
    char prev_key = -1;
    int level = 6;
    std::vector<cv::Rect> faces;

    for (;;)
    {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty())
        {
            printf("frame is empty\n");
            break;
        }

        // Apply modifications based on the last key pressed 
        cv::Mat new_frame = frame.clone();
        cv::Mat threshFrame, morphFrame, connectedComponentFrame, depthAnythingFrame;

        switch (prev_key) 
        {
            // Quit
            case 'q':
                break;
            // Go into training mode
            case 'n':
                break;
            // Evaluate performance
            case 'e':
                evaluatePerformance();
                break;

            // NEUFlow
            case 'f':
            {
                break;
            }
        }

        // thresholding
        // threshold(new_frame, threshFrame);

        // morphological filtering
        // morphological(new_frame, morphFrame);

        // connected components analysis
        //connectedComponentAnaylsis(new_frame, connectedComponentFrame);

        // moment calculations
        // computeFeaturesForRegions();

        // performance evaluation

        // DepthAnything
        depthAnything(new_frame, depthAnythingFrame);


        // Look for new key press
        char key = cv::waitKey(10);
        if (key != -1) { 
            std::cout << "New key press detected: " << key << std::endl;
            prev_key = key; 
        }

        // Make sure to display the new frames and the OG
        // cv::imshow("Video", frame);
        // cv::imshow("Thresholded Video", threshFrame);
        // cv::imshow("Morphological Video", morphFrame);
        //cv::imshow("Region Map", connectedComponentFrame);
        // cv::imshow("NeuFlowV2", neuflowv2);
        cv::imshow("DepthAnythingV2", depthAnythingFrame);

        // Look for quit
        if (prev_key == 'q') {
            std::cout << "q key detected...quitting " << std::endl;
            break;
        }

    }

    delete capdev;
    return (0);
}