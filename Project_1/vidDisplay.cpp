/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  Capture live video from a camera and add effects to it
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters.h"
#include "faceDetect.h"

int main()
{
    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
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

        // Switch statement containing all the functions
        switch (prev_key) 
        {
            // Quit
            case 'q':
                break;
            // cvtColor version of the greyscale
            case 'g':
                cv::cvtColor(new_frame, new_frame, cv::COLOR_BGR2GRAY);
                break;
             // Greyscale custom version
            case 'h':
                greyscale(new_frame, new_frame);
                break;
             // Sepia filter
            case 's':
                sepia(new_frame, new_frame);
                break;
            // 5x5 blur filter
            case 'p':
                blur5x5_1(new_frame, new_frame);
                break;
            // Faster 5x5 blur filter
            case 'b':
                blur5x5_2(new_frame, new_frame);
                break;
            // Sobel X
            case 'x':
                sobelX3x3(new_frame, new_frame);
                break;
             // Sobel Y
            case 'y':
                sobelY3x3(new_frame, new_frame);
                break;
             // Magnitude
            case 'm':
            {
                // Store Sobel results
                cv::Mat sobelX;
                cv::Mat sobelY;

                // Apply Sobel filters to new_frame and store results in sobelX and sobelY
                sobelX3x3(new_frame, sobelX);
                sobelY3x3(new_frame, sobelY);

                // Convert sobelX and sobelY to CV_16SC3 !!
                sobelX.convertTo(sobelX, CV_16SC3);
                sobelY.convertTo(sobelY, CV_16SC3);

                // Calculate the magnitude of the gradients
                magnitude(sobelX, sobelY, new_frame);
            }
            break;
            // Blurs and quantizes a color image
            case 'l': // This is a lowercase L,
                blurQuantize(new_frame, new_frame, level);
                break;
            // Detect faces
            case 'f':
                // Convert to greyscale first
                cv::cvtColor(new_frame, new_frame, cv::COLOR_BGR2GRAY);
                detectFaces(new_frame, faces);
                 // draw boxes around the faces
                drawBoxes(new_frame, faces);
                break;
            // Extra effect 1: Put the detected face in color, backgorund greyscale
            case '1':
            {
                // Convert to greyscale first
                cv::Mat greyFrame;
                cv::cvtColor(new_frame, greyFrame, cv::COLOR_BGR2GRAY);
                detectFaces(greyFrame, faces);
                // Draw boxes around the faces
                drawBoxes(greyFrame, faces);
                cv::cvtColor(greyFrame, greyFrame, cv::COLOR_GRAY2BGR);
                for (const auto& face : faces) {
                    new_frame(face).copyTo(greyFrame(face));
                }
                // Display in color
                cv::imshow("Video", greyFrame);
            }
                break;
            // Extra effect 2: Put the detected face in color, backgorund blurred
            case '2':
                {
                // Run face detection on greyscale first
                cv::Mat greyFrame;
                cv::cvtColor(new_frame, greyFrame, cv::COLOR_BGR2GRAY);
                detectFaces(greyFrame, faces);

                // Get the blur
                cv::Mat blurFrame;
                blur5x5_2(new_frame, blurFrame);

                for (const auto& face : faces) {
                    new_frame(face).copyTo(blurFrame(face));
                }
                // Display in color
                cv::imshow("Video", blurFrame);
            }
                break;
            // Extra effect 3: Pick a strong color to remain and set everything else to greyscale
            case 'c':
                keepOneColor(new_frame, new_frame);
                break;
            // Extra effect 4: vignette
            case 'v':
                vignetting(new_frame, new_frame, level);
                break;
            // Save the frame as an image
            case 'k':
            {
                std::string filename = "images/image_" + std::to_string(img_count++) + ".png";
                cv::Mat image;
                bool ret = capdev->read(image);
                if (ret) {
                    cv::imwrite(filename, image);
                    std::cout << "Image saved: " << filename << std::endl;   
                }
                else {
                    std::cerr << "Failed to save image" << std::endl;
                }
                std::cout << "K pressed, saving image, moving back to no filter" << std::endl;
                prev_key = 'n';
            }
            break;
            default:
                cv::imshow("Video", frame);
                break;
        }

        // Look for new key press
        char key = cv::waitKey(10);
        if (key != -1) { 
            std::cout << "New key press detected: " << key << std::endl;
            prev_key = key; 
        }


        // Make sure to display the new frame
        cv::imshow("Video", new_frame);

        // Look for quit
        if (prev_key == 'q') {
            std::cout << "q key detected...quitting " << std::endl;
            break;
        }

    }

    delete capdev;
    return (0);
}