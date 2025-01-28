/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  Read an image from a file and display it
*/

#include<iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[])
{
    std::string image_path = cv::samples::findFile("C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_1\\winterpark.jpg");
    std::cout << "Opening image..." << std::endl;
    cv::Mat image;
    image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) 
    { 
        std::cerr << "Could not open or find the image: " << image_path << std::endl; 
        return -1; 
    } 

    // Resize the image 
    cv::Mat resizedImage; 
    double scaleFactor = 0.25; // Resize to 25% of the original size 
    cv::resize(image, resizedImage, cv::Size(), scaleFactor, scaleFactor);

    imshow("Display Window", resizedImage);
    int k = cv::waitKey(0);

    // Quit when 'q' key is pressed
    if (k == 'q')
    {
        cv::imwrite("winterpark.jpg", image); // Wait for a keystroke in the window
    }
    return 0;
}