/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  Functions for adding filters to a video stream
*/

#include "filters.h"
#include <cmath>
#include <corecrt_math_defines.h>
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
  A 1x5 vertical blur filter
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the 5x5 blur filter result
 */
int vert_blur( cv::Mat &src, cv::Mat &dst ) {
    for (int y = 2; y < src.rows - 2; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                dst.ptr<uchar>(y)[3*x + c] = (src.ptr<uchar>(y-2)[3*x + c] + 2 * src.ptr<uchar>(y-1)[3*x + c] + 
                                              4 * src.ptr<uchar>(y)[3*x + c] + 2 * src.ptr<uchar>(y+1)[3*x + c] + 
                                              src.ptr<uchar>(y+2)[3*x + c]) / 10;
            }
        }
    }
    return 0;
}

/*
  A 1x5 horizontal blur filter
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the 5x5 blur filter result
 */
int horiz_blur( cv::Mat &src, cv::Mat &dst ) {
    for (int y = 0; y < src.rows; ++y) {
        const uchar* srcPtr = src.ptr<uchar>(y);
        uchar* dstPtr = dst.ptr<uchar>(y);

        for (int x = 2; x < src.cols - 2; ++x) {
            for (int c = 0; c < 3; ++c) {
                dstPtr[3*x + c] = (srcPtr[3*(x-2) + c] + 2 * srcPtr[3*(x-1) + c] + 4 * srcPtr[3*x + c] +
                                   2 * srcPtr[3*(x+1) + c] + srcPtr[3*(x+2) + c]) / 10;
            }
        }
    }
    return 0;
}

/*
  A 5x5 blur filter from scratch using separable 1x5 filters (vertical and horizontal) to make the function faster
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the 5x5 blur filter result
 */
int blur5x5_2( cv::Mat &src, cv::Mat &dst ) {
    // std::cout << "Running faster 5x5 blur filter" << std::endl;
    cv::Mat tmp;
    src.copyTo(tmp);
    dst.create(src.size(), src.type());

    // Apply the vert and horiz blur
    vert_blur(src, tmp);
    horiz_blur(tmp, dst);
    return 0;
}

/*
  A thresholding function
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the threshold result
 */
int threshold( cv::Mat &src, cv::Mat &dst ) {
    cv::Mat greyFrame;

    // Convert the frame to greyscale
    // cv::cvtColor(src, greyFrame, cv::COLOR_BGR2GRAY);
    greyscale(src, greyFrame);

    // Apply binary thresholding
    cv::threshold(greyFrame, dst, 128, 255, cv::THRESH_BINARY);

    return 0;
}

/*
  A morphological filter function. Takes an image, thresholds it, then runs morphological functions
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the threshold result
 */
int morphological( cv::Mat &src, cv::Mat &dst ) {
    cv::Mat blurred, thresholdedFrame;

    blur5x5_2(src, blurred);

    threshold(blurred, thresholdedFrame);

    // Apply morphological filtering (erode or dilate)
    cv::dilate(thresholdedFrame, dst, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

    return 0;
}

/*
  Runs connected component analysis with stats from OpenCV library
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the threshold result
 */
int connectedComponentAnaylsis( cv::Mat &src, cv::Mat &dst ) {
    cv::Mat thresholdedFrame;

    threshold(src, thresholdedFrame);

    // Apply morphological filtering (erode or dilate)
    cv::dilate(thresholdedFrame, dst, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7)));

    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(dst, labels, stats, centroids, 8, CV_32S);

    // Display the number of connected components
    std::cout << "Number of connected components: " << numComponents << std::endl;

    return 0;
}

int computeFeaturesForRegions() {
    return 0;
}

int evaluatePerformance() {
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
    const float reduction = 0.25;

    // make a DANetwork object
    DA2Network da_net( "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_3\\model_fp16.onnx" );

    // for speed purposes, reduce the size of the input frame by half
    cv::resize( src, src, cv::Size(), reduction, reduction );
    // printf("Expected size: %d %d\n", src.cols, src.rows);
    float scale_factor = 256.0 / (src.rows*reduction);
    // printf("Using scale factor %.2f\n", scale_factor);

    // set the network input
    da_net.set_input( src, scale_factor );

    // run the network
    da_net.run_network( dst, src.size() );

    // apply a color map to the depth output to get a good visualization
    cv::applyColorMap(dst, dst, cv::COLORMAP_INFERNO );
    return 0;
}
