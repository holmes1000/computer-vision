/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  Functions for adding filters to a video stream
*/

#include "filters.h"
#include <cmath>
#include <corecrt_math_defines.h>


/*
  An alternative greyscale function to cv::cvtColor(new_frame, new_frame, cv::COLOR_BGR2GRAY);
  Greyscale formula = 0.299R + 0.587G + 0.114B
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the greyscale result
 */
int greyscale( cv::Mat &src, cv::Mat &dst ) {
    std::cout << "Running greyscale filter" << std::endl;
    std::vector<cv::Mat> channels(3);
    cv::split(src, channels); // Split the source image into 3 channels

    dst = 0.299*channels[2] + 0.587*channels[1] + 0.114*channels[0];
    
    return 0;
}

/*
  Applies a sepia tone filter
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the greyscale result
 */
int sepia( cv::Mat &src, cv::Mat &dst ) {
    std::cout << "Running sepia tone filter" << std::endl;
    // Define the tone constants
    float color_constants[3][3] = {
        {0.272, 0.534, 0.131},  // Blue (RGB order)
        {0.349, 0.686, 0.168},  // Green (RGB order)
        {0.393, 0.769, 0.189}   // Red (RGB order)
    };

    std::vector<cv::Mat> channels(3);
    std::vector<cv::Mat> new_channels(3);
    cv::split(src, channels); // Split the source image into 3 channels (original values)
    
    // New colors
    new_channels[2] = color_constants[2][0]*channels[2] + color_constants[2][1]*channels[1] + color_constants[2][2]*channels[0];     // Red
    new_channels[1] = color_constants[1][0]*channels[2] + color_constants[1][1]*channels[1] + color_constants[1][2]*channels[0];     // Green
    new_channels[0] = color_constants[0][0]*channels[2] + color_constants[0][1]*channels[1] + color_constants[0][2]*channels[0];     // Blue

    // Apply the new channels into the destination image
    cv::merge(new_channels, dst);

    // Make sure values are within [0, 255]
    dst.convertTo(dst, CV_8UC3);

    return 0;
}

/*
  A 5x5 blur filter from scratch using a single nested loop
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the 5x5 blur filter result
 */
int blur5x5_1( cv::Mat &src, cv::Mat &dst ) {
    std::cout << "Running 5x5 blur filter" << std::endl;
    // Define the 5x5 blur kernel
    int kernel[5][5] = {
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1}
    
    };

    // Apply the 5x5 blur kernel to each pixel
    for (int y = 2; y < src.rows - 2; ++y) {
        for (int x = 2; x < src.cols - 2; ++x) {
            cv::Vec3i sum = cv::Vec3i(0, 0, 0);

            // Apply kernel to the neighboring pixels
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);
                    sum[0] += pixel[0] * kernel[ky + 2][kx + 2];
                    sum[1] += pixel[1] * kernel[ky + 2][kx + 2];
                    sum[2] += pixel[2] * kernel[ky + 2][kx + 2];
                }
            }

            // Assign the averaged value to the destination image
            dst.at<cv::Vec3b>(y, x)[0] = cv::saturate_cast<uchar>(sum[0] / 25);
            dst.at<cv::Vec3b>(y, x)[1] = cv::saturate_cast<uchar>(sum[1] / 25);
            dst.at<cv::Vec3b>(y, x)[2] = cv::saturate_cast<uchar>(sum[2] / 25);
        }
    }

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
    std::cout << "Running faster 5x5 blur filter" << std::endl;
    cv::Mat tmp;
    src.copyTo(tmp);
    dst.create(src.size(), src.type());

    // Apply the vert and horiz blur
    vert_blur(src, tmp);
    horiz_blur(tmp, dst);
    return 0;
}

/*
  Gets the 3x3 Sobel filter horizontal (X) result
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the 3x3 Sobel filter horizontal (X) result
 */
int sobelX3x3( cv::Mat &src, cv::Mat &dst ) {
    std::cout << "Running Sobel X filter" << std::endl;

    // Define the 3x3 Sobel X kernel
    int kernel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    cv::Mat sobelx = cv::Mat::zeros(src.size(), CV_16SC3);
    
    // Apply the kernel
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            cv::Vec3s sum = {0, 0, 0};
            // Apply the kernel to the neighbors the current pixel
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    for (int i = 0; i < 3; i++) {     // Loop for each color
                        sum[i] += src.at<cv::Vec3b>(y + ky, x + kx)[i] * kernel_x[ky + 1][kx + 1];
                    }
                }
            }
            sobelx.at<cv::Vec3s>(y,x) = sum;
        }
    }

    cv::convertScaleAbs(sobelx, dst);

    return 0;
}

/*
  Gets the 3x3 Sobel filter vertical (Y) result
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image to store the 3x3 Sobel filter vertical (Y) result
 */
int sobelY3x3( cv::Mat &src, cv::Mat &dst ) {
    std::cout << "Running Sobel Y filter" << std::endl;
    // Define the 3x3 Sobel Y kernel
    int kernel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    //std::cout << "src type: " << src.type() << std::endl;
    cv::Mat sobely = cv::Mat::zeros(src.size(), CV_16SC3);
    
    // Apply the kernel
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            cv::Vec3s sum = {0, 0, 0};
            // Apply the kernel to the neighbors the current pixel
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    for (int i = 0; i < 3; i++) {     // Loop for each color
                        sum[i] += src.at<cv::Vec3b>(y + ky, x + kx)[i] * kernel_y[ky + 1][kx + 1];
                    }
                }
            }
            sobely.at<cv::Vec3s>(y,x) = sum;
        }
    }

    cv::convertScaleAbs(sobely, dst);

    return 0;
}

/*
  Calculates the magnitude of the gradients from the x and y directional gradients
  Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
  Arguments:
  cv::Mat &sx  - a source image representing x-direction gradient (CV_16SC3)
  cv::Mat &sy  - a source image representing y-direction gradient (CV_16SC3)
  cv::Mat &dst - a destination image to store the magnitudes of the gradients
 */
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ) {
    std::cout << "Calculating magnitude of the gradients" << std::endl;
    //std::cout << "sx size " << sx.size() << std::endl;
    CV_Assert(sx.type() == CV_16SC3);
    CV_Assert(sy.type() == CV_16SC3);
    cv::Mat mag_matrix(sx.size(), CV_16SC3);
    // Calculate the magnitude using the Euclidean distance formula
    for (int y = 0; y < sx.rows; ++y) {
        for (int x = 0; x < sx.cols; ++x) {
            // Access the gradient values at (x, y)
            cv::Vec3s gx = sx.at<cv::Vec3s>(y, x);
            cv::Vec3s gy = sy.at<cv::Vec3s>(y, x);
            cv::Vec3s mag; // 3-channel unsigned char for display

            // Calculate the magnitude for each color channel
            for (int i = 0; i < 3; ++i) {
                mag[i] = std::sqrt(static_cast<float>(gx[i] * gx[i] + gy[i] * gy[i]));
            }

            // Assign the computed magnitude to the output image
            mag_matrix.at<cv::Vec3s>(y, x) = mag;
        }
    }
    mag_matrix.convertTo(dst, CV_8UC3);
    return 0;
}

/*
  Applies blur and quantizes the image into a fixed number of levels as specified by a parameter
  Use the functions you have already written to implement this effect
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image
  int levels - number of quantization levels
 */
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ) {
    // Blur the image using the vert and horiz blur 1x5 filters
    vert_blur(src, dst);
    horiz_blur(dst, dst);

    // Calculate the size of each bucket
    float b = 255.0f / levels;

    // Quantize each pixel in the image
    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            cv::Vec3b &color = dst.at<cv::Vec3b>(y, x);
            for (int i = 0; i < 3; ++i) {
                int xt = static_cast<int>(color[i] / b);
                int xf = static_cast<int>(xt * b);
                color[i] = static_cast<uchar>(xf);
            }
        }
    }
    return 0;
}



// Extra effects
/*
  Greys out everything except orange colors
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image
 */
int keepOneColor( cv::Mat &src, cv::Mat &dst ) {
    // Convert to RGB
    cv::Mat rgb;
    cv::cvtColor(src, rgb, cv::COLOR_RGB2BGR);
    // Create a mask for my favorite color
    // https://tuneform.com/tools/color/rgb-color-creator
    cv::Scalar range_upper(230, 113, 30);
    cv::Scalar range_lower(230, 140, 50);
    cv::Mat mask;
    inRange(rgb, range_lower, range_upper, mask);
    cv::bitwise_and(dst, dst, mask);

    // Grey the image
    //cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);

    cv::Mat grey(src.size(), src.type(), cv::Scalar(128,128,128));

    // Combine
    //src.copyTo(dst, mask); 
    grey.copyTo(dst, ~mask);

    return(0);
}

/*
  Vignettes the video stream
  Arguments:
  cv::Mat &src  - a source image to convert
  cv::Mat &dst - a destination image
  int levels - level of vignette
 */
int vignetting( cv::Mat &src, cv::Mat &dst, int levels) {
    // Create a vignette mask
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_32F);
    int rows = src.rows;
    int cols = src.cols;
    cv::Point center = cv::Point(cols / 2, rows / 2);
    double maxDist = sqrt(center.x * center.x + center.y * center.y);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            double dx = std::min(x, cols - x - 1); // Distance from the closest vert edge
            double dy = std::min(y, rows - y - 1); // Distance from the closest horiz edge
            double dist = sqrt(dx * dx + dy * dy);
            float val = static_cast<float>(levels * 0.25 * (dist / maxDist));
            mask.at<float>(y, x) = std::max(0.0f, val);
        }
    }

    // Convert the source image to float
    cv::Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);

    // Create black dst image
    dst = cv::Mat::zeros(src.size(), src.type());

    // Apply the mask to each channel
    std::vector<cv::Mat> channels(3);
    cv::split(srcFloat, channels);
    for (int i = 0; i < 3; i++) {
        channels[i] = channels[i].mul(mask);
    }

    cv::merge(channels, dst);

    // Convert back to 8-bit image
    dst.convertTo(dst, CV_8U);

    return 0;
}