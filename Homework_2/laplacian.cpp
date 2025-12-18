/*
  Samara Holmes
  Spring 2025
*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <time.h>
#include "opencv2/opencv.hpp"  // the top include file
#include <corecrt_math_defines.h>

/*
  To visualize the Fourier Transform of the image after multiplying it by a Laplacian in the Fourier domain, you need to follow these steps:

    1. Compute the Discrete Fourier Transform (DFT) of the image.

    2. Multiply the DFT of the image by the DFT of the Laplacians filter.

    3. Visualize the resulting Fourier Transform.

  cv::Mat fft is the original DFT of an image

  cv::Mat mag holds the image for visualization on output, it will be of type CV_32F
 */
int visPowerSpectrum( cv::Mat &fft, cv::Mat &mag ) {
  
  // in order to visualize the spectrum, we compute the magnitude of the complex number and take the log
  mag.create(fft.size(), CV_32F );

  // compute the magnitude and the log
  for(int i=0;i<fft.rows;i++) {
    float *data = fft.ptr<float>(i);
    float *mptr = mag.ptr<float>(i);
    for(int j=0;j<fft.cols;j++) {
      float x = data[j*2];
      float y = data[j*2 + 1];
      mptr[j] = log( 1 + sqrt(x*x + y*y) );
    }
  }

  cv::normalize( mag, mag, 0, 1, cv::NORM_MINMAX );

  // reorganize the quadrants to be centered on the middle of the image
  int cx = mag.cols/2;
  int cy = mag.rows/2;

  cv::Mat q0(mag, cv::Rect( 0, 0, cx, cy ) ); // x, y, width, height
  cv::Mat q1(mag, cv::Rect( cx, 0, cx, cy ) );
  cv::Mat q2(mag, cv::Rect( 0, cy, cx, cy ) );
  cv::Mat q3(mag, cv::Rect( cx, cy, cx, cy ) );

  cv::Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  return(0);
}


// Function to create a Laplacian filter in the frequency domain 
void buildLaplacianFilterImage(int rows, int cols, cv::Mat &laplacianFilter) 
{ 
  laplacianFilter = cv::Mat::zeros(cv::Size(cols, rows), CV_32F); 
  int cx = cols / 2; 
  int cy = rows / 2; 
  for (int i = 0; i < rows; i++) 
  { 
    float *data = laplacianFilter.ptr<float>(i); 
    for (int j = 0; j < cols; j++) 
    { 
      float dx = static_cast<float>(j - cx); 
      float dy = static_cast<float>(i - cy); 
      data[j] = -1.0f * (dx * dx + dy * dy); 
      } 
  }
  // Shift the quadrants of the filter so that the origin is at the image center 
  cv::Mat tmp; 
  cv::Mat q0(laplacianFilter, cv::Rect(0, 0, cx, cy)); // Top-Left 
  cv::Mat q1(laplacianFilter, cv::Rect(cx, 0, cx, cy)); // Top-Right 
  cv::Mat q2(laplacianFilter, cv::Rect(0, cy, cx, cy)); // Bottom-Left 
  cv::Mat q3(laplacianFilter, cv::Rect(cx, cy, cx, cy)); // Bottom-Right 
  q0.copyTo(tmp);
  q3.copyTo(q0); 
  tmp.copyTo(q3); 
  q1.copyTo(tmp); 
  q2.copyTo(q1); 
  tmp.copyTo(q2);
}

// reading an image (path on command line), modifying it
int main( int argc, char *argv[] ) {
  cv::Mat src; // standard image data type
  cv::Mat grey; // greyscale version of the image
  cv::Mat paddedGrey; 

  std::string filename = cv::samples::findFile("C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Homework_2\\cathedral.jpeg");
  std::cout << "Opening image..." << std::endl;

  // read the image
  src = cv::imread( filename ); // allocates the image data, reads as a BGR 8-bit per channel image

  // test of the image read was successful
  if( src.data == NULL ) { // no image data read from file
    printf("error: unable to read image %s\n", filename );
    exit(-1);
  }

  // convert the image to greyscale (easier to deal with 1-D data )
  cv::cvtColor( src, grey, cv::COLOR_BGR2GRAY );

  cv::namedWindow( filename, 1 );
  cv::imshow( filename, grey );

  // The fast DFT wants images to be nicely sized
  int m = cv::getOptimalDFTSize( grey.rows );
  int n = cv::getOptimalDFTSize( grey.cols );
  printf("Resizing image to %d x %d with zero padding\n", m, n);
  cv::copyMakeBorder( grey, paddedGrey, 0, m - grey.rows, n - grey.cols, cv::BORDER_CONSTANT, 0 );

  cv::namedWindow( "padded", 2 );
  cv::imshow( "padded", paddedGrey );

  // make a 2-channel 32-bit float array of images, with zeros for the second band
  cv::Mat planes[] = { cv::Mat_<float>(paddedGrey), cv::Mat::zeros(paddedGrey.size(), CV_32F ) };
  cv::Mat complex;
  cv::Mat fft;
  cv::merge( planes, 2, complex );

  // take the discrete Fourier transform of the image
  cv::dft( complex, fft ); 

  // in order to visualize the spectrum, we compute the magnitude of the complex number and take the log
  cv::Mat mag;
  visPowerSpectrum( fft, mag );
  cv::imshow("DFT2", mag );
  
  double sigma = 10.0; // sigma for the Gaussian
  cv::Mat laplacianFilter;
  char key = 0;

  // make a Laplacian filter image
  buildLaplacianFilterImage(src.rows, src.cols, laplacianFilter);
  
  // start a loop here
  for(;key != 'q';) {

    // get any keypresses
    key = cv::waitKey(10);

    // quit on a q
    if (key == 'q'){
      break;
    }

    // visualize the filter
    cv::Mat vis;
    laplacianFilter.copyTo(vis);
    cv::normalize( vis, vis, 0, 1, cv::NORM_MINMAX );
    cv::imshow("Laplacian Filter", vis);

    // take the dft of the Laplacian image
    cv::Mat gplanes[] = { cv::Mat_<float>(laplacianFilter), cv::Mat::zeros(laplacianFilter.size(), CV_32F ) };
    cv::Mat gcomp;
    cv::merge( gplanes, 2, gcomp );

    cv::dft( gcomp, gcomp );

    // visualize the Laplacian filter FT
    visPowerSpectrum( gcomp, mag );
    cv::imshow("Laplacian DFT", mag );

    // multiply the two images
    cv::Mat product;
    product.create( fft.size(), fft.type() );
    cv::mulSpectrums( fft, gcomp, product, 0 ); // use this function b/c the data format can be complicated

    // visualize the product spectrum
    visPowerSpectrum( product, mag );
    cv::imshow( "product FFT", mag );
  
    // take the inverse DFT
    cv::Mat filtered;
    cv::idft( product, filtered, cv::DFT_SCALE ); // dft scale sets it back to the original intensity

    cv::split( filtered, planes );
    cv::Mat result = cv::Mat_<unsigned char>( planes[0] );

    // Display the filtered real image
    cv::imshow( "Filtered Real", result );

    // Add code to visualize the Fourier Transform of the image after Laplacian multiplication
    visPowerSpectrum(filtered, mag);
    cv::imshow("Filtered Real FT", mag);
  }

  printf("Terminating\n");

  return(0);
}
