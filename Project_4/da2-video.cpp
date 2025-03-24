/*
  Bruce A. Maxwell, Samara Holmes
  January 2025

  An example of using the DA2Network class with OpenCV for a video stream

*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"

// opens a video stream and runs it through the depth anything network
// displays both the original video stream and the depth stream
int main(int argc, char *argv[]) {
  // cv::VideoCapture *capdev; 
  cv::VideoCapture capdev;
  cv::VideoWriter videoWriter;
  cv::Mat src; 
  cv::Mat dst;
  cv::Mat dst_vis;
  const float reduction = 0.9;

  // make a DANetwork object
  DA2Network da_net( "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_3\\model_fp16.onnx" );

  // open the video device
  // capdev = new cv::VideoCapture(1);
  // if( !capdev->isOpened() ) {
  //   printf("Unable to open video device\n");
  //   return(-1);
  // }


  // open the video file
  std::string videoFilePath = "C:\\Users\\samar\\Desktop\\computer-vision\\avatar\\avatar-clip.mp4";
  capdev.open(videoFilePath);
  if (!capdev.isOpened()) {
    std::cout << "Unable to open video file: " << videoFilePath << std::endl;
    return -1;
  }
  // cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
	// 	 (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  cv::Size refS( (int) capdev.get(cv::CAP_PROP_FRAME_WIDTH ),
		 (int) capdev.get(cv::CAP_PROP_FRAME_HEIGHT));

  printf("Expected size: %d %d\n", refS.width, refS.height);

  float scale_factor = 512.0 / (refS.height*reduction);
  printf("Using scale factor %.2f\n", scale_factor);

  // Initialize VideoWriter
  std::string outputVideoFilePath = "C:\\Users\\samar\\Desktop\\computer-vision\\avatar\\output\\depth_video.mp4";
  int codec = cv::VideoWriter::fourcc('M', 'P', '4', 'V'); // mp4 codec
  double fps = capdev.get(cv::CAP_PROP_FPS);
  cv::Size frameSize(refS.width * reduction, refS.height * reduction);
  videoWriter.open(outputVideoFilePath, codec, fps, frameSize, true);

  if (!videoWriter.isOpened()) {
    std::cout << "Unable to open video writer for file: " << outputVideoFilePath << std::endl;
    return -1;
  }

  cv::namedWindow( "Video", 1 );
  cv::namedWindow( "Depth", 2 );

  for(;;) {
    // capture the next frame
    // *capdev >> src;
    capdev >> src;
    if( src.empty()) {
      printf("frame is empty\n");
      break;
    }
    // for speed purposes, reduce the size of the input frame by half
    cv::resize( src, src, cv::Size(), reduction, reduction );

    // set the network input
    da_net.set_input( src, scale_factor );

    // run the network
    da_net.run_network( dst, src.size() );

    // apply a color map to the depth output to get a good visualization
    cv::applyColorMap(dst, dst_vis, cv::COLORMAP_INFERNO );

    // if you want to modify the src image based on the depth image, do that here
    /*
    for(int i=0;i<src.rows;i++) {
      for(int j=0;j<src.cols;j++) {
	if( dst.at<unsigned char>(i, j) < 128 ) {
	  src.at<cv::Vec3b>(i,j) = cv::Vec3b( 128, 100, 140 );
	}
      }
    }
    */

    // write the depth visualization frame to the video file
    videoWriter.write(dst_vis);

    // display the images
    cv::imshow("Video", src);
    cv::imshow("Depth", dst_vis);

    // terminate if the user types 'q'
    char key = cv::waitKey(10);
    if( key == 'q' ) {
      break;
    }
  }

  printf("Terminating\n");

  return(0);
}

