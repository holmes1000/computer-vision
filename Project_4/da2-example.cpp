/*
  Bruce A. Maxwell, Samara Holmes
  January 2025

  An example of using the DA2Network class with OpenCV.

  Reads an image provided on the command line, then applies it to the
  network and shows the original image and the depth image.

  Modified to save the image to a directory

*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"

int main(int argc, char *argv[]) {
  cv::Mat src; 
  cv::Mat dst;
  cv::Mat dst_vis;
  std::string filename; // a string for the filename

  // usage: checking if the user provided a filename
  // if(argc < 2) {
  //   printf("Usage %s <image filename>\n", argv[0]);
  //   exit(-1);
  // }
  //filename = argv[1]; // copying 2nd command line argument to filename variable
  // filename = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\DA_Example\\gull.jpeg";
  filename = "C:\\Users\\samar\\Desktop\\computer-vision\\images\\DSC09747.jpg";
  // read the image, assuming 3-channel BGR
  src = cv::imread(filename); 

  // test if the read was successful
  if(src.data == NULL) {  // src.data is the reference to the image data
    printf("Unable to read image %s\n", filename);
    exit(-1);
  }
  
  // make a DANetwork object, if you use a different network, you have
  // to include the input and output layer names
  // DA2Network da_net( "model_fp16.onnx" );
  // DA2Network da_net( "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_3\\model_fp16.onnx" );
  DA2Network da_net( "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_3\\neuflow_mixed.onnx" );
  
  // scale the network input so it's not larger than 512 on the small side
  float scale_factor = 512.0 / (src.rows > src.cols ? src.cols : src.rows);
  scale_factor = scale_factor > 1.0 ? 1.0 : scale_factor;

  printf("Scale factor: %.2f\n", scale_factor);

  // set up the network input
  da_net.set_input( src, scale_factor );

  // see how big the network input image is
  printf("Finished setting input: %d %d\n", da_net.in_height(), da_net.in_width() );

  // run the network
  printf("Running the network");
  da_net.run_network( dst, src.size() );

  // apply a color map to the depth outputs
  printf("color map");
  cv::applyColorMap(dst, dst_vis, cv::COLORMAP_INFERNO );

  // show the original and depth images
  cv::imshow(filename, src);
  cv::imshow("depth", dst_vis);
  // cv::imwrite("C:\\Users\\samar\\Desktop\\computer-vision\\images\\output\\depth_image.png", dst_vis);

  // wait for the user to press a key
  cv::waitKey(0);

  // cv::imwrite("depth_image.png", dst_vis);
  // Debugging information
  if (!dst_vis.empty()) {
    std::string outputPath = "C:\\Users\\samar\\Desktop\\computer-vision\\images\\output\\depth_image_DSC097472.png";
    if (cv::imwrite(outputPath, dst_vis)) {
      std::cout << "Depth image saved successfully at " << outputPath << std::endl;
    } else {
        std::cout << "Failed to save depth image at " << outputPath << std::endl;
    }
  } else {
    std::cout << "Depth image is empty, not saving." << std::endl;
  }


  printf("Terminating\n");

  return(0);
}

