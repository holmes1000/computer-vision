/*
  Samara Holmes
  Spring 2025
  CS 5330 Computer Vision

  YOLOv11 on live video stream

*/

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>

int main(int argc, char *argv[]) {
    cv::VideoCapture capdev;
    cv::Mat frame;

    // Load the YOLOv11 model
    std::string model_path = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Project_3\\yolo11.onnx";
    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);

    // Check if the network was loaded successfully
    if (net.empty()) {
        std::cerr << "Failed to load the model from " << model_path << std::endl;
        return -1;
    }

    // Open the video device
    capdev.open(0, cv::CAP_DSHOW); // 0 for usb cam
    if (!capdev.isOpened()) {
        std::cerr << "Unable to open video device" << std::endl;
        return -1;
    }

    cv::Size refS((int) capdev.get(cv::CAP_PROP_FRAME_WIDTH), (int) capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Expected size: " << refS.width << " " << refS.height << std::endl;

    cv::namedWindow("Video", 1);

    while (true) {
        // Capture the next frame
        capdev >> frame;
        if (frame.empty()) {
            std::cerr << "Frame is empty" << std::endl;
            break;
        }

        // Preprocess the frame
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0/255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);

        // Set the network input
        net.setInput(blob);

        // Run the model
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Post-process the outputs
        for (const auto& output : outputs) {  // Iterate through output layers
            for (int i = 0; i < output.rows; i++) {  // Iterate through detections
                float confidence = output.at<float>(i, 4);  // Confidence score

                if (confidence > 0.2) {  // Confidence threshold (lower it for testing)
                    int probability_start = 5; // Class probabilities start at index 5
                    cv::Mat scores = output.row(i).colRange(probability_start, output.cols);
                    cv::Point classIdPoint;
                    double maxConfidence;
                    cv::minMaxLoc(scores, nullptr, &maxConfidence, nullptr, &classIdPoint);

                    int class_id = classIdPoint.x;  // The class with the highest probability

                    // Extract bounding box
                    float x_center = output.at<float>(i, 0) * frame.cols;
                    float y_center = output.at<float>(i, 1) * frame.rows;
                    float width = output.at<float>(i, 2) * frame.cols;
                    float height = output.at<float>(i, 3) * frame.rows;
                    int left = static_cast<int>(x_center - width / 2);
                    int top = static_cast<int>(y_center - height / 2);

                    cv::rectangle(frame, cv::Rect(left, top, width, height), cv::Scalar(0, 255, 0), 2);
                    std::string label = std::to_string(class_id) + ": " + std::to_string(maxConfidence);
                    cv::putText(frame, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        // Display the image
        cv::imshow("Video", frame);

        // Terminate if the user types 'q'
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
    }

    std::cout << "Terminating" << std::endl;

    return 0;
}


