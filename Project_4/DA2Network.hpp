/*
  Bruce A. Maxwell
  January 2025

  A simple wrapper for a Depth Anything V2 Network loaded and run
  using the ONNX Runtime API.  It's currently set up to use the CPU.

  When creating a DA2Network object, pass in the path to the Depth
  Anything network.  If using the included model_fp16.onnx method, use
  the constructor with just the path.

  If you want to use a different DA network, use the other
  constructor, which also requires passing the names of the input
  layer and the output layer.  You can find out these values by
  loading the network into the Netron web-app (netron.app).

  This wrapper is intended for a DA network with dynamic sizing.  It
  seems to work best if you use an image of at least 200x200.  Smaller
  images give pretty approximate results.

  The class handles resizing and normalizing the input image with the set_input function.

  The function run_network applies the current input image to the
  network. The result is resized back to the specified image size.
  The result image is a greyscale image with value sin the range of
  [0..255] with 0 being the minimum depth and 255 being the maximum
  depth.  These are not metric values but are scaled relative to the
  network output.

*/
#include <cstdio>
#include <cstring>
#include <cmath>
#include <array>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class DA2Network {
public:
    // Constructor with just the network pathname, layer names are hard-coded
    DA2Network(const std::string& network_path) {
        network_path_ = network_path;
        input_names_ = "pixel_values"; // default values for the network mode_fp16.onnx
        output_names_ = "predicted_depth";

        // Convert std::string to std::wstring
        std::wstring w_network_path = std::wstring(network_path.begin(), network_path.end());

        // Set up the Ort session
        Ort::SessionOptions session_options;
        session_ = std::make_unique<Ort::Session>(env, w_network_path.c_str(), session_options);

        // Set up the Ort session with GPU execution provider
        // Ort::SessionOptions session_options;
        // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
        // session_ = std::make_unique<Ort::Session>(env, w_network_path.c_str(), session_options);
    }

    // Constructor with both the network path and the layer names
    DA2Network(const std::string& network_path, const std::string& input_layer_name, const std::string& output_layer_name) {
        network_path_ = network_path;
        input_names_ = input_layer_name;
        output_names_ = output_layer_name;

        // Convert std::string to std::wstring
        std::wstring w_network_path = std::wstring(network_path.begin(), network_path.end());

        // Set up the Ort session
        Ort::SessionOptions session_options;
        session_ = std::make_unique<Ort::Session>(env, w_network_path.c_str(), session_options);

        // Set up the Ort session with GPU execution provider
        // Ort::SessionOptions session_options;
        // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
        // session_ = std::make_unique<Ort::Session>(env, w_network_path.c_str(), session_options);
    }

    // Destructor
    ~DA2Network() {
        if (input_data != NULL) {
            delete[] input_data;
        }
    }

    // Accessors
    int in_height() const {
        return height_;
    }

    int in_width() const {
        return width_;
    }

    int out_height() const {
        return out_height_;
    }

    int out_width() const {
        return out_width_;
    }

    // Given a regular image read using cv::imread
    // Rescales and normalizes the image data appropriate for the network
    // scale_factor lets the user resize the image for application to the network
    // smaller images are faster to process, images smaller than 200x200 don't work as well
    int set_input(const cv::Mat& src, const float scale_factor = 1.0) {
        cv::Mat tmp;

        // Check if we need to resize the input image before applying it to the network
        if (scale_factor != 1.0) {
            cv::resize(src, tmp, cv::Size(), scale_factor, scale_factor);
        } else {
            tmp = src;
        }

        // Resize to 224x224 to match the model's expected input dimensions (neuflow)
        cv::resize(tmp, tmp, cv::Size(224, 224));

        // Check if we need to allocate memory for the input tensor
        if (tmp.rows != height_ || tmp.cols != width_) {
            height_ = tmp.rows;
            width_ = tmp.cols;

            if (input_data != NULL) {
                delete[] input_data;
            }

            // Allocate the image data
            input_data = new float[height_ * width_ * 3];
            input_shape_[2] = height_;
            input_shape_[3] = width_;

            // Make the input tensor using the data
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_data, height_ * width_ * 3, input_shape_.data(), input_shape_.size());
            
        }

        // Copy the data over to the input tensor data
        const int image_size = height_ * width_;
        for (int i = 0; i < tmp.rows; ++i) {
            cv::Vec3b* ptr = tmp.ptr<cv::Vec3b>(i);
            float* fptrR = &(input_data[i * width_]);
            float* fptrG = &(input_data[image_size + i * width_]);
            float* fptrB = &(input_data[image_size * 2 + i * width_]);
            for (int j = 0; j < tmp.cols; ++j) {
                fptrR[j] = ((ptr[j][2] / 255.0) - 0.485) / 0.229;
                fptrG[j] = ((ptr[j][1] / 255.0) - 0.456) / 0.224;
                fptrB[j] = ((ptr[j][0] / 255.0) - 0.406) / 0.225;
            }
        }

        // All set to run
        return 0;
    }

    int run_network(cv::Mat& dst, const cv::Size& output_size) {
        std::cout << "In the run_network function" << std::endl;
        if (height_ == 1 || width_ == 1) {
            std::cout << "Input tensor not set up, Terminating" << std::endl;
            exit(-1);
        }

        // Input tensor is already set up in set_input
        Ort::RunOptions run_options;

        // Run the network, it will dynamically allocate the necessary output memory
        std::cout << "Allocate the output memory" << std::endl;
        const char* input_names[] = { input_names_.c_str() };
        const char* output_names[] = { output_names_.c_str() };
        // auto outputTensor = session_->Run(run_options, input_names, &input_tensor_, 1, output_names, 1);
        std::vector<Ort::Value> outputTensor;
        try {
            outputTensor = session_->Run(run_options, input_names, &input_tensor_, 1, output_names, 1);
        } catch (const Ort::Exception& e) {
            std::cerr << "Ort exception: " << e.what() << std::endl;
            return -1;
        }        

        // Get the output data size (not quite the same as the input size)
        std::cout << "Get output data size" << std::endl;
        auto outputInfo = outputTensor[0].GetTensorTypeAndShapeInfo();
        out_height_ = outputInfo.GetShape()[1];
        out_width_ = outputInfo.GetShape()[2];

        // Get the output data
        std::cout << "Get output data" << std::endl;
        const float* tensorData = outputTensor[0].GetTensorData<float>();
        static cv::Mat tmp(out_height_, out_width_, CV_8UC1); // might as well re-use it if possible

        // Get the min and max of the output tensor and copy to a temporary cv::Mat
        std::cout << "Get the min and max of the output tensor" << std::endl;
        float max = -1e+6;
        float min = 1e+6;
        for (int i = 0; i < out_height_ * out_width_; ++i) {
            const float value = tensorData[i];
            min = value < min ? value : min;
            max = value > max ? value : max;
        }

        // Copy the normalized data over to a temporary cv::Mat
        std::cout << "Copy the normalized data over to a temporary cv::Mat" << std::endl;
        for (int i = 0, k = 0; i < out_height_; ++i) {
            unsigned char* ptr = tmp.ptr<unsigned char>(i);
            for (int j = 0; j < out_width_; ++j, ++k) {
                float value = 255 * (tensorData[k] - min) / (max - min);
                ptr[j] = value > 255.0 ? (unsigned char)255 : (unsigned char)value;
            }
        }

        // Reshape tmp to 2D and apply the colormap
        cv::Mat tmp_2d = tmp.reshape(1, out_height_); // Ensure tmp is 2D
        cv::applyColorMap(tmp_2d, dst, cv::COLORMAP_JET); // Apply the colormap

        // Rescale the output to the output size
        std::cout << "Rescale the output to the output size" << std::endl;
        cv::resize(tmp, dst, output_size);

        // outputTensor should de-allocate here automatically

        return 0;
    }

private:
    // Height and width of the most recent input
    int height_ = 0;
    int width_ = 0;

    // Height and width of the most recent output
    int out_height_ = 0;
    int out_width_ = 0;

    // Network path and input/output layer names
    std::string network_path_;
    std::string input_names_; // use Netron.app to see the name of the first layer
    std::string output_names_; // use Netron.app to see the name of the last layer

    // ORT variables
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "DA2Network"};
    std::unique_ptr<Ort::Session> session_;

    // Input data and input tensor variables
    float* input_data = NULL;
    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 4> input_shape_{1, 3, height_, width_}; // batch, channel, height, width: 3-channel color image
};