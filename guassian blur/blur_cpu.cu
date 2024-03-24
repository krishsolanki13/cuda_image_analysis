#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>

#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2

// CPU function for Gaussian blur operation
void cpuGaussianBlur(const cv::Mat& input, cv::Mat& output) {
    int width = input.cols;
    int height = input.rows;

    // Initialize Gaussian kernel
    float kernel[MASK_WIDTH][MASK_WIDTH];
    float sigma = 1.0f;
    float sum = 0.0f;
    int r = MASK_WIDTH / 2;
    for (int x = -r; x <= r; ++x) {
        for (int y = -r; y <= r; ++y) {
            kernel[x + r][y + r] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[x + r][y + r];
        }
    }
    // Normalize the kernel
    for (int i = 0; i < MASK_WIDTH; ++i) {
        for (int j = 0; j < MASK_WIDTH; ++j) {
            kernel[i][j] /= sum;
        }
    }

    // Apply Gaussian blur
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float newValue = 0.0f;
            for (int i = -r; i <= r; ++i) {
                for (int j = -r; j <= r; ++j) {
                    int newX = x + j;
                    int newY = y + i;
                    if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                        newValue += input.at<uchar>(newY, newX) * kernel[i + r][j + r];
                    }
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(newValue);
        }
    }
}

// CUDA kernel for Gaussian blur operation
__global__ void gaussianBlur(float *input, float *output, int width, int height) {
    // CUDA kernel implementation (you can use existing CUDA libraries for Gaussian blur)
    // ...
}

int main() {
    // Read input image using OpenCV
    cv::Mat inputImage = cv::imread("./Dude.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        printf("Failed to read input image\n");
        return -1;
    }

    // Display input image
    cv::imshow("Input Image", inputImage);
    cv::waitKey(0);

    // Get input image dimensions
    int width = inputImage.cols;
    int height = inputImage.rows;

    // Allocate memory for output images
    cv::Mat outputImageCPU(height, width, CV_8UC1);
    cv::Mat outputImageGPU(height, width, CV_8UC1);

    // Measure CPU execution time
    auto startCPU = std::chrono::high_resolution_clock::now();

    // Perform Gaussian blur using CPU
    cpuGaussianBlur(inputImage, outputImageCPU);

    auto stopCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(stopCPU - startCPU);
    printf("CPU Execution Time: %lld ms\n", durationCPU.count());

    // Allocate memory for input and output images on GPU
    // ...

    // Copy input image to device
    // ...

    // Call CUDA kernel
    // ...

    // Copy result back to host
    // ...

    // Convert output array to output image
    // ...

    // Display output images
    cv::imshow("Blurred Image (CPU)", outputImageCPU);
    cv::imshow("Blurred Image (GPU)", outputImageGPU);
    cv::waitKey(0);

    // Save output images
    // ...

    return 0;
}
