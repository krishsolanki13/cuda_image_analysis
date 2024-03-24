#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>

#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2

// CPU function for dilation operation
void cpuDilation(const cv::Mat& input, cv::Mat& output) {
    int width = input.cols;
    int height = input.rows;

    // Iterate over each pixel in the image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Compute maximum value in the neighborhood
            float maxVal = 0.0f;
            for (int i = -MASK_RADIUS; i <= MASK_RADIUS; ++i) {
                for (int j = -MASK_RADIUS; j <= MASK_RADIUS; ++j) {
                    int rowIndex = y + i;
                    int colIndex = x + j;
                    if (rowIndex >= 0 && rowIndex < height && colIndex >= 0 && colIndex < width) {
                        float val = static_cast<float>(input.at<uchar>(rowIndex, colIndex));
                        if (val > maxVal)
                            maxVal = val;
                    }
                }
            }
            // Set the output pixel value
            output.at<uchar>(y, x) = static_cast<uchar>(maxVal);
        }
    }
}

int main() {
    // Read input image using OpenCV
    cv::Mat inputImage = cv::imread("./input_image.jpg", cv::IMREAD_GRAYSCALE);
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

    // Perform dilation using CPU
    cpuDilation(inputImage, outputImageCPU);

    auto stopCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(stopCPU - startCPU);
    printf("CPU Execution Time: %lld ms\n", durationCPU.count());

    // Display output images
    cv::imshow("Dilated Image (CPU)", outputImageCPU);
    cv::imshow("Dilated Image (GPU)", outputImageGPU);
    cv::waitKey(0);

    return 0;
}
