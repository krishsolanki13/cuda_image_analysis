#include <iostream>
#include <opencv2/opencv.hpp>

// CUDA kernel for 2D convolution
__global__ void convolutionKernel(const unsigned char* inputImage, unsigned char* outputImage,
                        int width, int height, const float* filter, int filterSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float sum = 0.0f;
        int filterRadius = filterSize / 2;

        for (int i = -filterRadius; i <= filterRadius; ++i) {
            for (int j = -filterRadius; j <= filterRadius; ++j) {
                int currentRow = row + i;
                int currentCol = col + j;

                // Ensure the pixel is inside the image bounds
                if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) {
                    int pixelIndex = currentRow * width + currentCol;
                    int filterIndex = (i + filterRadius) * filterSize + (j + filterRadius);
                    sum += inputImage[pixelIndex] * filter[filterIndex];
                }
            }
        }

        outputImage[row * width + col] = static_cast<unsigned char>(sum);
    }
}

// Sobel filter for edge detection
const float sobelFilter[9] = {
    -1.0f, 0.0f, 1.0f,
    -2.0f, 0.0f, 2.0f,
    -1.0f, 0.0f, 1.0f
};

int main() {
    // Load image using OpenCV
    cv::Mat image = cv::imread("input_image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image." << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate memory for input and output images on the host
    unsigned char* inputImage = image.data;
    unsigned char* outputImage = new unsigned char[imageSize];

    // Allocate memory for input and output images on the device
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;

    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    // Copy input image data from host to device
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel for convolution
    convolutionKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, sobelFilter, 3);

    // Copy the result back from device to host
    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // Free allocated memory on the device
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Create output image using OpenCV
    cv::Mat output(height, width, CV_8UC1, outputImage);

    // Display original and processed images
    cv::imshow("Original Image", image);
    cv::imshow("Edge Detected Image", output);
    cv::waitKey(0);

    // Release memory
    delete[] outputImage;

    return 0;
}