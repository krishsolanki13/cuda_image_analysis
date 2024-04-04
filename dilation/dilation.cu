#include <stdio.h>
#include <opencv2/opencv.hpp>

#define MASK_WIDTH 15
#define MASK_RADIUS MASK_WIDTH/2
#define BLOCK_SIZE 16

// CUDA kernel for dilation operation
__global__ void dilation(float *input, float *output, int width, int height) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx < width && ty < height) {
        int centerIndex = ty * width + tx;
        float maxVal = 0.0f;
        for (int i = -MASK_RADIUS; i <= MASK_RADIUS; ++i) {
            for (int j = -MASK_RADIUS; j <= MASK_RADIUS; ++j) {
                int rowIndex = ty + i;
                int colIndex = tx + j;
                if (rowIndex >= 0 && rowIndex < height && colIndex >= 0 && colIndex < width) {
                    float val = input[rowIndex * width + colIndex];
                    if (val > maxVal)
                        maxVal = val;
                }
            }
        }
        output[centerIndex] = maxVal;
    }
}

int main() {
    // Read input image using OpenCV
    cv::Mat inputImage = cv::imread("./input_image.png", cv::IMREAD_GRAYSCALE);
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

    // Allocate memory for input and output images
    float *h_input = (float*)malloc(sizeof(float) * width * height);
    float *h_output = (float*)malloc(sizeof(float) * width * height);
    float *d_input, *d_output;

    // Convert input image to float array
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_input[y * width + x] = static_cast<float>(inputImage.at<uchar>(y, x));
        }
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_input, sizeof(float) * width * height);
    cudaMalloc((void**)&d_output, sizeof(float) * width * height);

    // Copy input image to device
    cudaMemcpy(d_input, h_input, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Call CUDA kernel
    dilation<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);

    // Record stop event
    cudaEventRecord(stop);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Kernel Execution Time: %.2f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    // Create output image
    cv::Mat outputImage(height, width, CV_8UC1);

    // Convert output array to output image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            outputImage.at<uchar>(y, x) = static_cast<uchar>(h_output[y * width + x]);
        }
    }

    // Display output image
    cv::imshow("Dilated Image", outputImage);
    cv::waitKey(0);

    // Save output image
    cv::imwrite("output_image.jpg", outputImage);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
