// medianFilter.h

#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#define THREAD_DIM 256
#define NUM_STREAMS 32
#define MAX_IMAGE_SIZE (1920 * 1080)

typedef struct FakeMat_ {
    unsigned char *Ptr;
    int rows;
    int cols;
} FakeMat;

// CUDA kernel function declarations
__global__ void rgbaToGreyscaleGPU(uchar4 *rgbaImage, unsigned char *greyImage, int rows, int cols);
__global__ void medianFilterGPU(unsigned char* greyImageData, unsigned char *filteredImage, int rows, int cols);

// Utility function declarations
void read_directory(const std::string& name, std::vector<std::string>* v);
int readImage(const std::string& filename, cv::Mat* inputImage, cv::Mat* imageGrey);
void writeImage(const std::string& dirname, const std::string& filename, const std::string& prefix, cv::Mat outputImage);
void printTime(const std::string& task, struct timespec start, struct timespec end);

#endif // MEDIAN_FILTER_H
