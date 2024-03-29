#include <cuda_runtime.h>
#include <iostream>
#include <ctime>
#include "EBMP/EasyBMP.h"
#include <algorithm>

using namespace std;

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void saveImage(float* image, int height, int width, bool method) {
    BMP Output;
    Output.SetSize(width, height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            RGBApixel pixel;
            pixel.Red = image[i * width + j];
            pixel.Green = image[i * width + j];
            pixel.Blue = image[i * width + j];
            Output.SetPixel(j, i, pixel);
        }
    }
    if (method)
        Output.WriteToFile("GPUoutAngelina.bmp");
    else
        Output.WriteToFile("CPUoutAngelina.bmp");
}

void noiseImg(float* image, int height, int width, int per) {
    BMP Output;
    Output.SetSize(width, height);
    int countOfPixels = int(height * width / 100 * per);
    while (countOfPixels > 0) {
        int i = rand() % height;
        int j = rand() % width;
        int c = rand() % 2;
        if (c == 1)
            image[i * width + j] = 255;
        else
            image[i * width + j] = 0;
        countOfPixels--;
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            RGBApixel pixel;
            pixel.Red = image[i * width + j];
            pixel.Green = image[i * width + j];
            pixel.Blue = image[i * width + j];
            Output.SetPixel(j, i, pixel);
        }
    }
    Output.WriteToFile("NoiseAngelina.bmp");
}

void medianFilterCPU(float* image, float* result, int height, int width) {
    int m = 3;
    int n = 3;
    int mean = m * n / 2;
    int pad = m / 2;
    float* expandImageArray = (float*)calloc((height + 2 * pad) * (width + 2 * pad), sizeof(float));
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            expandImageArray[(j + pad) * (width + 2 * pad) + i + pad] = image[j * width + i];
        }
    }
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            float* window = (float*)calloc(m * n, sizeof(float));
            for (int k = 0; k < m; k++) {
                for (int t = 0; t < n; t++) {
                    window[k * n + t] = expandImageArray[j * (width + 2 * pad) + i + k * (width + 2 * pad) + t];
                }
            }
            bool swapped = true;
            int t = 0;
            float tmp;
            while (swapped) {
                swapped = false;
                t++;
                for (int i = 0; i < m * n - t; i++) {
                    if (window[i] > window[i + 1]) {
                        tmp = window[i];
                        window[i] = window[i + 1];
                        window[i + 1] = tmp;
                        swapped = true;
                    }
                }
            }
            result[j * width + i] = window[mean];
            free(window);
        }
    }
    free(expandImageArray);
}
// CUDA kernel to apply the median filter
__global__ void myFilter(cudaTextureObject_t texObj, float* output, int imageWidth, int imageHeight) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col >= imageWidth || row >= imageHeight) return;

    float window[9];
    int m = 3, n = 3, mean = m * n / 2, pad = m / 2;
    for (int i = -pad; i <= pad; i++) {
        for (int j = -pad; j <= pad; j++) {
            window[(i + pad) * n + j + pad] = tex2D<float>(texObj, col + j + 0.5f, row + i + 0.5f);
        }
    }
    // Bubble sort
    for(int i = 0; i < 9 - 1; i++) {
        for(int j = 0; j < 9 - i - 1; j++) {
            if(window[j] > window[j + 1]) {
                // swap temp and window[j]
                float temp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = temp;
            }
        }
    }
    output[row * imageWidth + col] = window[mean];
}

int main(void) {
    int nIter = 100;
    BMP Image;
    Image.ReadFromFile("angelina.bmp");
    int height = Image.TellHeight();
    int width = Image.TellWidth();
    float* imageArray = (float*)calloc(height * width, sizeof(float));
    float* outputCPU = (float*)calloc(height * width, sizeof(float));
    float* outputGPU = (float*)calloc(height * width, sizeof(float));
    float* outputDevice;

    for (int j = 0; j < Image.TellHeight(); j++) {
        for (int i = 0; i < Image.TellWidth(); i++) {
            imageArray[j * width + i] = Image(i, j)->Red;
        }
    }

    noiseImg(imageArray, height, width, 8);

    unsigned int start_time = clock();

    for (int j = 0; j < nIter; j++) {
        medianFilterCPU(imageArray, outputCPU, height, width);
    }

    unsigned int elapsedTime = clock() - start_time;
    float msecPerMatrixMulCpu = elapsedTime / nIter;

    cout << "CPU time: " << msecPerMatrixMulCpu << " ms" << endl;

    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        cout << "Sorry! You don't have a CUDA Device" << endl;
    } else {
        cout << "CUDA Device found! Device count: " << device_count << endl;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaArray* cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width, height));
        size_t width_in_bytes = width * sizeof(float);
        checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, imageArray, width_in_bytes, width_in_bytes, height, cudaMemcpyHostToDevice));


        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t texObj = 0;
        checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        checkCudaErrors(cudaMalloc(&outputDevice, height * width * sizeof(float)));

        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        checkCudaErrors(cudaEventRecord(start, 0));

        for (int j = 0; j < nIter; j++) {
            myFilter<<<blocksPerGrid, threadsPerBlock>>>(texObj, outputDevice, width, height);
        }

        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
        float msecPerMatrixMul = msecTotal / nIter;
        cout << "GPU time: " << msecPerMatrixMul << " ms" << endl;

        checkCudaErrors(cudaMemcpy(outputGPU, outputDevice, height * width * sizeof(float), cudaMemcpyDeviceToHost));
        saveImage(outputGPU, height, width, true);
        saveImage(outputCPU, height, width, false);

        cudaFreeArray(cuArray);
        cudaDestroyTextureObject(texObj);
        cudaFree(outputDevice);
        free(imageArray);
        free(outputCPU);
        free(outputGPU);
    }
    return 0;
}


