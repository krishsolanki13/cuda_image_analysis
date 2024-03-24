#include <stdio.h>

#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2
#define BLOCK_SIZE 16

// Define the input image dimensions
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512

// Define the struct for 2D float image
typedef struct {
    float *data;
    int width;
    int height;
} Image;

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

// Host function to perform dilation on CPU
void dilationCPU(float *input, float *output, int width, int height) {
    for (int ty = 0; ty < height; ++ty) {
        for (int tx = 0; tx < width; ++tx) {
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
}

int main() {
    // Allocate memory for input and output images
    float *h_input = (float*)malloc(sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT);
    float *h_output = (float*)malloc(sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT);
    float *d_input, *d_output;

    // Initialize input image
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
        h_input[i] = (float)(i % 256); // Just some random values
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_input, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT);
    cudaMalloc((void**)&d_output, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT);

    // Copy input image to device
    cudaMemcpy(d_input, h_input, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((IMAGE_WIDTH + dimBlock.x - 1) / dimBlock.x, (IMAGE_HEIGHT + dimBlock.y - 1) / dimBlock.y);

    // Call CUDA kernel
    dilation<<<dimGrid, dimBlock>>>(d_input, d_output, IMAGE_WIDTH, IMAGE_HEIGHT);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Perform dilation on CPU for comparison
    //float *h_output_cpu = (float*)malloc(sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT);
    //dilationCPU(h_input, h_output_cpu, IMAGE_WIDTH, IMAGE_HEIGHT);

    // Verify results
    //for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
    //    if (h_output[i] != h_output_cpu[i]) {
    //        printf("Mismatch found at index %d\n", i);
    //        break;
    //    }
    //}

    // Free host memory
    free(h_input);
    free(h_output);
    //free(h_output_cpu);

    printf("Done\n")

    return 0;
}
