
#include "medianFilter.h"

__global__ void rgbaToGreyscaleGPU(
    uchar4 *rgbaImage, 
    unsigned char *greyImage,
    int rows,
    int cols
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > cols || y > rows)
    {
        return;
    }

    uchar4 rgba = rgbaImage[y * cols + x];
    unsigned char greyValue =  (0.299f * rgba.x) + (0.587f * rgba.y) + (0.114f * rgba.z);
    greyImage[y * cols + x] = greyValue;
}

__global__ void medianFilterGPU(
    unsigned char* greyImageData, 
    unsigned char *filteredImage, 
    int rows, 
    int cols
)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int windowSize = 3;

    int filter[9] {
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
    };

    unsigned char pixelValues[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    if (
        x > cols - windowSize + 1 ||
        y > rows - windowSize + 1 ||
        x < windowSize - 1 ||
        y < windowSize - 1
    )
    {
        return;
    }

    for (int hh = 0; hh < windowSize; hh++) 
    {
        for (int ww = 0; ww < windowSize; ww++) 
        {
            if (filter[hh * windowSize + ww] == 1)
            {
                int idx = (y + hh - 1) * cols + (x + ww - 1);
                pixelValues[hh * windowSize + ww] = greyImageData[idx];
            }
        }
    }

    // Get median pixel value and assign to filteredImage
    for (int i = 0; i < (windowSize * windowSize); i++) {
	    for (int j = i + 1; j < (windowSize * windowSize); j++) {
	        if (pixelValues[i] > pixelValues[j]) {
		        //Swap the variables.
		        char tmp = pixelValues[i];
		        pixelValues[i] = pixelValues[j];
		        pixelValues[j] = tmp;
	        }
	    }
    }

    unsigned char filteredValue = pixelValues[(windowSize * windowSize) / 2];
    filteredImage[y * cols + x] = filteredValue;
}

