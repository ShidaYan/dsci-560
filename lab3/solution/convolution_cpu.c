#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// 2D Convolution on CPU
void convolution2D_CPU(unsigned char *image, unsigned char *output, 
                        float *kernel, int M, int N, int channels) {
    int half_N = N / 2;
    
    for (int ch = 0; ch < channels; ch++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
                float sum = 0.0f;
                
                // Apply kernel
                for (int ki = 0; ki < N; ki++) {
                    for (int kj = 0; kj < N; kj++) {
                        int ii = i + ki - half_N;
                        int jj = j + kj - half_N;
                        
                        // Handle boundaries with zero-padding
                        if (ii >= 0 && ii < M && jj >= 0 && jj < M) {
                            int img_idx = (ii * M + jj) * channels + ch;
                            int kernel_idx = ki * N + kj;
                            sum += image[img_idx] * kernel[kernel_idx];
                        }
                    }
                }
                
                // Clamp to [0, 255]
                int val = (int)(sum + 0.5f);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                output[(i * M + j) * channels + ch] = (unsigned char)val;
            }
        }
    }
}

// Create common edge detection filters
void create_sobel_x(float *kernel) {
    float sobel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    memcpy(kernel, sobel, 9 * sizeof(float));
}

void create_sobel_y(float *kernel) {
    float sobel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    memcpy(kernel, sobel, 9 * sizeof(float));
}

void create_gaussian_blur(float *kernel, int N) {
    // 3x3 Gaussian blur
    if (N == 3) {
        float gauss[9] = {1/16.0f, 2/16.0f, 1/16.0f,
                          2/16.0f, 4/16.0f, 2/16.0f,
                          1/16.0f, 2/16.0f, 1/16.0f};
        memcpy(kernel, gauss, 9 * sizeof(float));
    }
    // 5x5 Gaussian blur
    else if (N == 5) {
        float gauss[25] = {1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f,
                           4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
                           7/273.0f, 26/273.0f, 41/273.0f, 26/273.0f, 7/273.0f,
                           4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
                           1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f};
        memcpy(kernel, gauss, 25 * sizeof(float));
    }
}

void create_sharpen(float *kernel) {
    float sharp[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    memcpy(kernel, sharp, 9 * sizeof(float));
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <M> <N> <filter_type>\n", argv[0]);
        printf("filter_type: 0=Sobel-X, 1=Sobel-Y, 2=Gaussian, 3=Sharpen\n");
        return 1;
    }
    
    int M = atoi(argv[1]);  // Image size (M x M)
    int N = atoi(argv[2]);  // Kernel size (N x N)
    int filter_type = atoi(argv[3]);
    int channels = 1;  // Grayscale
    
    // Allocate memory
    size_t img_size = M * M * channels * sizeof(unsigned char);
    size_t kernel_size = N * N * sizeof(float);
    
    unsigned char *image = (unsigned char *)malloc(img_size);
    unsigned char *output = (unsigned char *)malloc(img_size);
    float *kernel = (float *)malloc(kernel_size);
    
    // Initialize test image (gradient pattern)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            image[i * M + j] = (unsigned char)((i + j) % 256);
        }
    }
    
    // Create filter
    switch(filter_type) {
        case 0: create_sobel_x(kernel); break;
        case 1: create_sobel_y(kernel); break;
        case 2: create_gaussian_blur(kernel, N); break;
        case 3: create_sharpen(kernel); break;
        default: create_gaussian_blur(kernel, N);
    }
    
    // Perform convolution and measure time
    clock_t start = clock();
    convolution2D_CPU(image, output, kernel, M, N, channels);
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU Convolution (M=%d, N=%d): %f seconds\n", M, N, elapsed);
    
    // Cleanup
    free(image);
    free(output);
    free(kernel);
    
    return 0;
}