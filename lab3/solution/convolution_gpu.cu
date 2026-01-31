#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_KERNEL_SIZE 32

// CUDA kernel for 2D convolution
__global__ void convolution2D_GPU(unsigned char *image, unsigned char *output,
                                   float *kernel, int M, int N, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    
    if (row < M && col < M && ch < channels) {
        int half_N = N / 2;
        float sum = 0.0f;
        
        // Apply convolution kernel
        for (int ki = 0; ki < N; ki++) {
            for (int kj = 0; kj < N; kj++) {
                int ii = row + ki - half_N;
                int jj = col + kj - half_N;
                
                // Zero-padding for boundaries
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
        
        output[(row * M + col) * channels + ch] = (unsigned char)val;
    }
}

// Optimized version with constant memory for kernel
__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__global__ void convolution2D_GPU_Optimized(unsigned char *image, unsigned char *output,
                                             int M, int N, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z;
    
    if (row < M && col < M && ch < channels) {
        int half_N = N / 2;
        float sum = 0.0f;
        
        for (int ki = 0; ki < N; ki++) {
            for (int kj = 0; kj < N; kj++) {
                int ii = row + ki - half_N;
                int jj = col + kj - half_N;
                
                if (ii >= 0 && ii < M && jj >= 0 && jj < M) {
                    int img_idx = (ii * M + jj) * channels + ch;
                    int kernel_idx = ki * N + kj;
                    sum += image[img_idx] * d_kernel[kernel_idx];
                }
            }
        }
        
        int val = (int)(sum + 0.5f);
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        
        output[(row * M + col) * channels + ch] = (unsigned char)val;
    }
}

// Filter creation functions
void create_sobel_x(float *kernel) {
    float sobel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    memcpy(kernel, sobel, 9 * sizeof(float));
}

void create_sobel_y(float *kernel) {
    float sobel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    memcpy(kernel, sobel, 9 * sizeof(float));
}

void create_gaussian_blur(float *kernel, int N) {
    if (N == 3) {
        float gauss[9] = {1/16.0f, 2/16.0f, 1/16.0f,
                          2/16.0f, 4/16.0f, 2/16.0f,
                          1/16.0f, 2/16.0f, 1/16.0f};
        memcpy(kernel, gauss, 9 * sizeof(float));
    }
    else if (N == 5) {
        float gauss[25] = {1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f,
                           4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
                           7/273.0f, 26/273.0f, 41/273.0f, 26/273.0f, 7/273.0f,
                           4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
                           1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f};
        memcpy(kernel, gauss, 25 * sizeof(float));
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <M> <N> <filter_type>\n", argv[0]);
        printf("filter_type: 0=Sobel-X, 1=Sobel-Y, 2=Gaussian\n");
        return 1;
    }
    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int filter_type = atoi(argv[3]);
    int channels = 1;
    
    // Allocate host memory
    size_t img_size = M * M * channels * sizeof(unsigned char);
    size_t kernel_size = N * N * sizeof(float);
    
    unsigned char *h_image = (unsigned char *)malloc(img_size);
    unsigned char *h_output = (unsigned char *)malloc(img_size);
    float *h_kernel = (float *)malloc(kernel_size);
    
    // Initialize test image
    for (int i = 0; i < M * M; i++) {
        h_image[i] = (unsigned char)(rand() % 256);
    }
    
    // Create filter
    switch(filter_type) {
        case 0: create_sobel_x(h_kernel); break;
        case 1: create_sobel_y(h_kernel); break;
        case 2: create_gaussian_blur(h_kernel, N); break;
        default: create_gaussian_blur(h_kernel, N);
    }
    
    // Allocate device memory
    unsigned char *d_image, *d_output;
    float *d_kernel_mem;
    
    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_kernel_mem, kernel_size);
    
    // Copy to device
    cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_mem, h_kernel, kernel_size, cudaMemcpyHostToDevice);
    
    // Copy kernel to constant memory for optimized version
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_size);
    
    // Configure kernel launch
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((M + 15) / 16, (M + 15) / 16, channels);
    
    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch and time the kernel
    cudaEventRecord(start);
    convolution2D_GPU_Optimized<<<gridSize, blockSize>>>(d_image, d_output, M, N, channels);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("GPU Convolution (M=%d, N=%d): %f milliseconds\n", M, N, milliseconds);
    
    // Copy result back
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernel_mem);
    free(h_image);
    free(h_output);
    free(h_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}