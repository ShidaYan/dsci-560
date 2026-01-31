#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16
#define MAX_KERNEL_SIZE 32

// ========== Matrix Multiplication (from before) ==========
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0.0;
    
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;
        
        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        
        __syncthreads();
    }
    
    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

extern "C" void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, 
                 (N + TILE_WIDTH - 1) / TILE_WIDTH);
    
    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// ========== Image Convolution ==========
__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__global__ void convolution2D_GPU(unsigned char *image, unsigned char *output,
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

extern "C" void gpu_convolution(unsigned char *h_image, unsigned char *h_output,
                                 float *h_kernel, int M, int N, int channels) {
    size_t img_size = M * M * channels * sizeof(unsigned char);
    size_t kernel_size = N * N * sizeof(float);
    
    unsigned char *d_image, *d_output;
    
    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_output, img_size);
    
    cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_size);
    
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((M + 15) / 16, (M + 15) / 16, channels);
    
    convolution2D_GPU<<<gridSize, blockSize>>>(d_image, d_output, M, N, channels);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_image);
    cudaFree(d_output);
}