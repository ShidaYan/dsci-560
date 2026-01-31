import ctypes
import numpy as np
import time
from PIL import Image

# Load shared library
lib = ctypes.cdll.LoadLibrary("./libimage.so")

# Define function signatures
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

lib.gpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

# ========== Test Filters ==========
def create_sobel_x():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

def create_sobel_y():
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

def create_gaussian_blur():
    return np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0

def create_sharpen():
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

# ========== Test Convolution ==========
def test_convolution():
    print("="*60)
    print("Testing GPU Convolution")
    print("="*60)
    
    # Load or create test image
    M = 512
    image = np.random.randint(0, 256, (M, M), dtype=np.uint8)
    output = np.zeros((M, M), dtype=np.uint8)
    
    # Test different filters
    filters = {
        "Sobel X": create_sobel_x(),
        "Sobel Y": create_sobel_y(),
        "Gaussian Blur": create_gaussian_blur(),
        "Sharpen": create_sharpen()
    }
    
    for filter_name, kernel in filters.items():
        N = kernel.shape[0]
        
        start = time.time()
        lib.gpu_convolution(
            image.ravel(),
            output.ravel(),
            kernel.ravel(),
            M, N, 1
        )
        end = time.time()
        
        print(f"{filter_name} ({N}x{N}): {(end-start)*1000:.2f} ms")
        
        # Save result
        Image.fromarray(output).save(f"output_{filter_name.replace(' ', '_')}.png")
    
    print()

# ========== Performance Comparison ==========
def performance_comparison():
    print("="*60)
    print("Performance Comparison: Different Image Sizes")
    print("="*60)
    
    sizes = [256, 512, 1024, 2048]
    kernel = create_gaussian_blur()
    N = kernel.shape[0]
    
    for M in sizes:
        image = np.random.randint(0, 256, (M, M), dtype=np.uint8)
        output = np.zeros((M, M), dtype=np.uint8)
        
        start = time.time()
        lib.gpu_convolution(
            image.ravel(),
            output.ravel(),
            kernel.ravel(),
            M, N, 1
        )
        end = time.time()
        
        print(f"M={M:4d}: {(end-start)*1000:7.2f} ms")
    
    print()

if __name__ == "__main__":
    test_convolution()
    performance_comparison()