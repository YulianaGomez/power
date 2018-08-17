// Utilities and system includes
#include <assert.h>
#include </home/yzamora/power/nvidia_gpus/gpu_analysis/common/inc/helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include </soft/compilers/cuda/cuda-9.1.85/include/cuda_runtime.h>
#include </soft/compilers/cuda/cuda-9.1.85/include/cublas_v2.h>

// CUDA and CUBLAS functions
#include </home/yzamora/power/nvidia_gpus/gpu_analysis/common/inc/helper_functions.h>
#include </home/yzamora/power/nvidia_gpus/gpu_analysis/common/inc/helper_cuda.h>

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif


// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}


int main()
{
// allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);
    
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

}
