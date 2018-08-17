#include </home/yzamora/power/nvidia_gpus/cusp/blas.h>
#include </home/yzamora/power/nvidia_gpus/cusp/array2d.h>
#include </home/yzamora/power/nvidia_gpus/cusp/print.h>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <float.h>
#include <fstream>
#include <stdlib.h>
#include <string.h>

#define NTIMES 1000

static double avgtime = 0;

__global__ void set_array(float *a,  float value, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) a[idx] = value;
}

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

//__global__ void blas_ex(float scalar, int len)
void blas_ex(float scalar, int len)
{
    //avgtime = mysecond();
    // initialize x vector
    /*cusp::array1d<float, cusp::host_memory> x(4);
    x[0] = 1;
    x[1] = 2;
    x[2] = 5000;
    x[3] = 100000;

    // initialize y vector
    cusp::array1d<float, cusp::host_memory> y(4);
    y[0] = 1;
    y[1] = 2;
    y[2] = 5000;
    y[3] = 100000;*/
    cusp::array1d<float,cusp::host_memory>x(len, scalar);
    cusp::array1d<float,cusp::host_memory>y(len, scalar);
    cusp::array1d<float,cusp::host_memory>z(len);
    cusp::blas::xmy(x,y,z);
   /*
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
    {
        // compute y = alpha * x + y
        cusp::blas::axpy(x,y,scalar);
        // print y
        //cusp::print(y);

        // allocate output vector
        //cusp::array1d<float, cusp::host_memory> z(4);
        //Element wise multiplication
        cusp::blas::xmy(x,y,z);
        
        // compute z = x .* y (element-wise multiplication)
        //cusp::blas::xmy(x,y,z);
         print z
        cusp::print(z);

        // compute the l_2 norm of z in 2 different ways
        std::cout << "|z| = " << cusp::blas::nrm2(z) << std::endl;
        std::cout << "sqrt(z'z) = " << sqrt(cusp::blas::dotc(z,z)) << std::endl;
        // compute the l_1 norm of z (manhattan distance)
        std::cout << "|z|_1 = " << cusp::blas::nrm1(z) << std::endl;
        // compute the largest component of a vector in absolute value
        std::cout << "max(|z_i|) = " << cusp::blas::nrmmax(z) << std::endl;
        //avgtime = mysecond() - avgtime;
        //std::cout << "Time to run: " << avgtime << "s" << std::endl;
       
    }*/


}

int main(void)
{
float *d_a, *d_b, *d_c;
int k;
double times[1][NTIMES];
int N = 100;
int bsize = 128;
float scalar;

/* Allocate memory on device */
cudaMalloc((void**)&d_a, sizeof(float)*N);
cudaMalloc((void**)&d_b, sizeof(float)*N);
cudaMalloc((void**)&d_c, sizeof(float)*N);

/* Compute execution configuration */
 dim3 dimBlock(bsize);
 dim3 dimGrid(N/dimBlock.x );
 if( N % dimBlock.x != 0 ) dimGrid.x+=1;

 printf(" using %d threads per block, %d blocks\n",dimBlock.x,dimGrid.x);

/* Initialize memory on the device */
 set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
 set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
 set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);


scalar = 3.0f;
/* ---- Running NTIMES for average time ----*/
for (k=0; k<NTIMES; k++)
{
    times[0][k] = mysecond();
    blas_ex(scalar, N);
    cudaThreadSynchronize();
    times[0][k] = mysecond() - times[0][k];
}

static double bytes = 3 * sizeof(float) * N;
/* --- Summary ---*/

for (k=1; k<NTIMES; k++)
{
    avgtime = avgtime + times[0][k];
}

avgtime = avgtime/(double(NTIMES-1));
printf("Rate (MB/s)   Avg time\n");

printf("%11.8f %11.8f\n", avgtime, 1.0E-06 * bytes/avgtime);
}
