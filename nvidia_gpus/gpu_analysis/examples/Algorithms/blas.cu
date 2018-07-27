#include </home/yzamora/power/nvidia_gpus/cusp/blas.h>
#include </home/yzamora/power/nvidia_gpus/cusp/array2d.h>
#include </home/yzamora/power/nvidia_gpus/cusp/print.h>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <float.h>

#define NTIMES 1000

static double avgtime = 0;

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int blas_ex(void)
{
    //avgtime = mysecond();    
    // initialize x vector
    cusp::array1d<float, cusp::host_memory> x(4);
    x[0] = 1;
    x[1] = 2;
    x[2] = 5000;
    x[3] = 100000;

    // initialize y vector
    cusp::array1d<float, cusp::host_memory> y(4);
    y[0] = 1;
    y[1] = 2;
    y[2] = 5000;
    y[3] = 100000;

    // compute y = alpha * x + y
    cusp::blas::axpy(x,y,4);
    // print y
    cusp::print(y);

    // allocate output vector
    cusp::array1d<float, cusp::host_memory> z(4);    
    // compute z = x .* y (element-wise multiplication)
    cusp::blas::xmy(x,y,z);
    // print z
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
    return 0;
}

int main(void)
{
int k;
double times[1][NTIMES];
for (k=0; k<NTIMES; k++)
{
    times[0][k] = mysecond();
    blas_ex();
    times[0][k] = mysecond() - times[0][k];
}

/* --- Summary ---*/

for (k=1; k<NTIMES; k++)
{
    avgtime = avgtime + times[0][k];
} 

avgtime = avgtime/(double(NTIMES-1));

printf("Average time %11.8f \n", avgtime);

}

