#include </home/yzamora/power/nvidia_gpus/cusp/array1d.h>
#include </home/yzamora/power/nvidia_gpus/cusp/print.h>

// include cusp blas header file
#include </home/yzamora/power/nvidia_gpus/cusp/blas/blas.h>

int main()
{ 
 // create an array
 cusp::array1d<float,cusp::host_memory> x(10);
cusp::random_array<float> rand(10);
int index = cusp::blas::amax(x);
std::cout << "Max value at pos: " << index << std::endl;
  return 0;
}

