#!/bin/bash

cd /home/yzamora 
source activate gpu_testing
source set.env

cd /home/yzamora/power/nvidia_gpus/all_apps/v100_testing

#python profile_sweep_hybridsort.py hybridsort Sort
#python profile_sweep_small.py gemm h
#python profile_sweep_gauss.py gemm_dgx H #running gemm routine
#python profile_sweep_gauss.py gaussian G
#python profile_sweep_kmeans.py kmeans_cuda K
python profile_sweep.py stream_dgx S
