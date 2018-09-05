#!/bin/bash

cd /home/yzamora 
source activate gpu_testing
source set.env

cd /home/yzamora/power/nvidia_gpus/all_apps

python profile_sweep.py stream_maud.exe S
