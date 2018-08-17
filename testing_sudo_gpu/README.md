# Saving all GPU configurations

Python 3.6.5

## How to run - saving configurations:
* python default_gpu.py
    * will save all current configurations in config_results directory. config_results directory will be created where script is run

## Reset Clocks
* python default_gpu.py -r  -  will reset GPU clocks

## Config_results:
* clock_allinfo.default - result from nvidia-smi -q -d CLOCK
    * Includes current clock configuration, max clocks, application clocks, etc
* clock_min.default 
    * results from pynvml wrapper - returns current default clock for memory and graphics
* config_all.default - result from nvidia-smi -q 
    * Displays GPU infor - includes all data list in GPU attributes of smi document
* performance.default - performance state for the GPU
