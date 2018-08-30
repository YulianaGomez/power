import datetime
import nvidia_smi
import os, errno
import subprocess
import time



command = "./hotspot 512 2 2 data/hotspot/temp_512 data/hotspot/power_512"
#options = ["--print-gpu-trace", "--system-profiling on --print-gpu-trace","--metrics flop_count_dp_fma","--metrics l2_write_throughput","--print-gpu-trace --print-api-trace","--events warps_launched,local_load --metrics ipc","--print-gpu-summary"]
options = ["--print-gpu-trace", "--system-profiling on --print-gpu-trace","--metrics flop_count_dp_fma","--metrics l2_write_throughput","--print-gpu-trace","--events warps_launched,local_load --metrics ipc","--print-gpu-summary"]
try:
    os.makedirs("sweep_results")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

#sweep_results = open("sweep_results.metrics")
count = 0
for i in options:
    sweep_results = open("sweep_results/sweep"+ str(count)+ i[10:14]+ ".results",'a')
    sweep_results.flush()
    count += 1
    pargs = ["nvprof %s ./hotspot 512 2 2 data/hotspot/temp_512 data/hotspot/power_512" % i]
    print (pargs)
    p = subprocess.Popen(pargs, stdout=sweep_results,stderr=sweep_results,shell=True)
