import datetime
import os, errno
import subprocess
import time

options = ["--print-gpu-trace", "--system-profiling on --print-gpu-trace","--metrics flop_count_dp_fma","--metrics l2_write_throughput","--print-gpu-trace","--events warps_launched,local_load --metrics ipc","--print-gpu-summary"]
blocks = [128, 256, 512, 1024]

try:
    os.makedirs("sweep_stream_results")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


for n_size in range(100000,1000000,100000):
    for block_size in blocks:
        for i in options:
            stream_results = open("sweep_stream_results/stream" + str(count) + i[10:14] + "S" str(n_size) + "b" + str(block_size) + ".csv", 'a')
            sweep_results.flush()
            count +=1
            pargs=["nvprof %s ./stream_neddy.exe -B %s -N %s" % (i,str(block_size),str(n_size))]
            print(pargs)
            p = subprocess.Popen(pargs,stdout=stream_results,stderr=stream_results, shell=True)
