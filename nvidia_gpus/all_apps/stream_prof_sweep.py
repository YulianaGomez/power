import datetime
import os, errno
import subprocess
import time

#options = ["--print-gpu-trace", "--system-profiling on --print-gpu-trace","--metrics flop_count_dp_fma","--metrics l2_write_throughput","--print-gpu-trace","--events warps_launched,local_load --metrics ipc","--print-gpu-summary"]
#options = ["", "--system-profiling on --print-gpu-trace","--metrics l2_write_throughput","--events warps_launched,local_load --metrics ipc","--print-gpu-summary"]
options = ["--metrics inst_executed","--metrics ipc","--metrics local_memory_overhead","--metrics inst_per_warp"]
blocks = [128, 256, 512, 1024]
#blocks = [128]#, 256, 512, 1024]
#name = ["system","l2_through","warps","ipc","sum"]#,"flop","l2_through","trace","warps","load","summary"]
name = ["inst","ipc","local","instwarp"]
try:
    os.makedirs("high_stream_results")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

#count = 0
#for n_size in range(100000,1000000,100000):
"""for n_size in range(100000,100001,1):
    for block_size in blocks:
        for i in options:
            stream_results = open("sweep_stream_results/stream" + str(count) + i[10:14] + "S" + str(n_size) + "b" + str(block_size) + ".csv", 'a')
            stream_results.flush()
            count +=1
            #pargs=["nvprof %s ./stream_maud.exe -B %s -N %s" % (i,str(block_size),str(n_size))]
            pargs=["nvprof ./stream_maud.exe" ]
            print(pargs)
            p = subprocess.Popen(pargs,stdout=stream_results,stderr=stream_results, shell=True)
"""
for n_size in range(100000,1100000,100000):
    for block_size in blocks:
        count = 0
        for i in options:
            stream_results = open("high_stream_results/stream_" + name[count] + "_N" + str(n_size) + "_B" + str(block_size) + ".csv",'a')
            stream_results.flush()
            pargs=["nvprof --csv %s ./maud_stream.exe -B %s -N %s" % (str(i),str(block_size),str(n_size)) ]
            print(pargs)
            count += 1
            p = subprocess.run(pargs,stdout=stream_results,stderr=stream_results, shell=True)
            time.sleep(1)
