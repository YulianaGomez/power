

import datetime
import os, errno
import subprocess
import time
import sys

#options = ["--print-gpu-trace", "--system-profiling on --print-gpu-trace","--metrics flop_count_dp_fma","--metrics l2_write_throughput","--print-gpu-trace","--events warps_launched,local_load --metrics ipc","--print-gpu-summary"]
#options = ["", "--system-profiling on --print-gpu-trace","--metrics l2_write_throughput","--events warps_launched,local_load --metrics ipc","--print-gpu-summary"]
options = ["--metrics inst_executed","--metrics ipc","--metrics sysmem_read_throughput","--metrics sysmem_write_throughput","--metrics l2_l1_read_hit_rate" ,"--metrics l1_cache_local_hit_rate", "--metrics l1_cache_global_hit_rate"]
blocks = [128, 256, 512, 1024]
executable = sys.argv[1]
#blocks = [128]#, 256, 512, 1024]
#name = ["system","l2_through","warps","ipc","sum"]#,"flop","l2_through","trace","warps","load","summary"]
name = ["inst","ipc","read","write", "l2hit", "l1hit","globall1"]
try:
    os.makedirs(executable + "_results")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


#for n_size in range(128,4096*4,128):
for n_size in range(128,3968,128):
    #for block_size in blocks:
    count = 0
    for i in options:
        stream_results = open(executable +"_results/" + executable + "_" + name[count] + "_N" + str(n_size) + ".csv",'a')
        stream_results.flush()
        
        if (sys.argv[2] == "S"):
            pargs=["nvprof --csv %s ./%s -N %s" % (str(i),executable, str(n_size)) ]
        elif (sys.argv[2] == "G"):
            print("In Gaussian loop")
            pargs = ["nvprof --csv %s ./%s -f gaus_data/matrix%s.txt" % (str(i),executable, str(n_size)) ]
        else:
            print("In Gemm loop")
            pargs = ["nvprof --csv %s ./%s %s" % (str(i),executable, str(n_size)) ]
        print(pargs)
        count += 1
        p = subprocess.run(pargs,stdout=stream_results,stderr=stream_results, shell=True)
        time.sleep(1)
