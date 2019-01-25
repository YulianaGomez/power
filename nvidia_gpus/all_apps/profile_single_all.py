

import datetime
import os, errno
import subprocess
import time
import sys

executable_path = sys.argv[2]
executable_name = sys.argv[1]
for n_size in range(1,2,1):
    #for block_size in blocks:
    count = 0
    if True:
        stream_results = open("/home/yzamora/power/nvidia_gpus/all_apps/single_run_results/" + executable_name + "_single" + ".csv",'a')
        stream_results.flush()
        
        if (sys.argv[3] == "A"):
            #print("Hybrid Sort")
            pargs=["nvprof --csv --metrics all ./%s" % (executable_path) ]
            #print(pargs)
            #sys.exit(0)
        elif (sys.argv[2] == "G"):
            print("In Gaussian loop")
            pargs = ["nvprof --csv --metrics all ./%s -f gaus_data/matrix%s.txt" % (executable, str(n_size)) ]
        else:
            print("In Gemm loop")
            pargs = ["nvprof --csv --metrics all ./%s %s" % (executable, str(n_size)) ]
        print(pargs)
        count += 1
        p = subprocess.run(pargs,stdout=stream_results,stderr=stream_results, shell=True)
        time.sleep(1)
