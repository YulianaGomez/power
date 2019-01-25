

import datetime
import os, errno
import subprocess
import time
import sys

#path = '/home/yzamora/power/nvidia_gpus/all_apps/v100_testing/'
path = '/gpfs/jlse-fs0/users/yzamora/v100_testing/'
executable = sys.argv[1]
try:
    os.makedirs(path + executable + "_results")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


#for n_size in range(128,4096*4,128):
for n_size in range(64,16384,32):
#for n_size in range(64,128,32):
    #for block_size in blocks:
    count = 0
    #print(os.path.isfile(path + executable +"_results/" + executable + "_N" + str(n_size) + ".csv"))
    if not os.path.isfile(path + executable +"_results/" + executable + "_N" + str(n_size) + ".csv"):
        print(path + executable +"_results/" + executable + "_N" + str(n_size) + ".csv")
        
        stream_results = open(path + executable +"_results/" + executable + "_N" + str(n_size) + ".csv",'a')
        stream_results.flush()
        if (sys.argv[2] == "S"):
            pargs=["nvprof --csv --metrics all ./%s -N %s" % (executable, str(n_size)) ]
        elif (sys.argv[2] == "G"):
            print("In Gaussian loop")
            pargs = ["nvprof --csv --metrics all ./%s -f ../gaus_data/matrix%s.txt" % (executable, str(n_size)) ]
        else:
            print("In Gemm loop")
            pargs = ["nvprof --csv --metrics all ./%s %s" % (executable, str(n_size)) ]
        print(pargs)
        count += 1
        p = subprocess.run(pargs,stdout=stream_results,stderr=stream_results, shell=True)
        time.sleep(1)
