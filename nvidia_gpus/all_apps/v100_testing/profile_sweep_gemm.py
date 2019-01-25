

import datetime
import os, errno
import subprocess
import time
import sys

executable = sys.argv[1]
try:
    os.makedirs(executable + "_results")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


#for n_size in range(128,4096*4,128):
for n_size in range(32,16384,32):
    #for block_size in blocks:
    count = 0
    print(os.path.isfile(executable +"_results/" + executable + "_N" + str(n_size) + ".csv"))
    if not os.path.isfile(executable +"_results/" + executable + "_N" + str(n_size) + ".csv"):
        print(executable +"_results/" + executable + "_N" + str(n_size) + ".csv")
        
        stream_results = open(executable +"_results/" + executable + "_N" + str(n_size) + ".csv",'a')
        stream_results.flush()
        if (sys.argv[2] == "S"):
            pargs=["nvprof --csv --metrics all ./%s -N %s" % (executable, str(n_size)) ]
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
