

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

for file in os.listdir("/home/yzamora/rodinia/data/kmeans/inputGen"):
    if file.endswith(".txt"):
        print (file)
    count = 0
    if True:
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
        time.sleep(1)"""
