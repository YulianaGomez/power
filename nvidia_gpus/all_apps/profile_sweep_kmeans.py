

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
        size = file.split('_').[0]
        count = 0
        stream_results = open(executable +"_results/" + executable + "_N" + size + ".csv",'a')
        stream_results.flush()

        if (sys.argv[2] == "K"):
            pargs=["nvprof --csv --metrics all ./%s -i %s" % (executable, file) ]

        print(pargs)
        count += 1
        p = subprocess.run(pargs,stdout=stream_results,stderr=stream_results, shell=True)
        time.sleep(1)
