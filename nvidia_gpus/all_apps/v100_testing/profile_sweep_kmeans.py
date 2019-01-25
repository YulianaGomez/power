

import datetime
import os, errno
import subprocess
import time
import sys
import glob
from os.path import basename

path = '/gpfs/jlse-fs0/users/yzamora/v100_testing/'
executable = sys.argv[1]
try:
    os.makedirs(path + executable + "_results")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
path = "/home/yzamora/rodinia/data/kmeans/inputGen/*.txt"
#for file in os.list"/home/yzamora/rodinia/data/kmeans/inputGen"):
for f in glob.glob(path):
    print (f)
    if True:
        data= f
        size = basename(f).split('_')[0]
        count = 0
        stream_results = open(path + executable +"_results/" + executable + "_N" + size + ".csv",'a')
        stream_results.flush()

        if (sys.argv[2] == "K"):
            pargs=["nvprof --csv --metrics all ./%s -i %s" % (executable, data) ]

        print(pargs)
        count += 1
        p = subprocess.run(pargs,stdout=stream_results,stderr=stream_results, shell=True)
        time.sleep(1)
