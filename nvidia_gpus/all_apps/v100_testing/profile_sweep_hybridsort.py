

import datetime
import os, errno
import subprocess
import time
import sys
import glob
from os.path import basename

executable = sys.argv[1]
try:
    os.makedirs(executable + "_V100results")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
path = "/home/yzamora/rodinia_v100/data/hybridsort/*.txt"
#for file in os.list"/home/yzamora/rodinia/data/kmeans/inputGen"):
#for f in glob.glob(path):
for f in range(3,100,1):
    print (f)
    if True:
        data= f
        #size = basename(f).split('.')[0]
        size = f
        count = 0
        stream_results = open(executable +"_V100results/" + executable + "_N" + str(size) + ".csv",'a')
        stream_results.flush()

        if (sys.argv[2] == "Sort"):
            pargs=["nvprof --csv --metrics all ./%s r %s" % (executable, data) ]

        print(pargs)
        count += 1
        p = subprocess.run(pargs,stdout=stream_results,stderr=stream_results, shell=True)
        time.sleep(1)
