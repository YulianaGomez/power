import sys
import csv
import os, errno
import subprocess
#Rerunning applications to get time to finish

#Make directory unless it exists already
executable = sys.argv[1]
try:
    os.makedirs(executable + "_timing")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for n_size in range(128,10496,128):
    results = open(executable +"_timing/" + executable + "_N" + str(n_size) + ".csv",'a')
    if (sys.argv[2]=="G"):
        #print("Gaussian Loop")
        pargs = ["./%s -f gaus_data/matrix%s.txt" % (executable, str(n_size)) ]
    if (sys.argv[2]=="H"):
        #print("Gemm Loop")
        pargs = ["./%s %s" % (executable, str(n_size)) ]
    #print(pargs)
    p = subprocess.run(pargs,stdout=results,stderr=results, shell=True)
    """with open(executable +"_timing/" + executable + "_N" + str(n_size) + ".csv",'r') as results_:
           for l in results_:
               if "Time total" in l:
                   print(l.split("\t")[1].split()[0])"""
        
        
