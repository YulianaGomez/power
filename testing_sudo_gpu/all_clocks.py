import datetime
#import nvidia_smi
import time
import argparse
import os, errno
import subprocess

from pynvml import *
from subprocess import call


def clock_cycle():
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    for i in range(0,deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_clocks = nvmlDeviceGetSupportedMemoryClocks(handle)
        graphic_clocks = nvmlDeviceGetSupportedGraphicsClocks(handle,mem_clocks[0])
        #Cycling through all available graphi_clocks
        for k in graphic_clocks:
            print("Changing Clock value to: %f"%k)
            pargs = ["sudo", "/usr/bin/nvidia-smi", "-ac", "715,"+str(k)]
            p = subprocess.Popen(pargs)
            time.sleep(20)
      

if __name__=="__main__":
    clock_cycle()
