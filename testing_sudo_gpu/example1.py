import datetime
#import nvidia_smi
import os
import time
import argparse

from pynvml import *
from subprocess import call

nvmlInit()
deviceCount = nvmlDeviceGetCount()
for i in range(0, deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    pciInfo = nvmlDeviceGetPciInfo(handle)
    nvmlDeviceSetApplicationsClocks(handle,2500,745)
