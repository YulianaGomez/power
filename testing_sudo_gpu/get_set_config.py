import datetime
#import nvidia_smi
import os
import time
import argparse

from pynvml import *
from subprocess import call

#Saving default configs using pynvml nvidia-smi wrapper
def save_configs():
    strResult = ''
    strResult_clocks = ''
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    for i in range(0, deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        pciInfo = nvmlDeviceGetPciInfo(handle)

        #Saving default configurations - CLOCKS
        strResult_clocks += '  GPU id = "%s"\n' % pciInfo.busId + "\n"
        strResult_clocks += "Device name:  " + str(nvmlDeviceGetName(handle)) + "\n"
        strResult_clocks += '    Default Applications Clock \n'
        try: 
            graphics = str(nvmlDeviceGetDefaultApplicationsClock(handle, NVML_CLOCK_GRAPHICS)) + ' MHz'
        except NVMLError as err:
            graphics = handleError(err)
        strResult_clocks += '      Default Graphics Clock: ' +graphics + '\n'
    
        try:
            mem = str(nvmlDeviceGetDefaultApplicationsClock(handle, NVML_CLOCK_MEM)) + ' MHz'
        except NVMLError as err:
            mem = handleError(err)
        strResult_clocks += '      Default Memory Clock: ' + mem + ' \n'
        strResult_clocks += '    Default Application Clock End\n'

        #Saving clock default configurations
        identity = ('---------------------\nGPU id = "%s"\n' % pciInfo.busId + 
                   "\n" + "Device name:  " + str(nvmlDeviceGetName(handle)) + 
                   "\n" + "---------------------\n")
   
        try:
            clock_defaults_file = open("config_results/clock_min.default", "a")
        except:
            os.mkdir("config_results")
            clock_defaults_file = open("config_results/clock_min.default", "a")

        clock_defaults_file.write(strResult_clocks)
        
        #Getting supported memory and graphic clocks
        try:
            mem_clocks = nvmlDeviceGetSupportedMemoryClocks(handle)
            graphic_clocks = nvmlDeviceGetSupportedGraphicsClocks(handle,mem_clocks[0])
            print(mem_clocks) 
            print(graphic_clocks)
        except:
            print("Trying to get supported clocks - not working")
        try:
            nvmlDeviceSetApplicationsClocks(handle, 2500,745)
        except:
            print("not working")
 
    #Saving nvidia standard calls
    clock_all = open("config_results/clock_allinfo.default","w")
    clock_all.write(identity)
    perfo_file = open("config_results/performance.default", "w")
    perfo_file.write(identity)
    default_config = open("config_results/config_all.default", "w")
    default_config.write(identity)

    #Saving all default configurations
    os.system("nvidia-smi -q >> config_results/config_all.default")   
    os.system("nvidia-smi -q -d PERFORMANCE >> config_results/performance.default")
    os.system("nvidia-smi -q -d CLOCK >> config_results/clock_allinfo.default")

#Reseting Clocks
def reset_clocks():
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    for i in range(0, deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        try:
            nvmlDeviceResetApplicationsClocks(handle)
            print("Reseted Device %s to default" % str(nvmlDeviceGetName(handle))) 
        except: 
            print("Exception raised when trying to reseting clocks to default.")

if __name__ == '__main__':
    print("\nSaving default configurations of current gpus.\nUse -r to reset clocks, -c to save configs (default=True), or -h for help. \n")
    print("Configurations saved in config_results directory created where script is run. \n")
    parser = argparse.ArgumentParser(description="short sample app")
    parser.add_argument("-r", action="store_true", default=False, dest="Reset_clocks", help="Reset clock to default.")
    parser.add_argument("-c", action="store_true", default=True, dest="Save_Default_Config", help="Saving default configurations.")
    results = parser.parse_args()
    if results.Reset_clocks:
        reset_clocks()
    if results.Save_Default_Config:
        save_configs()
