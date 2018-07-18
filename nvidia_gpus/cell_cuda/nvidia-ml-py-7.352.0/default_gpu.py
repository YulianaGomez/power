import datetime
#import nvidia_smi
import os
import time
from pynvml import *
from subprocess import call

strResult = ''
strResult_clocks = ''
nvmlInit()
deviceCount = nvmlDeviceGetCount()
for i in range(0, deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    pciInfo = nvmlDeviceGetPciInfo(handle)

    #strResult += '  <gpu id="%s">\n' % pciInfo.busId
    
    #strResult += '    <product_name>' + nvmlDeviceGetName(handle) + '</product_name>\n'

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
        clock_defaults_file = open("config_results/clock_min.default", "w")
    except:
        os.mkdir("config_results")
        clock_defaults_file = open("config_results/clock_min.default", "w")

    clock_defaults_file.write(strResult_clocks)
    
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
    try:
        nvmlDeviceResetApplicationsClocks(handle)
        print("Reseted clocks to default") 
    except: 
        print("Exception raised when trying to reseting clocks to default.")


