from pynvml import *
import datetime
import nvidia_smi
import os
import subprocess
import time

#os.system("./cell 1000 1000 1000 2 500 &")
#os.system("/home/yzamora/kmeans/benchmark.sh &")
strResult = ''
results = ''
results +='Memory Total \t Memory Used \t    Memory Free \t Power Drawed \t Clocks \t \n'
nvmlInit()
deviceCount = nvmlDeviceGetCount()
start = time.time()
end_time = 30
loops = 0

blocks = [128, 256, 516, 1024]
for n_size in range(1000000,10000100,100):
    for block_size in blocks:
        print ("N size: %i, Block size: %i" %(n_size,block_size))
        #os.system("./stream_long.exe -B %i -N %i >> stream_benchmark.results" % (block_size,n_size))
        subprocess.run("./stream_bigtime.exe -B %i -N %i >> stream_all_benchmark.results" % (block_size,n_size))
        while p.pull() == none:
            if loops%10 == 0:
                print("Obtaining stat results: Loop %i"% loops)
        loops += 1
        for i in range(0, deviceCount):
            print ("in device loop")
            handle = nvmlDeviceGetHandleByIndex(i)
            strResult += "N size: %i, Block size: %i" %(n_size,block_size)
            """try:
                powMan = nvmlDeviceGetPowerManagementMode(handle)
                powManStr = 'Supported' if powMan != 0 else 'N/A'
            except NVMLError as err:
                powManStr = handleError(err)
            strResult += '      <power_management>' + powManStr + '</power_management>\n'
            """
            try:
                powDraw = (nvmlDeviceGetPowerUsage(handle) / 1000.0)
                powDrawStr = '%.2f W' % powDraw
            except NVMLError as err:
                powDrawStr = handleError(err)
            strResult += '      <power_draw>' + powDrawStr + '</power_draw>\n'
            strResult += '    <clocks>\n'
            try:
                graphics = str(nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)) + ' MHz'
            except NVMLError as err:
                graphics = handleError(err)
            strResult += '      <graphics_clock>' +graphics + '</graphics_clock>\n'
            try:
                memInfo = nvmlDeviceGetMemoryInfo(handle)
                mem_total = str(memInfo.total / 1024 / 1024) + ' MiB'
                mem_used = str(memInfo.used / 1024 / 1024) + ' MiB'
                mem_free = str(memInfo.total / 1024 / 1024 - memInfo.used / 1024 / 1024) + ' MiB'
            except NVMLError as err:
                error = handleError(err)
                mem_total = error
                mem_used = error
                mem_free = error
            
            
            strResult += '    <fb_memory_usage>\n'
            strResult += '      <total>' + mem_total + '</total>\n'
            strResult += '      <used>' + mem_used + '</used>\n'
            strResult += '      <free>' + mem_free + '</free>\n'
            strResult += '    </fb_memory_usage>\n'
            if memInfo.used > 0:
                results += mem_total + '\t ' + mem_used + '\t    ' + mem_free + '\t   ' + powDrawStr + '\t   ' + graphics +'\n'


    if time.time() > start + end_time : break    
#f = open("gpu_sample.results",'w')
#f.write(strResult)
f_clean = open("gpu_clean.results",'w')
f_clean.write(results)
