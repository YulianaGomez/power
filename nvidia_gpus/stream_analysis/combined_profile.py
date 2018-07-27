from pynvml import *
import datetime
import nvidia_smi
import os, errno
import subprocess
import time

#os.system("./cell 500 500 500 2 500 &")
#os.system("/home/yzamora/kmeans/benchmark.sh &")
strResult = ''
results = ''
header ='Memory Total (MiB) \t Memory Used (MiB) \t    Memory Free (MiB) \t Power Drawed (W) \t Clocks (Mhz) \t \n'
combined_results = open("combined.results",'a')
combined_results.write(results)
combined_results.flush()
nvmlInit()
deviceCount = nvmlDeviceGetCount()
start = time.time()
end_time = 30
loops = 0

try:
    os.makedirs("combined_results")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

blocks = [128, 256, 512, 1024]
#blocks = [128] #, 256, 512, 1024]
for n_size in range(1000,1200,100):
    for block_size in blocks:
        print("this is n_size", n_size)
        #results +='Memory Total \t Memory Used \t    Memory Free \t Power Drawed \t Clocks \t \n'
        combined_results = open("combined_results/combined.n" + str(n_size) + "b" + str(block_size) + ".results",'a')
        combined_rate = open("combined_results/combined.n" + str(n_size) + "b" + str(block_size) + ".rate",'a')
        combined_results.write(header)
        combined_results.flush()
        print ("N size: {0:4d}, Block size: {1:4d}".format(n_size,block_size))
        #os.system("./stream_long.exe -B %i -N %i  stream_benchmark.results" % (block_size,n_size))
        #pargs = ["./stream_bigtime.exe","-B",str(block_size),"-N",str(n_size)]
        pargs = ["./cooley.exe -B %s -N %s" % (str(block_size),str(n_size))]
        print (pargs)
        p = subprocess.Popen(pargs,stdout=combined_rate,stderr = combined_rate,shell=True)
        #p = subprocess.Popen("./stream_bigtime.exe -B %i -N %i  stream_all_benchmark.results" % (block_size,n_size))
        while p.poll() == None:
            if loops%5 == 0:
                print("Obtaining stat results: Loop %i"% loops)
            loops += 1
            for i in range(0, deviceCount):
                #print ("in device loop")
                handle = nvmlDeviceGetHandleByIndex(i)
                strResult += "N size: {0:5}, Block size: {1:5}".format(n_size,block_size)
                """try:
                    powMan = nvmlDeviceGetPowerManagementMode(handle)
                    powManStr = 'Supported' if powMan != 0 else 'N/A'
                except NVMLError as err:
                    powManStr = handleError(err)
                strResult += '      power_management' + powManStr + '/power_management\n'
                """
                try:
                    powDraw = (nvmlDeviceGetPowerUsage(handle) / 500.0)
                    powDrawStr = "{0:5} W".format(powDraw)
                except NVMLError as err:
                    powDrawStr = handleError(err)
                strResult += "\t power_draw {0:5} /power_draw\n".format(powDrawStr)
                strResult += '\t clocks\n'
                try:
                    graphics = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)
                except NVMLError as err:
                    graphics = handleError(err)
                strResult += "\t graphics_clock {0:5} /graphics_clock\n".format(graphics)
                try:
                    memInfo = nvmlDeviceGetMemoryInfo(handle)
                    mem_total = memInfo.total / 524 / 524
                    mem_used = memInfo.used / 524 / 524
                    mem_free = memInfo.total / 524 / 524 - memInfo.used / 524 / 524
                except NVMLError as err:
                    error = handleError(err)
                    mem_total = error
                    mem_used = error
                    mem_free = error


                strResult += '\t fb_memory_usage\n'
                strResult += "\t total {0:5} /total\n".format(mem_total)
                strResult += "\t used {0:5} /used\n".format(mem_used)
                strResult += "\t free {0:5} /free\n".format(mem_free)
                strResult += '\t fb_memory_usage\n'
                if memInfo.used > 0:
                    results = " {0:.5f}\t\t {1:4.5f}\t\t\t{2:.5f}\t\t{3:.5f}\t\t{4:.5f} \n".format(mem_total, mem_used, mem_free, powDraw, graphics)
                combined_results.write(results)

#if time.time() > start + end_time : break
#f = open("gpu_sample.results",'w')
#f.write(strResult)
f_clean = open("gpu_clean.results",'w')
f_clean.write(results)
