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
nvmlInit()
deviceCount = nvmlDeviceGetCount()
start = time.time()
loops = 0

try:
    os.makedirs("combined_results_2")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

blocks = [128, 256, 512, 1024]
#blocks = [128] #, 256, 512, 1024]

for i in range(0,deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    mem_clocks = nvmlDeviceGetSupportedMemoryClocks(handle)
    graphic_clocks = nvmlDeviceGetSupportedGraphicsClocks(handle,mem_clocks[0])
    #print("Going through %f clock speeds" % len(graphic_clocks))
    #sys.exit()
    for k in graphic_clocks:

        # 1. Change Clock frequency - wait for GPU to make change
        # 2. While the program is running, collect all data
       
        print("Changing Clock value to: %f"%k)
        pargs = ["sudo", "/usr/bin/nvidia-smi", "-ac", "715,"+str(k)]
        p = subprocess.Popen(pargs)
        print("Waiting for GPU Clock Frequency Chang. \n")
        time.sleep(2)
        print("Running application and collecting data\n")
        print("\n \n \n")
        for n_size in range(1000000,10100000,1000000): #og
        #for n_size in range(10000000,10100000,1000000):
        #for n_size in range(100,100000,10000):
    	    for block_size in blocks:
                #time.sleep(5)
                print("Testing array size: ", n_size)
        	#results +='Memory Total \t Memory Used \t    Memory Free \t Power Drawed \t Clocks \t \n'
                combined_results = open("combined_results_2/combined.n" + str(n_size) + "b" + str(block_size) + "c" + str(k)+ ".results",'a')
                combined_rate = open("combined_results_2/combined.n" + str(n_size) + "b" + str(block_size) + "c" + str(k) +".rate",'a')
                combined_results.write(header)
       	        combined_results.flush()
                print ("N size: {0:4d}, Block size: {1:4d}".format(n_size,block_size))
                #os.system("./stream_long.exe -B %i -N %i  stream_benchmark.results" % (block_size,n_size))
                #pargs = ["./stream_bigtime.exe","-B",str(block_size),"-N",str(n_size)]
                #pargs = ["./cooley.exe -B %s -N %s" % (str(block_size),str(n_size))]
                #pargs = ["./stream_neddy.exe -B %s -N %s" % (str(block_size),str(n_size))]
                pargs = ["./stream_maud.exe -B %s -N %s" % (str(block_size),str(n_size))]
                print (pargs)
                p = subprocess.Popen(pargs,stdout=combined_rate,stderr = combined_rate,shell=True)
                #p = subprocess.Popen("./stream_bigtime.exe -B %i -N %i  stream_all_benchmark.results" % (block_size,n_size))
                while p.poll() == None:
                    strResult += "N size: {0:5}, Block size: {1:5}".format(n_size,block_size)
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
