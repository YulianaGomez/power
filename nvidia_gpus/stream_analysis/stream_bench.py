#Going through stream array and block sizes


import os
blocks = [128, 256, 516, 1024]
for n_size in range(10000000,10000100,100):
   for block_size in blocks:
       print ("N size: %i, Block size: %i" %(n_size,block_size))
       #os.system("./stream_long.exe -B %i -N %i >> stream_benchmark.results" % (block_size,n_size))
       os.system("./stream_bigtime.exe -B %i -N %i >> stream_benchmark.results" % (block_size,n_size))
