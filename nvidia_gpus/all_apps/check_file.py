import os
import glob 

files = glob.glob("gaus_data/matrix*")

for f in files:
   if os.stat(f).st_size == 0:
       print(f)


