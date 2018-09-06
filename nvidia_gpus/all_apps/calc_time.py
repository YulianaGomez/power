#Rerunning applications to get time to finish

#Make directory unless it exists already
executable = sys.argv[1]
try:
    os.makedirs(executable + "timing")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for n_size in range(128,256,128):
    results = open(executable +"_timing/" + executable + "_N" + str(n_size) + ".csv",'a')
    if (sys.argv[2]=="G"):
        print("Gaussian Loop")
        pargs = ["./%s -f gaus_data/matrix%s.txt" % (executable, str(n_size)) ]
        print(pargs)
        count += 1
        p = subprocess.run(pargs,stdout=results,stderr=results, shell=True)
