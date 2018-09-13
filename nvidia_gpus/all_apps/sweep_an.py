import re
import numbers
import glob
import os
import csv

# Used to create organized dictionary of STREAM metric data and timing results
#Going through all metrics to check whether significant
def all_metrics(combined_data_):
    #combined_data_ = {}

    metric_targets = []
    all_sig_metrics = []
    bench_targets = [ "gaussian" ,"gemm", "stream" ]
    for filen_ in glob.glob("/home/yzamora/power/nvidia_gpus/all_apps/stream_results/*.csv"):
    #for filen_ in glob.glob("/home/yzamora/power/nvidia_gpus/all_apps/gaussian_results/*.csv"):
        filen = os.path.basename(filen_)
        #print (filen)
        filen_split = filen.split('.')[0].split('_')
        bench_name = filen_split[0]
        #metric_name = filen_split[1]
        #print(filen_split)
        size_str = filen_split[1].split('N')[1]

        #if not (metric_name in metric_targets): continue
        if not (bench_name in bench_targets): continue

        key_root = bench_name+"_"+size_str
        #print(key_root)
        levels = ["Idle", "Low","High", "Max"]
        bw_units = ["GB", "MB", "KB" ,"0B"]
        # Now open the file and look for the data
        with open(filen_ ,'r') as file_handle:
            #print (file_handle)
            data_found = False
            ncols = 1
            fdata = csv.reader(file_handle)
            index_lookup = {}
            #print(filen_)
            for line_split in fdata:
                #print (line_split)
                lsplt = (len(line_split) > 0)

                if data_found:
                    #print("data found")
                    if lsplt and len(line_split) == ncols:
                        #percent - strip off end
                        # Get metric name here
                        #mname_index = index_lookup['Metric Name']
                        #metric_name = line_split[ mname_index ]
                        #if not (metric_name in metric_targets): continue

                        # Read in desired value for the current metric
                        target_index = index_lookup['Avg']; value = 0
                        metric_name = line_split[index_lookup['Metric Name']]
                        #print (line_split[target_index].isdecimal())
                        if line_split[target_index].isdecimal():
                            if line_split[target_index]!= '0':
                                #print(line_split[target_index])
                                all_sig_metrics.append(metric_name)
                                value = int(line_split[ target_index ])

                            # Labeled with percentage
                        elif "%" == line_split[target_index][-1]:
                            #print ("percentage loop")
                            all_sig_metrics.append(metric_name)
                            value = float(line_split[ target_index ][0:7]) / 100.0

                        # Labeled with bandwidth units
                        elif line_split[ target_index ][-4:-2] in bw_units:
                            # Just take the first
                            units = line_split[ target_index ][-4:-2]
                            all_sig_metrics.append(metric_name)
                            mfact = 1.0
                            if   units == "KB": mfact = 1024
                            elif units == "MB": mfact = 1024*1024
                            elif units == "GB": mfact = 1024*1024*1024
                            elif units == "0B":  mfact = 1
                            value = float(line_split[ target_index ][0:7]) * mfact

                        # idle, low, max
                        elif line_split[ target_index ][-1] == ")":
                            #print ("low")
                            all_sig_metrics.append(metric_name)
                            value = int(line_split[ target_index].split('(')[1].split(")")[0])

                        # otherwise, float
                        #elif not(float(line_split[ target_index ]).is_integer()):
                        else:
                            #print(line_split[ target_index ].split('(')[0])
                            #print("in float")
                            #print(line_split[ target_index ].split('(')[0])
                            value = float(line_split[ target_index ])



                        # Parse name of kernel
                        kernel_name = line_split[ index_lookup['Kernel'] ].split('(')[0]

                        # Define kernel-specific key
                        key = key_root + "_" + kernel_name

                        # Initialize dict for this key, if it is new
                        if not (key in combined_data_):
                            combined_data_ [ key ] = {}
                            combined_data_ [ key ][ 'size' ] = int( size_str )

                        # Store value for the metric being read right now
                        combined_data_ [ key ][ metric_name ] = value

                    else: data_found = False


                elif lsplt and line_split[0] == 'Device' and line_split[1] == 'Kernel':
                    # Set flag that we are at the data:
                    data_found = True
                    # Set number of columns in table:
                    ncols = len(line_split)
                    # Generate an index lookup table:
                    idx = 0
                    for term in line_split:
                        index_lookup[term] = idx
                        idx += 1
                    #print(index_lookup)
    return(combined_data_)


def stream_time():
    combined_data_ = {}
    combined_data_ = all_metrics(combined_data_)
    #kernels - STREAM_Scale, STREAM_Triad, set_array, STREAM_Scale, STREAM_Copy, STREAM_Add
    executable = "stream"
    for file in glob.glob("/home/yzamora/power/nvidia_gpus/all_apps/stream_results/*.csv"):
        #nf = "gaussian_N128.csv"
        nf = os.path.basename(file)
        filesplt = os.path.basename(nf).split(".")[0].split('N')[1]
        filename = executable + "_" + filesplt
        #print(filename)
        for key, value in combined_data_.items():
            #print(key)
            if filename in key:
                with open(file) as results:
                    for i,l in enumerate(results):
                        if i < 13:
                            if "Copy" in l and "Copy" in key:
                                #combined_data_[key]["Time(s)"] =
                                combined_data_[key]["Time(s)"] = float((l.split(":")[1]).split()[1])
                            elif "Scale" in l and "Scale" in key:
                                combined_data_[key]["Time(s)"] = float((l.split(":")[1]).split()[1])
                            elif "Add" in l and "Add" in key:
                                combined_data_[key]["Time(s)"] = float((l.split(":")[1]).split()[1])
                            elif "Triad" in l and "Triad" in key:
                                combined_data_[key]["Time(s)"] = float((l.split(":")[1]).split()[1])
        #file.close()
    return (combined_data_)

def all_st():
    combined_data_ = {}
    #global combined_data_={}
    #all_metrics()
    combined_data_ = stream_time()
    return combined_data_

if __name__=='__main__':
    #combined_data_ = {}
    #all_metrics()
    stream_time()
