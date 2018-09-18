#Going through metrics
import numbers
import glob
import os
import csv

def single_run():
    if True:
        combined_data_ = {}

    metric_targets = []
    all_sig_metrics = []
    bench_targets = [ "gaussian" ,"gemm", "stream" ]
    #for filen_ in glob.glob("/home/yzamora/power/nvidia_gpus/all_apps/single_run_results/*.csv"):
    for filen_ in glob.glob("/home/yzamora/power/nvidia_gpus/all_apps/stream_gemm_results/*.csv"):
        filen = os.path.basename(filen_)
        #print (filen)
        filen_split = filen.split('.')[0].split('_')
        bench_name = filen_split[0]
        #metric_name = filen_split[1]
        #print(filen_split)
        #size_str = filen_split[1].split('N')[1]

        #if not (metric_name in metric_targets): continue
        #if not (bench_name in bench_targets): continue

        key_root = bench_name
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
                            #combined_data_ [ key ][ 'size' ] = int( size_str )

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
    return combined_data_

def compare_metrics(dict1,dict2,x):
    compared ={}
    #dict1 = combined_data_['hotspot_calculate_temp']
    #dict2 = combined_data_['huffman_vlc_encode_kernel_sm64huff']
    if x:
        diffkeys = [k for k in dict1 if dict1[k] != dict2[k]]
    else:
        #False - if values of same metric  are the same
        diffkeys = [k for k in dict1 if dict1[k] == dict2[k]]
    for k in diffkeys:
        compared[k] =  dict1[k], dict2[k],abs(dict1[k]-dict2[k])

    return (compared)

def compare_all(large_dict,x):
    dif = {}
    for k,v in large_dict.items():
        for k2,v2 in large_dict.items():
            if k != k2:
                dict_compared = str(k) + "." + str(k2)
                dif[dict_compared] = compare_metrics(large_dict[k],large_dict[k2],x)
    return(dif)

#acquiring list of metrics that have the highest differences in value
def top_metrics(input_dic,threshold,x):
    #use top_values only if values only are needed
    #top_values = []
    top_dic = []
    for k,v in input_dic.items():
        for k2,v2 in v.items():
            if x:
                if (v2[-1]) > threshold:
                    #print (k2, v2, k)
                    #if min(top_values) < v2[-1]:
                        #if len(top_values) == 10:
                            #top_values.remove(min(top_values))
                            #top_dic.remove(min(top_dic))
                        #top_values.append(v2[-1])
                    top_dic.append((k,v2[0],k2,v2[1],v2[-1]))
            else:
                if (v2[-1]) <= threshold:
                    top_dic.append((k,v2[0],k2,v2[1],v2[-1]))
            
    #print(sorted(top_values))
    #print(top_dic)
    if x:
        list_sorted = (sorted(top_dic, key=lambda tup:tup[4],reverse=True))
    else:
        list_sorted = (sorted(top_dic, key=lambda tup:tup[4]))

    return (list_sorted)

def metric_analysis(list_sorted,size):
    top10 = list_sorted[:size]
    for c in top10:
        print("First Kernel: %s, Metric: %s, Metric Value Dif: %5.5f, Metric Value: %5.5f" % (c[0].split('.')[0],c[2],c[4], c[1]))
        print("Second Kernel: %s, Metric: %s, Metric Value Dif: %5.5f, Metric Value: %5.5f" % (c[0].split('.')[1],c[2], c[4],c[3]))
        print("\n")

def main(size):
    combined_apps = single_run()
    diff_values = compare_all(combined_apps,True)
    same_values = compare_all(combined_apps,False)
    sorted_list = top_metrics(diff_values)
    metric_analysis(sorted_list,size)
    #return compared_metrics_dic
def sort_all(x,threshold):
    combined_apps = single_run()
    if x:
        values = compare_all(combined_apps,True)
    else:
        values = compare_all(combined_apps,True)
    #print(values)
    sorted_list = top_metrics(values,threshold,x)
    #metric_analysis(sorted_list,size)
    return sorted_list
#print(sort_all(False,20))
