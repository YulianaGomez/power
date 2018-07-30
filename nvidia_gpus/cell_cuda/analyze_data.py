#importing necessary libraries
import csv
import numpy as np
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

###---Getting data from results and rates files -----###
"""
0 - Mem total
2 - Mem used
4 - Mem Free
6 - Power Drawed
8 - Clocks
"""

"""
    Splits given output - removes spaces, replaces with commas
    Each individual word is considered a column (start at 0)
"""
def splitting_data(f):

    lines = f.readlines()
    results = []
    for x in lines:
        results.append(x.split(None))
    f.close()
    return results

"""
    Getting average of column
    {"Mem Total":0, "Mem Used":2, "Mem Free":4 ,"Power Drawed": 6, "Clocks": 8})
"""
def average_results(f):
    results = np.array(splitting_data(f))
    data_dic = {}
    #data_dic = []
    data_size = len(results)
    for col in range (0,10,2):
        total = 0
        for y in enumerate(results):
            total += float(y[1][col])
        average = total/data_size

        data_dic[col] = (average)

    #print (data_dic[2])
    return (data_dic)

"""
   Calculating average time and average rate from stream data
"""
def rate_data(f):
    #files = glob.glob("*.rate")
    #files = open("stream_one.rate","r")
    results = np.array(splitting_data(f))
    #print(results)
    all_4 = results[4:]
    rates = 0
    time = 0
    for i in all_4:
        rates += float(i[1])
        time += float(i[2])
    average_rate = (rates/4)
    average_time = (time/4)

    return average_rate, average_time


"""
   Plotting data from *all* results file
   {"Mem Total":0, "Mem Used":2, "Mem Free":4 ,"Power Drawed": 6, "Clocks": 8})
"""
def data_analysis():
    files = glob.glob("*.results")
    num = len(files)
    all_data = defaultdict(list)

    #Combining data from all files to one dictionary: all_data
    for data in files:
        onef_data = []
        averages = {}
        f = open(data,"r")
        averages = average_results(f)
        #print("these are the averages: \n")
        #print (averages)
        for k,v in averages.items():
            all_data[k].append(v)

    #obtaining average rate and time data from all rate files
    file_rate = glob.glob("*.rate")
    rate_all_data = {"average rate": [ ], "average_time":[ ]}
    for data in file_rate:
        f = open(data,"r")
        average_rate, average_time = rate_data(f)
        rate_all_data["average_time"].append(average_time)
        rate_all_data["average rate"].append(average_rate)
    print (rate_all_data)



    #print ("total averages \n")
    #print (all_data)
    x = all_data[2]
    y = all_data[6]
    plt.title("PIC(CUDA) Performance Results ")
    plt.ylabel('Memory Used (MiB)')
    plt.xlabel('Power Drawed (Watts)')
    plt.plot(x,y)
    plt.show()


if __name__=="__main__":
    data_file = "gpu_clean.results"
    #f = open(data_file,"r")
    #average_results()
    data_analysis()
    #rate_data()
