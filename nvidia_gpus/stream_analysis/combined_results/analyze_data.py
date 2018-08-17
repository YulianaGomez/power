#importing necessary libraries
import csv
import numpy as np
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import pylab
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
    for col in range (5):
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
    #f = open("combined.n1000b512.rate","r")
    results = np.array(splitting_data(f))
    #print(results)
    all_4 = results[4:]
    #print (all_4)
    rates = 0
    time = 0
    for i in all_4:
        rates += float(i[1])
        time += float(i[2])
    average_rate = (rates/4)
    average_time = (time/4)
    #print (average_rate, average_time)
    return average_rate, average_time

"""
    From Stream results data
    obtaining average rate and time data from all rate files
    Combining data from all files into one dictionary: rate_all_data
"""
def rate_cleanup():
    file_rate = glob.glob("*.rate")
    rate_all_data = {"average rate": [ ], "average time":[ ]}
    for data in file_rate:
        f = open(data,"r")
        average_rate, average_time = rate_data(f)
        rate_all_data["average time"].append(average_time)
        rate_all_data["average rate"].append(average_rate)
    #print (rate_all_data)
    return rate_all_data

def rate_one(file):
    #file_rate = glob.glob("*.rate")
    rate_all_data = {"average rate": [ ], "average time":[ ]}
    f = open(file,"r")
    average_rate, average_time = rate_data(f)
    rate_all_data["average time"].append(average_time)
    rate_all_data["average rate"].append(average_rate)
    #print (rate_all_data)
    return rate_all_data

"""
    Combing data from all results files and calculating averages
    Data contains memory used, power draw,etc
"""
def results_cleanup(files):
    #files = glob.glob("*.results")
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
    return all_data

def results_one(files):
    #files = glob.glob("*.results")
    num = len(files)
    all_data = defaultdict(list)
    averages = {}
    f = open(files,"r")
    averages = average_results(f)
    #print("these are the averages: \n")
    #print (averages)
    for k,v in averages.items():
        all_data[k].append(v)
    return all_data

"""
   Plotting data from *all* results file
   {"Mem Total":0, "Mem Used":2, "Mem Free":4 ,"Power Drawed": 6, "Clocks": 8})
"""
def data_analysis(x,y):
    #mem used, mem total, power draw, etc
    try:
        results_all = results_cleanup()

        #average rate and time
        rate_all = rate_cleanup()
        print(rate_all)
        print (results_all)
        x = results_all[2] #mem_used
        y = results_all[6] #power_drawed
        y2 = rate_all["average rate"]
        #print (y)
        #print(y2[1:])

    except:
        pass

    plt.title("PIC(CUDA) Performance Results ")
    plt.ylabel('Power Drawed (Watts)')
    plt.xlabel('Array Size')
    print (x)
    print("this is x {0}".format(x))
    print("this is y {0}".format(y))

    plt.scatter(x,y)
    plt.show()

def spec_file():
    block_sizes = [128,256,512,1024]
    array_sizes = [10000,1000,100]
    #all_files = glob.glob("*.results")
    all_files = glob.glob("*.rate")
    ##TODO: be able to do rate and results at the same time, want to plot power against average time
    all_data = {}
    rate_all= {}
    array_range = []
    for f in all_files:
        for j in range(1000,1200,100):
            if str(j) in f:
                array_range.append(j)
                #all_data[f] = results_one(f)[3]
                rate_all[f] = rate_one(f)["average time"]

    #print(all_data)
    #data_analysis(array_range,all_data.values())
    data_analysis(array_range,rate_all.values())
"""
    memory total = 0
    memory used = 1
    memory free = 2
    power drawed = 3
    clocks = 4
"""
if __name__=="__main__":
    data_file = "gpu_clean.results"
    #f = open(data_file,"r")
    #average_results()
    #data_analysis()
    spec_file()
    #rate_data()
