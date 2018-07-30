#importing necessary libraries
import csv
import numpy as np
import os

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
def splitting_data():

    lines = f.readlines()
    results = []
    for x in lines:
        results.append(x.split(None))
    f.close()
    return results

"""
    Getting average of column
    {"Mem Total":0, "Mem Used":0, "Mem Free":0 ,"Power Drawed": 0, "Clocks": 0})
"""
def average_results():
    results = np.array(splitting_data())
    data_dic = {}
    data_size = len(results)
    for col in range (0,10,2):
        total = 0
        for y in enumerate(results):
            total += float(y[1][col])
        average = total/data_size

        data_dic[col] = average
    return (data_dic)

"""
   Plotting data from results file
"""
def data_analysis():
    


if __name__=="__main__":
    data_file = "gpu_clean.results"
    f = open(data_file,"r")
    average_results()
