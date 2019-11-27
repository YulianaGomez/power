import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

AL_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/AL_val_MAPE_count_summary')
C_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/C_val_MAPE_count_summary')
Random_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/Random_val_MAPE_count_summary')
OldNew_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/OldNew_val_MAPE_count_summary')
RF_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/RF_val_MAPE_count_summary')
RFAL_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/RFAL_val_MAPE_count_summary')
#master_df = pd.concat([AL_df,C_df,Random_df,OldNew_df,RF_df,RFAL_df])
master_df = pd.concat([OldNew_df, RF_df, RFAL_df, C_df, Random_df, AL_df])
groups = ['Selection', 'Percent', 'Application', 'MAPE', 'MAPE-std', 'count']
##application_groups = master_df.groupby('Application')
#print(application_groups)

# Setting the positions and width for the bars

width = 0.15 #.1

# Plotting the bars
fig, ax = plt.subplots(figsize=(7,6))
master_df_20 = master_df[master_df['Percent']==20].drop(columns=['Percent','Unnamed: 0'])
# Create a bar with pre_score data,
# in position pos,

df_plot = pd.DataFrame()
df_plot_std = pd.DataFrame()
#import pdb; pdb.set_trace()


for name, group in master_df_20.groupby('Selection'):

    df_plot[name] = pd.Series(group["MAPE"].values, index=group["Application"].values)
    df_plot_std[name] = pd.Series(group["MAPE-std"].values, index=group["Application"].values)
    #import pdb; pdb.set_trace()


df_plot.index.name = "Application"
print(df_plot.reset_index())
print(df_plot_std.reset_index())

df_plot = df_plot.reset_index()
pos = np.arange(len(df_plot['AL']))

r0 = pos
r1 = [p + width for p in r0]
r2 = [p + width for p in r1]
r3 = [p + width for p in r2]
r4 = [p + width for p in r3]
r5 = [p + width for p in r4]

#import pdb; pdb.set_trace()
#[p + width*2 for p in pos]
on = plt.bar(r0,
        #using df['post_score'] data,
        df_plot['OldNew'],
        # of width
        width,
        yerr = df_plot_std['OldNew'],
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#FFC222',
        # with label the third value in first_name
        label=df_plot['Application'][0])
# Create a bar with post_score data,
# in position pos + some width buffer,

rf = plt.bar(r1,
        #using df['post_score'] data,
        df_plot['RF'],
        # of width
        width,
        yerr = df_plot_std['RF'],
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='green',
        # with label the third value in first_name
        label=df_plot['Application'][1])

rfal = plt.bar(r2,
        #using df['post_score'] data,
        df_plot['RFAL'],
        # of width
        width,
        yerr = df_plot_std['RFAL'],
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='brown',
        # with label the third value in first_name
        label=df_plot['Application'][2])

# Create a bar with mid_score data,
# in position pos + some width buffer,
c = plt.bar(r3,
        #using df['mid_score'] data,
        df_plot['C'],
        # of width
        width,
        yerr = df_plot_std['C'],
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#F78F1E',
        # with label the second value in first_name
        label=df_plot['Application'][3])

# Create a bar with post_score data,
# in position pos + some width buffer,
ran = plt.bar(r4,
        #using df['post_score'] data,
        df_plot['Random'],
        # of width
        width,
        yerr = df_plot_std['Random'],
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='lightskyblue',
        # with label the third value in first_name
        label=df_plot['Application'][4])

al = plt.bar(r5,
        #using df['pre_score'] data,
        df_plot['AL'],
        # of width
        width,
        yerr = df_plot_std['AL'],
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#EE3224',
        # with label the first value in first_name
        label=df_plot['Application'][5])


# Create a bar with post_score data,
# in position pos + some width buffer,






def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#Labeling MAPE values
"""autolabel(on)
autolabel(rf)
autolabel(rfal)
autolabel(c)
autolabel(ran)
autolabel(al)"""


# Set the y axis label
ax.set_ylabel('MAPE')
# Set the chart's title
ax.set_title('MAPE for Random, AL, Conventional, Random Forest, and Old vs New IPC (20 Percent)')
# Set the position of the x ticks
ax.set_xticks([p + 2.5*width for p in range(len(df_plot['OldNew']))])
# Set the labels for the x ticks
ax.set_xticklabels(df_plot['Application'])
# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*6)
plt.ylim([0, max( df_plot['OldNew'] + df_plot['RF'] + df_plot['RFAL'] + df_plot['C'] + df_plot['Random'] + df_plot['AL'] )])
ax.set_ylim([0,225])
# Adding the legend and showing the plot
plt.legend(['Old to New', 'Random Forest', 'Random Forest + AL', 'Conventional DL', 'DL + Random', 'DL + AL'], loc='upper right')
plt.grid()
plt.show()
