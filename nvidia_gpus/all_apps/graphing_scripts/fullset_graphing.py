import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

error_type = 'variance'
CFFULL_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/CFFULL_val_MAPE_fullset_summary')
RFFULL_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/RFFULL_val_MAPE_fullset_summary')
DHFUll_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/DHFULL_val_MAPE_newvar_summary')
OldNew_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/OldNew_val_MAPE_newvar_summary')

#master_df = pd.concat([AL_df,C_df,Random_df,OldNew_df,RF_df,RFAL_df])
master_df = pd.concat([OldNew_df,RFFULL_df,CFFULL_df,DHFUll_df])
groups = ['Selection', 'Percent', 'Application', 'MAPE']
##application_groups = master_df.groupby('Application')
#print(application_groups)

# Setting the positions and width for the bars

width = 0.10                                                                                                     #.1

# Plotting the bars
fig, ax = plt.subplots(figsize=(7,6))
master_df_20 = master_df[master_df['Percent']==70].drop(columns=['Percent','Unnamed: 0'])
# Create a bar with pre_score data,
# in position pos,

df_plot = pd.DataFrame()

#import pdb; pdb.set_trace()


for name, group in master_df_20.groupby('Selection'):

    df_plot[name] = pd.Series(group["MAPE"].values, index=group["Application"].values)

    #import pdb; pdb.set_trace()


df_plot.index.name = "Application"
print(df_plot.reset_index())
df_plot = df_plot.reset_index()
pos = np.arange(len(df_plot['CFFULL']))

r0 = pos
r1 = [p + width for p in r0]
r2 = [p + width for p in r1]
r3 = [p + width for p in r2]
r4 = [p + width for p in r3]

#import pdb; pdb.set_trace()
#[p + width*2 for p in pos]
on = plt.bar(r0,
        #using df['post_score'] data,
        df_plot['OldNew'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#FFC222',
        # with label the third value in first_name
        label=df_plot['Application'][0])

rf = plt.bar(r1,
        #using df['post_score'] data,
        df_plot['RFFULL'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='green',
        # with label the third value in first_name
        label=df_plot['Application'][1])
# Create a bar with post_score data,
# in position pos + some width buffer,

cf = plt.bar(r2,
        #using df['post_score'] data,
        df_plot['CFFULL'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#F78F1E',
        # with label the third value in first_name
        label=df_plot['Application'][2])

dh = plt.bar(r3,
        #using df['post_score'] data,
        df_plot['DHFULL'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='lightskyblue',
        # with label the third value in first_name
        label=df_plot['Application'][3])


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
autolabel(on)
autolabel(rf)
autolabel(cf)
autolabel(dh)


# Set the y axis label
ax.set_ylabel('MAPE')
ax.set_xlabel("Applications")
# Set the chart's title
ax.set_title('MAPE Models with full training data')
# Set the position of the x ticks
ax.set_xticks([p + 2.5*width for p in range(len(df_plot['DHFULL']))])
# Set the labels for the x ticks
ax.set_xticklabels(df_plot['Application'])
# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*6)
plt.ylim([0, max( df_plot['CFFULL'] + df_plot['RFFULL'] + df_plot['DHFULL'] + df_plot['OldNew'])])
ax.set_ylim([0,80])
# Adding the legend and showing the plot
plt.legend(['OldNew','RFFULL', 'CFULL', 'DHFULL'], loc='upper right')
##plt.legend(['Old to New', 'Random Forest', 'Random Forest + AL', 'Conv_DL', 'Conv_DL + AL', 'DH'], loc='upper right')

plt.grid()
plt.show()
