import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

AL_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/AL_val_MAPE_summary')
C_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/C_val_MAPE_summary')
Random_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/Random_val_MAPE_summary')
OldNew_df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/Old_New_val_MAPE_summary')

master_df = pd.concat([AL_df,C_df,Random_df,OldNew_df])
groups = ['Selection', 'Percent', 'Application', 'MAPE']
##application_groups = master_df.groupby('Application')
#print(application_groups)

# Setting the positions and width for the bars

width = 0.20

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))
master_df_20 = master_df[master_df['Percent']==20].drop(columns=['Percent','Unnamed: 0'])
# Create a bar with pre_score data,
# in position pos,

df_plot = pd.DataFrame()
for name, group in master_df_20.groupby('Selection'):

    df_plot[name] = pd.Series(group["MAPE"].values, index=group["Application"].values)

    #import pdb; pdb.set_trace()

df_plot.index.name = "Application"
print(df_plot.reset_index())
df_plot = df_plot.reset_index()
pos = list(range(len(df_plot['AL'])))
#import pdb; pdb.set_trace()
plt.bar(pos,
        #using df['pre_score'] data,
        df_plot['AL'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#EE3224',
        # with label the first value in first_name
        label=df_plot['Application'][0])

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        #using df['mid_score'] data,
        df_plot['C'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#F78F1E',
        # with label the second value in first_name
        label=df_plot['Application'][1])

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos],
        #using df['post_score'] data,
        df_plot['OldNew'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#FFC222',
        # with label the third value in first_name
        label=df_plot['Application'][2])

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width*3 for p in pos],
        #using df['post_score'] data,
        df_plot['Random'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='lightskyblue',
        # with label the third value in first_name
        label=df_plot['Application'][3])

# Set the y axis label
ax.set_ylabel('MAPE')
# Set the chart's title
ax.set_title('MAPE for Random, AL, Conventional, and Old vs New IPC')
# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])
# Set the labels for the x ticks
ax.set_xticklabels(df_plot['Application'])
# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, max(df_plot['AL'] + df_plot['C'] + df_plot['OldNew']+ df_plot['Random'])])

# Adding the legend and showing the plot
plt.legend(['AL', 'C', 'OldNew','Random'], loc='upper left')
plt.grid()
plt.show()
