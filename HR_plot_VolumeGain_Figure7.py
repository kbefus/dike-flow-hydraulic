# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 09:44:59 2022

Purpose: Produce the plots used to make Figure 7.

@author: kbefus
"""


import os
import glob
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import pandas as pd
#%% Directory and analysis settings

outputs_dir = r'path\to\dir'
annual_dir = os.path.join(outputs_dir,'annual')
out_means = glob.glob(os.path.join(annual_dir,'*_annualmeans.csv'))

time_delta= 30*60 # time step, seconds
herring_dir = r'path/to/dir' # path to uppermost directory
paper_dir = os.path.join(herring_dir,'papers','kurnizki_hr')
rnp_info = os.path.join(paper_dir,'analysis','sensitivity','sobol_runs',
                            'rnp_summary_higheff_dt{0:d}_220426.csv'.format(time_delta))
rnp_df = pd.read_csv(rnp_info)
skip_runs = rnp_df[rnp_df['rnp']<.8]['sobol_run'].values # only use sobol runs with rnp > 0.8


tcol = 'datetime_utc' # time column name
vcol = 'dVol_p' # volume column name, percent volume change from 2020
hcol = 'diked_model_m' # level column name

wlT_or_vF = True # True provides water level results; False provides volume change results
#%% Compile results

all_data = {}

for amean in out_means:
    sobol_run = int(os.path.basename(amean).split('_')[1])
    scen = "_".join(os.path.basename(amean).split('_')[:-1])
    
    if sobol_run in skip_runs:
        continue
    
    temp_df = pd.read_csv(amean)
    temp_df.set_index(pd.to_datetime(temp_df[tcol]),inplace=True)
    
    if tcol not in all_data:
        all_data.update({tcol:temp_df[tcol]})
        
        
    if wlT_or_vF:
        all_data.update({scen:temp_df[hcol]})
    else:
        all_data.update({scen:temp_df[vcol]})
    # ax.plot(temp_df[tcol],temp_df[vcol])


all_df = pd.DataFrame(all_data)
all_df.set_index(pd.to_datetime(all_df[tcol]),inplace=True)

#%% Plot results

slr_scens = ['GMSL+0.3m','GMSL+0.5m','GMSL+1.0m'] # sea level scenario names

invert_elev = -1.064 # invert elevation, m navd88

fig,ax = plt.subplots(1,len(slr_scens),sharex=True,sharey=True)
ax = np.array(ax)
for islr,slr_scen in enumerate(slr_scens):
    slr_cols = [i for i in all_df.columns.values if slr_scen in i]
    colors = []
    scen_cols = {'high':[],'mid':[],'low':[]}
    for scol in slr_cols:
        if 'High' in scol:
            colors.append('r')
            scen_cols['high'].append(scol)
        elif 'Scenario' in scol:
            colors.append('g')
            scen_cols['mid'].append(scol)
        elif 'Low' in scol:
            colors.append('b')
            scen_cols['low'].append(scol)
        
    for iscen in ['high','mid','low']:
        # Color source: https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40        
        if iscen == 'high':
            color = '#D81B60'
        elif iscen =='mid':
            color = '#1E88E5'
        else:
            color = '#FFC107'
            
        tempdf = all_df[scen_cols[iscen]].copy()
        times = tempdf.index.to_pydatetime()
        ax[islr].fill_between(times,y1=tempdf.max(axis=1).values,y2=tempdf.min(axis=1).values,
                              facecolor=color)
        

    ax[islr].set_title(slr_scen)
    ax[islr].set_xlim([times.min(),times.max()])

    if wlT_or_vF:
        ax[islr].set_ylabel('Annual mean water level [m NAVD88]')
    else:
        ax[islr].set_ylabel('Volume change [%]')


fig.autofmt_xdate(rotation=45)