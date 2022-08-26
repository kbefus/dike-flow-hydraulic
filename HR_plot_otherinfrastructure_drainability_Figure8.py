# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:05:02 2022

Purpose: Plot the outputs of the "alternative infrastructure" drainability scenarios, 
        which were then editted and combined in Adobe Illustrator to produce Figure 8.

@author: kbefus
"""

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt


#%% Define folder paths and load data
outputs_dir = r'path\to\dir'
annual_dir = os.path.join(outputs_dir,'annual')
out_flow_ratios = os.path.join(annual_dir,'all_flow_ratios_otherinfra_220325.csv') # file name with flow ratio data
fdr = pd.read_csv(out_flow_ratios)

#%%
plot=True
savedata = False

# Infrastructure combinations
infra_combos = [[0,1],[2,0],
                [3,1],[2,2],
                [4,1],[2,3],
                [5,1],[3,2]] # [number of flap gates,number of sluice gates]

if plot:
    val_range = np.linspace(0,1,101)
    trange = np.hstack([fdr.index.values,2100])
    X,Y = np.meshgrid(trange,val_range)
    
    slr_scens = ['GMSL+0.3m','GMSL+0.5m','GMSL+1.0m']
    slr_colors = ['#FFC107','#1E88E5','#D81B60']
    hydro_vals = ['drained']#,'impounded']
    styles = ['-','--',':']
    
    fig,ax = plt.subplots(4,2)
    ax = np.array(ax).ravel()
    
    fdr.fillna(0,inplace=True)
    
    time = fdr['year'].values
    
    for icombo,[nf,ns] in enumerate(infra_combos):
        active_infra = "s{0:d}f{1:d}".format(ns,nf)
        infra_cols = [i for i in fdr.columns.values if (active_infra in i) and (hydro_vals[0] in i)]
        
        for islr,slr_scen in enumerate(slr_scens):
            slr_cols = [i for i in infra_cols if slr_scen in i]
            
            # Plot each slr scenario
            for icol,scol in enumerate(slr_cols):
                ax[icombo].plot(time,fdr[scol],ls=styles[icol],color=slr_colors[islr])
            
        ax[icombo].set_title(active_infra)
        ax[icombo].set_xlim([time.min(),time.max()])
        ax[icombo].set_ylim([0,1])