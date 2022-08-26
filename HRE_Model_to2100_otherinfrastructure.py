# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 07:53:34 2021

Purpose: Script used to run dike flow model described in Section 4.3.

@author: Alexander Kurnizki & Kevin Befus
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time, warnings


import matplotlib.pyplot as plt

from dike_lib import diked_waterlevel_solver
#%% Define model options
plot_outputs = False
save_out = True
save_json = False
# Background inflow/gains (+) or outflow/losses (-) to the diked waterbody


#%% Load input files
herring_dir = r'path/to/dir'
data_dir = os.path.join(herring_dir,'data','water_levels')
paper_dir = os.path.join(herring_dir,'papers','kurnizki_hr')
out_dir = os.path.join(paper_dir,'analysis')
water_level_fname = os.path.join(paper_dir,'data','CHEQ_TidePredictions_20200101to21000101.csv.gz')#'hr_obs_usgs_hourly_170927to191113.csv')

time_col = 'datetime_utc'
mar_cols = pd.read_csv(water_level_fname,nrows=1,index_col=0,compression='gzip').columns.values

# If observation data for the diked waterbody are available, the initial water level
# can be set to be the first observed water level in the time series
h_init = 0.0# 0 or wl_df[diked_col][0] # initial value of water level in Herring River, m


# Load Stage-Storage Curve for Herring River Estuary from csv
# calculated from DEM with dike_lib.stage_storage_dem function
stage_storage_df = pd.read_csv(os.path.join(paper_dir,'data','HR_stagestorageHRE_toHTR.csv'))
dem_source = 'whgusgs' # Woods Hole Group mapped bathymetry with USGS NED 1 m data filling
stage_storage_dict = {'stage':stage_storage_df['stage_m'].values,      # m
                  'storage':stage_storage_df['{}_storage_m3'.format(dem_source)].values} # m**3

# n_flap = 2
# n_sluice=1
save_col = 'diked_model_m'
# Can use sobol outputs
s_fname=os.path.join(out_dir,'sensitivity',
                     'HRE_sensitivity_nsamps38912_ndata 4320_step 600s_220308.csv')
s_df = pd.read_csv(s_fname)
high_e_df = s_df.loc[s_df['rnp']==s_df['rnp'].max()] # rnp>90 = 249, rnp>95 = 27
# high_e_df
rerun=False

infra_combos = [[0,1],[2,0],
                [3,1],[2,2],
                [4,1],[2,3],
                [5,1],[3,2]] # [nflap,nsluice]

#%%
for n_flap,n_sluice in infra_combos[::-1]:
    for mar_col in mar_cols[::-1]:
        print(mar_col)
        
        
        wl_df = pd.read_csv(water_level_fname,usecols=[time_col,mar_col],compression='gzip')
        wl_df[time_col] = pd.to_datetime(wl_df[time_col])
        wl_df.set_index(time_col,inplace=True)
        
        time_delta = (wl_df.index[1]-wl_df.index[0]).seconds # timestep in seconds  
        
        marineside_array = wl_df[mar_col].values
       
    
    #%% Define hydraulic parameters for the control structures
        for ihigh in np.arange(high_e_df.shape[0])[:1]:
            wl_df[save_col] = np.nan
            val_df = high_e_df.iloc[ihigh]
            out_fname = os.path.join(r'F:\herring\slr','hr_{0}_{1:1.0f}_dt{2:d}s_nsluice{3:d}_nflap{4:d}_220324.csv.gz'.format(mar_col,val_df.name,time_delta,n_sluice,n_flap))
    
            if not os.path.isfile(out_fname) or rerun:
            
                print('Active sobol run = {0}, Starting RNP = {1:3.2f}'.format(val_df.name,val_df['rnp']))
                # Inputs from Sobol test ihigh
                control_dict={'flap0':{'type':'flap',
                                    'multiplier':n_flap, # the number of flap gates, 0=None
                                    'geom':{                    # Water control structure geometry information
                                            'invert_elev':-1.064, 
                                            'init_angle': 0.0872, # radians
                                            'width_in': 1.829,
                                            'width_out': 2.057,
                                            'height': 2.317,
                                            'weightN': 2e3, # Newtons
                                            'open_el_hinge': 1.222},
                                    'flow_coef':{                    # Discharge coefficient (C_d) definitions
                                                'flap_param': val_df['flap_p'], # sensitive, doesn't matter if max_flap = 0
                                                'max_hloss':val_df['flap_hl'],
                                                'ebb':{'subcritical':val_df['ebb_fl_sub'], # raising lowers mean
                                                        'supercritical':val_df['ebb_fl_sup']},
                                                'flood':{}}},
                          'sluice0':{'type':'sluice',
                                      'multiplier':n_sluice, # the number of sluices
                                      'geom':{'invert_elev':-1.064, # Water control structure geometry information
                                              'open_height': 0.485, # eventually allow arrays here
                                              'open_width': 1.829,},
                                      'flow_coef':{                 # Discharge coefficient (C_d) definitions
                                                  'max_flood':val_df['flood_max'], 
                                                  'flood_param': val_df['flood_param'],
                                                  'ebb':{'freeflow':val_df['ebb_free'], # raising lowers mean
                                                          'transitional':val_df['ebb_trans'],
                                                          'submerged_orifice':val_df['ebb_suborf'], 
                                                          'subcritical_weir':val_df['ebb_subwr'],
                                                          'supercritical_weir':val_df['ebb_supwr']}, # sensitive
                                                  'flood':{'freeflow':val_df['flood_free'], # sensitive, raising raises mean
                                                            'transitional':val_df['flood_trans'],
                                                            'submerged_orifice':val_df['flood_subor'],
                                                            'subcritical_weir':val_df['flood_subwr'],
                                                            'supercritical_weir':val_df['flood_supwr']}}
                                      }}
                
                Q_constant = val_df['q_in'] # m**3/s
                
                # Prepare input dictionary
                solver_in_dict = {'marineside_water_elev':marineside_array.copy(), 'h_init':h_init,
                                   'stage_storage_dict':stage_storage_dict,'dt':time_delta,
                                   'Q_const':Q_constant,
                                   'control_dict':control_dict,'verbose':True,'saveQ':True}
                
                start_time = time.time()
                
                # Run the water balance
                if solver_in_dict['saveQ']:
                    landside_water_elev,volumes,Q_dict = diked_waterlevel_solver(**solver_in_dict)
                else:
                    landside_water_elev,volumes = diked_waterlevel_solver(**solver_in_dict)
                
                end_time = time.time()
                elapsed_time = end_time-start_time
                print('Model run time = {0:3.0f} s'.format(elapsed_time))
                print('=====================================')
                
                wl_df[save_col] = landside_water_elev.copy()
                wl_df['volume_m3'] = volumes.copy()
               
                if plot_outputs:
                    # Measured Plots
                    fig,ax = plt.subplots(1,2)
                    times = wl_df.index.to_pydatetime()
                    # wl_df.plot(time_col,mar_col,ax=ax[0])
                    # wl_df.plot(time_col,diked_col, ax=ax[0])
                    ax[0].plot(times,wl_df[mar_col].values)
                    # ax[0].plot(wl_df[time_col].values,wl_df[diked_col].values)
                    # Modeled
                    
                    ax[0].plot(times,landside_water_elev)
                    ax[0].set_ylabel('Elevation [m NAVD88]')
                
                #Plot water level trajectories
                if plot_outputs:
                    # wl_df.plot(mar_col,diked_col,linestyle='none',marker='.',ax=ax[1])
                    years = [i.year for i in times]
                    c1=ax[1].scatter(wl_df[mar_col],landside_water_elev,s=1,c=years,label=save_col,cmap='inferno')
                    cbar= plt.colorbar(c1,ax=ax,shrink=0.85)
                    cbar.ax.set_ylabel('Time')
                    ax[1].set_ylabel('Diked water level [m NAVD88]')
                    ax[1].set_xlabel('Bay water level [m NAVD88]')
                    ax[1].legend()
                    
                    fig,ax2 = plt.subplots()
                    ax2.plot(times,volumes,'.')
                    ax2.set_ylabel('Volume [m3]')
                    
                
                if save_out:                    
                    wl_df.to_csv(out_fname,compression='gzip')

    
    
    