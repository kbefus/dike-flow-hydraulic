# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 07:53:34 2021

Purpose: Test model parameters with high rnp from Sobol tests with 2-yr 
         observed time series and 1800 s (30 min) time step used in forecast models.
         Used to create Figure 4b and discussed in Section 4.1.

@author: Alexander Kurnizki & Kevin Befus
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time, warnings


import matplotlib.pyplot as plt

from dike_lib import prepare_water_elev, diked_waterlevel_solver, nse, rnp
#%% Define model options
time_delta = 30*60 # s/min * min = timestep in seconds
plot_outputs = True
save_out = False
save_json = False
# Background inflow/gains (+) or outflow/losses (-) to the diked waterbody
# For variable background discharge to/from the waterbody, a similar variable
# could be defined that has len(Q)=len(marineside_array), and the diked_waterlevel_solver
# fuction could be changed to input this array and access each entry in the time loop.
# Q_constant = 0.04 # m**3/s

#%% Load input files
herring_dir = r'path/to/dir'
data_dir = os.path.join(herring_dir,'data','water_levels')
paper_dir = os.path.join(herring_dir,'papers','kurnizki_hr')
out_dir = os.path.join(paper_dir,'analysis')
water_level_fname = os.path.join(data_dir,'USGS_CHEQNeck_200227to220227.csv')#'hr_obs_usgs_hourly_170927to191113.csv')
mar_col = 'waterlevel_mnavd88'
diked_col = 'waterlevel_mnavd88_diked'
time_col = 'datetime_utc'

wl_df = prepare_water_elev(water_level_fname,dt=time_delta,time_col=time_col)


marineside_array = wl_df[mar_col].values
# If observation data for the diked waterbody are available, the initial water level
# can be set to be the first observed water level in the time series
h_init = 0.0# 0 or wl_df[diked_col][0] # initial value of water level in Herring River, m


# Load Stage-Storage Curve for Herring River Estuary from csv
# calculated from DEM with dike_lib.stage_storage_dem function
stage_storage_df = pd.read_csv(os.path.join(paper_dir,'data','HR_stagestorageHRE_toHTR.csv'))
dem_source = 'whgusgs' # Woods Hole Group mapped bathymetry with USGS NED 1 m data filling
stage_storage_dict = {'stage':stage_storage_df['stage_m'].values,      # m
                      'storage':stage_storage_df['{}_storage_m3'.format(dem_source)].values} # m**3

#%% Define hydraulic parameters for the control structures

n_flap = 2
n_sluice=1
save_col = 'diked_model_m'
# Can use sobol outputs
s_fname=os.path.join(out_dir,'sensitivity',
                     'HRE_sensitivity_nsamps38912_ndata 4320_step 600s_220308.csv')
s_df = pd.read_csv(s_fname)
high_e_df = s_df.loc[s_df['rnp']>.90].sort_values(['rnp'])
#%%
store_nse = []

for ihigh in np.arange(high_e_df.shape[0])[:1]:
    wl_df[save_col] = np.nan
    val_df = high_e_df.iloc[ihigh]
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


    # -- Running model with measured data
    
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
    # Measured Plots
    if plot_outputs:
        fig,ax = plt.subplots()
        wl_df.plot(time_col,mar_col,ax=ax)
        wl_df.plot(time_col,diked_col, ax=ax)
        wl_df.plot(time_col,save_col,ax=ax) # modeled
        
        # Modeled        
        # ax.plot(wl_df[time_col],landside_water_elev)
        ax.set_ylabel('Elevation [m NAVD88]')
    
    # Calculate Nash-Sutcliffe Efficiency (NSE)
    skip_spinup = True # if true, don't use ndays worth of beginning to calc nse
    ndays = 2
    skipnsteps = int((ndays*86400)/time_delta) # in seconds
    nseK = nse(landside_water_elev[skipnsteps:].copy(),
                        wl_df[diked_col].values[skipnsteps:].copy(),
                       K_version=True)
    nsev = nse(landside_water_elev[skipnsteps:].copy(),
                        wl_df[diked_col].values[skipnsteps:].copy(),
                       K_version=False)
    rnpv = rnp(landside_water_elev[skipnsteps:].copy(),
                        wl_df[diked_col].values[skipnsteps:].copy())
    kge = rnp(landside_water_elev[skipnsteps:].copy(),
                        wl_df[diked_col].values[skipnsteps:].copy(),parametric=True)
    

    print('RNP = {0:3.3f}'.format(nsev)) # Nash-Sutcliffe Efficiency closer to 1 is better. > 0.75 is a very good fit, > 0.64 is a good fit
    wl_df['residual_m'] = np.nan
    wl_df['residual_m'].iloc[skipnsteps:] = wl_df[diked_col].values[skipnsteps:] - landside_water_elev[skipnsteps:]
    print('Max over/under forecast = {0:3.2f} m and {1:3.2f}'.format(np.nanmax(wl_df['residual_m']),np.nanmin(wl_df['residual_m'])))
    
    #Plot water level trajectories
    if plot_outputs:
        fig2,ax2 = plt.subplots()
        wl_df.plot(mar_col,diked_col,linestyle='none',marker='o',ax=ax2,markersize=2,markerfacecolor="None")
        uniq_inds = dike_lib.unique_rows(np.round(np.column_stack([landside_water_elev,
                                                                   wl_df[mar_col]]),
                                                  decimals=2),sort=False)
        ax2.plot(wl_df[mar_col].iloc[uniq_inds],landside_water_elev[uniq_inds],linestyle='none',marker='o',label=save_col,markersize=2,markerfacecolor="None")
        ax2.set_ylabel('Diked water level [m NAVD88]')
        ax2.set_xlabel('Bay water level [m NAVD88]')
        ax2.legend()
        
    # Plot residuals
    if plot_outputs:
        fig3,ax3 = plt.subplots(2,1)
        
        # Cross plot
        dz =0.01
        xrange = np.arange(-1.2,0.5+dz,dz)
        H,xedges,yedges = np.histogram2d(wl_df[diked_col].values[skipnsteps:],
                                         landside_water_elev[skipnsteps:],
                                         bins=(xrange,xrange))
        H=H.T
        H[H==0] = np.nan
        
        X,Y = np.meshgrid(xrange,xrange)
        p1 = ax3[0].pcolormesh(X,Y,np.log10(H),vmax=2)
        ax3[0].plot(xrange,xrange,'k--',lw=2)
        # ax3[0].plot(wl_df[diked_col].values[skipnsteps:],out_df[save_col].values[skipnsteps:],'bo',markerfacecolor='none',markersize=2)
        ax3[0].set_xlabel('Diked water level observations [m NAVD88]')
        ax3[0].set_ylabel('Modeled diked water level [m NAVD88]')
        ax3[0].set_aspect('equal')
        cbar= plt.colorbar(p1,ax=ax3[0],extend='max')
        
        # Residual plot
        xrange = np.arange(-1.2,3.+dz,dz)
        yrange = np.arange(-1.3,1.+dz,dz)
        H,xedges,yedges = np.histogram2d(wl_df[mar_col].values[skipnsteps:],
                                         wl_df['residual_m'].values[skipnsteps:],
                                         bins=(xrange,yrange))
        H=H.T
        H[H==0] = np.nan
        
        X,Y = np.meshgrid(xrange,yrange)
        p2 = ax3[1].pcolormesh(X,Y,np.log10(H),vmax=2)
        
        ax3[1].plot([-1.2,3],[0,0],'k--',lw=2)
        # ax3[1].plot(wl_df[mar_col].values[skipnsteps:],wl_df['residual_m'],'bo',markerfacecolor='none',markersize=2)
        ax3[1].set_xlabel('Marine water level observations [m NAVD88]')
        ax3[1].set_ylabel('Model residual (observation-modeled water level) [m]')
        cbar= plt.colorbar(p2,ax=ax3[1],extend='max')
        ax3[1].set_aspect('equal')
    
    if save_out:
        out_fname = os.path.join(out_dir,'sensitivity','sobol_runs','hr_sobolrun_{0:1.2f}_dt{1:d}s_220426.csv'.format(val_df.name,time_delta))
        wl_df.to_csv(out_fname)
    
    store_nse.append([val_df.name,val_df['rnp'],nsev,nseK,rnpv,kge,*wl_df['residual_m'].describe().values[1:]])
    
if save_out:
    out_nse_info = os.path.join(out_dir,'sensitivity','sobol_runs','rnp_summary_higheff_dt{0:d}_220426.csv'.format(time_delta))
    nse_cols = ['sobol_run','rnp_orig','nse','erel','rnp','kge',*wl_df['residual_m'].describe().index.values[1:]]
    nse_df = pd.DataFrame(store_nse,columns=nse_cols)
    nse_df.to_csv(out_nse_info,index=False)
    
    
    
    