# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:21:23 2021

Purpose: Run sobol sensitivity test dike flow models. Section 4.1 and Figure 4.

@author: kbefus
"""
import os
import sys
import numpy as np
import pandas as pd
import time

from SALib.sample import saltelli
from SALib.analyze import sobol,hdmr

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
mpl.rc('xtick', labelsize=12)     
mpl.rc('ytick', labelsize=12)
mpl.rcParams['pdf.fonttype'] = 42

import matplotlib.pyplot as plt

from dike_lib import prepare_water_elev,diked_waterlevel_solver, nse, rnp
#%% Define model options
time_delta = 60*10 # timestep in seconds
n_flap = 2. # number of flap gates of same size/properties in dike
n_sluice = 1. # number of sluices of same size/properties in dike

#%% Load input files
herring_dir = r'path/to/dir'
data_dir = os.path.join(herring_dir,'data','water_levels')
ddir2 = os.path.join(herring_dir,'papers','kurnizki_hr','data')
out_dir = os.path.join(herring_dir,'papers','kurnizki_hr','analysis','sensitivity')
water_level_fname = os.path.join(data_dir,'USGS_CHEQNeck_200227to220227.csv')#'hr_obs_usgs_hourly_170927to191113.csv')
mar_col = 'waterlevel_mnavd88'
diked_col = 'waterlevel_mnavd88_diked'
time_col = 'datetime_utc'

wl_df = prepare_water_elev(water_level_fname,dt=time_delta,time_col=time_col)

maxnpts = int(30 * 24 * 60 * 60 / time_delta)# 1440 at 30 min interval for 30 days, 86400 s/day *30 days/ 60 s/min /30 min dt
marineside_array = wl_df[mar_col].values[:maxnpts]
# If observation data for the diked waterbody are available, the initial water level
# can be set to be the first observed water level in the time series
h_init = 0.0# [-1,0,1] or wl_df[diked_col][0] # initial value of water level in Herring River, m


# Load Stage-Storage Curve for Herring River Estuary - calculated from DEM
stage_storage_df = pd.read_csv(os.path.join(ddir2,'HR_stagestorageHRE_toHTR.csv'))
dem_source = 'whgusgs' # Woods Hole Group mapped bathymetry with USGS NED 10 m data filling, bilinear interpolation to 1 m resolution
stage_storage_dict = {'stage':stage_storage_df['stage_m'].values,      # m
                      'storage':stage_storage_df['{}_storage_m3'.format(dem_source)].values} # m**3

#%%
names = ['ebb_free','ebb_trans','ebb_suborf','ebb_subwr',
        'ebb_supwr','ebb_fl_sub','ebb_fl_sup','flood_free',
        'flood_trans','flood_subor','flood_subwr','flood_supwr',
        'flood_max','flood_param','q_in',
        'h_init','flap_p','flap_hl']
nnames = len(names)
prob_dict = {'num_vars':nnames,
             'names':names,
             'bounds':[[0.01,2]]*nnames,
             }

nsamps=2**10
printnsamps = 100
param_combos = saltelli.sample(prob_dict,nsamps)
nse_list = []
for i,(eff,et,eso,ebw,epw,efb,efp,ff,ft,fbo,fbw,fpw,fmax,fparm,q_in,h_in,flp,flh) in enumerate(param_combos):
    if np.mod(i,printnsamps)==0:
        print('Sample {} of {}'.format(i+1,len(param_combos)))
        
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
                                        'flap_param': flp, # sensitive, doesn't matter if max_hloss = 0
                                        'max_hloss':flh,
                                        'ebb':{'subcritical':efb, # raising lowers mean
                                               'supercritical':efp}, # 1.2
                                        'flood':{}}},
                  'sluice0':{'type':'sluice',
                             'multiplier':n_sluice, # the number of sluices
                             'geom':{'invert_elev':-1.064, # Water control structure geometry information
                                     'open_height': 0.485, # eventually allow arrays here
                                     'open_width': 1.829,},
                             'flow_coef':{                 # Discharge coefficient (C_d) definitions
                                          'max_flood':fmax, 
                                          'flood_param': fparm,
                                          'ebb':{'freeflow':eff, # raising lowers mean, extends Q to lower wl,1.75
                                                 'transitional':et,#0.85, 0.5-0.7
                                                 'submerged_orifice':eso, # 9) usually 0.8, not sensitive
                                                 'subcritical_weir':ebw,# 7) 0.5
                                                 'supercritical_weir':epw}, # 6) , 0.75
                                          'flood':{'freeflow':ff, # sensitive, raising raises mean
                                                   'transitional':ft,
                                                   'submerged_orifice':fbo,
                                                   'subcritical_weir':fbw,
                                                   'supercritical_weir':fpw}}
                             }}
    
    # Prepare input dictionary
    solver_in_dict = {'marineside_water_elev':marineside_array, 'h_init':h_in,
                       'stage_storage_dict':stage_storage_dict,'dt':time_delta,
                       'Q_const':q_in,
                       'control_dict':control_dict,'verbose':False,'saveQ':False}

    
    # Run the water balance
    landside_water_elev,volumes = diked_waterlevel_solver(**solver_in_dict)

    # Calculate Nash-Sutcliffe Efficiency (NSE)
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
    kgev = rnp(landside_water_elev[skipnsteps:].copy(),
                        wl_df[diked_col].values[skipnsteps:].copy(),parametric=True)
    
    nse_list.append([nsev,nseK,rnpv,kgev])


#%% Run Sobol analysis
nse_list = np.array(nse_list)
nse_list2 = nse_list.copy()
nse_list2[np.isnan(nse_list2)] = -1
nse_new =[]
for idim in range(nse_list.shape[1]):
    nse_new.append(nse_list[~np.isnan(nse_list[:,idim]),idim])
Si = sobol.analyze(prob_dict,nse_list2[:,1]) # https://salib.readthedocs.io/en/latest/basics.html
total_Si, first_Si, second_Si = Si.to_df()

Si.plot() # plot sobol results

#%% Organize nse data
outdata = np.column_stack([param_combos,nse_list])
out_cols = np.concatenate([names,['nse','erel','rnp','kge']])
out_df = pd.DataFrame(outdata,columns=out_cols)

out_fname = os.path.join(out_dir,'HRE_sensitivity_nsamps{0:5.0f}_ndata{1:5.0f}_step{2:4.0f}s_220308.csv'.format(param_combos.shape[0],maxnpts,time_delta))
out_df.to_csv(out_fname)

#%%
# Don't include nse=1 (likely nan bug)
main_e_col = 'rnp'
# out_df[(out_df[main_e_col]==1)] = np.nan
best_fit = out_df.loc[(out_df[main_e_col]==out_df[main_e_col].max())]
high_nse_df = out_df.loc[out_df[main_e_col]>0.9]

# Plot resulting param values with high nse
fig,ax=plt.subplots()
high_nse_df.hist(bins=50,ax=ax)

