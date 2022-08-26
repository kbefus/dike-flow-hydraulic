# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:39:48 2022

Purpose: Construct stage-storage relationship from shapefile and DEM.

@author: kbefus
"""

import sys, os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# Load dike_lib library
from dike_lib import load_dem, stage_storage_dem
#%%
# Dictionary for picking different features within the shp to test effect of stage-storage domain
poly_dict = {0:'HRE_toHTR',
             1:'HRE_DH',
             2:'HRE_toHTR_MC'}

polytouse = 0 # Ulimately chose this feature, shown in Figure 1
# Load shapefile
herring_dir = r'path/to/dir'
shp_dir = os.path.join(herring_dir,'papers','kurnizki_hr','data')
area_df = gpd.read_file(os.path.join(shp_dir,'HRE_lower_stagestorage.shp'))


area_poly = [area_df['geometry'].values[polytouse]] # extract only feature polygon # 3 scales

# Load DEMs - note shapefile and DEMs must all be in same coordinate system
dem_dir = r'F:\herring\data\dem' 
demfname_ned = os.path.join(dem_dir,'USGS_NED_Chequesset_one_meter_Combined.tif') # retrieved files from https://viewer.nationalmap.gov/basic/
demfname_lidar = os.path.join(dem_dir,'hr_2011las2rast_clip.tif')
demfname_whg = os.path.join(dem_dir,'whg_bathy_m_NAVD88_UTM.tif')
demfname_whgusgs = os.path.join(dem_dir,'whg_bathy_usgsfill_m_NAVD88_UTM.tif')
demfname_whglidar = os.path.join(dem_dir,'whg_bathy_lidarfill_m_NAVD88_UTM.tif')

dem_dict = {'ned':{'fname':demfname_ned},
            'lidar':{'fname':demfname_lidar},
            'whgbathy':{'fname':demfname_whg},
            'whgusgs':{'fname':demfname_whgusgs},
            'whglidar':{'fname':demfname_whglidar}}
#%%
# Construct stage array
min_stage = -2.5 # meters NAVD88, less than minimum value in DEMs inside the dike
max_stage = 3.6 # meters NAVD88, elevation of top of dike
nstages = 100 # number of stage increments
stage_range = [min_stage,max_stage]


for ikey in dem_dict: # Loop through DEMs
    # Load DEM data from area overlapping the polygon
    temp_dem,temp_profile = load_dem(dem_dict[ikey]['fname'],area_poly)
    
    # Calculate stage-storage
    ss_dict = stage_storage_dem(dem_array=temp_dem,
                                         dem_profile=temp_profile,
                                         stage_range=stage_range,
                                         nstages=nstages)
    
    # Store data in dictionary
    dem_dict[ikey].update({'dem':temp_dem,'profile':temp_profile})
    dem_dict[ikey].update(ss_dict) # stage, area, and storage volume

#%% Save to csv
ss_df = None
ss_keys = ['stage_m','storage_m3','area_m2']
for ikey in dem_dict:
    ss_dict_temp = {}
    for ss_key in ss_keys:
        if 'stage' in ss_key:
            new_key = ss_key
        else:
            new_key = '{}_{}'.format(ikey,ss_key)
        ss_dict_temp.update({new_key:dem_dict[ikey][ss_key.split('_')[0]]})
        temp_df = pd.DataFrame.from_dict(ss_dict_temp)
        
    if ss_df is None:
        ss_df = temp_df.copy()
    else:
        ss_df = pd.concat([ss_df,temp_df[temp_df.columns[1:]]],axis=1) # append columns

out_ss = os.path.join(shp_dir,'HR_stagestorage{}.csv'.format(poly_dict[polytouse]))
ss_df.to_csv(out_ss,index=False)
#%%
plot_bool = True

if plot_bool:
    fig,ax = plt.subplots()
    
    for ikey in dem_dict:
        ax.plot(dem_dict[ikey]['stage'],dem_dict[ikey]['storage'],'o-',label=ikey)
    
    ax.legend()
