# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:26:46 2022

Purpose: Post-process models and calculate drainability used in Section 4.2. and for Figures 6 and 7.

@author: kbefus
"""
import os
import glob
import numpy as np
import pandas as pd

from datetime import timedelta

import matplotlib.pyplot as plt

#%%
herring_dir = r'path/to/dir'
paper_dir = os.path.join(herring_dir,'papers','kurnizki_hr')

water_level_fname = os.path.join(paper_dir,'data','CHEQ_TidePredictions_20200101to21000101.csv.gz')#'hr_obs_usgs_hourly_170927to191113.csv')

time_col = 'datetime_utc'
est_col = 'diked_model_m'
vol_col = 'volume_m3'
a_col = 'area_m2'
q_col = 'Qm3s'
mar_cols = pd.read_csv(water_level_fname,nrows=1,index_col=0,compression='gzip').columns.values


stage_storage_df = pd.read_csv(os.path.join(paper_dir,'data','HR_stagestorageHRE_toHTR.csv'))
dem_source = 'whgusgs' # Woods Hole Group mapped bathymetry with USGS NED 1 m data filling
stage_storage_dict = {'stage':stage_storage_df['stage_m'].values,      # m
                  'storage':stage_storage_df['{}_storage_m3'.format(dem_source)].values,
                  'area':stage_storage_df['{}_area_m2'.format(dem_source)]} # m**3

outputs_dir = r'F:\herring\slr'
out_fmt = 'hr_{0}_{1:1.0f}_dt{2:d}s_220309.csv.gz'


# Need to remove runs where initial rnp was high but 2yr rnp test was low
time_delta= 30*60 # s
rnp_info = os.path.join(paper_dir,'analysis','sensitivity','sobol_runs',
                            'rnp_summary_higheff_dt{0:d}_220426.csv'.format(time_delta))
rnp_df = pd.read_csv(rnp_info)
skip_runs = rnp_df[rnp_df['rnp']<.8]['sobol_run'].values



avg_window = 24*365 # 5 year window
new_dt = timedelta(days=365)

annual_dir = os.path.join(outputs_dir,'annual')
if not os.path.isdir(annual_dir):
    os.makedirs(annual_dir)

# Post-processing ideas
# - plots of annual/decadal volumes through time
# - plots of dVolume/dt through time - not helpful, just discharge
# - hydroperiod for invert and bottom of sluice
# - hydroperiod/flooded area through time using stage-storage area column
flow_ratio_list={}
for mar_col in mar_cols:
    print('---------------')
    print(mar_col)
    scen_files = glob.glob(os.path.join(outputs_dir,'hr_{}*.csv.gz'.format(mar_col)))
    
    for scen_file in scen_files:
        scen,sobol_run = os.path.basename(scen_file).split('_')[3:5]
        rname = '_'.join([scen,sobol_run])
        
        sobol_run=int(sobol_run)
        if sobol_run in skip_runs:
            continue
        
        print(rname)
        
        temp_df = pd.read_csv(scen_file,compression='gzip')
        temp_df.set_index(pd.to_datetime(temp_df[time_col]),inplace=True)
        temp_df[a_col] = np.interp(temp_df[est_col],stage_storage_dict['stage'],stage_storage_dict['area'])
        
        # Discharge
        temp_df[q_col] = temp_df[vol_col].diff()/((temp_df.index.values[1]-temp_df.index.values[0]).tolist()/(1e9)) # Discharge,m3/s, opposite sign as model, + into HRE, - out
        temp_df['qdir'] = 'inflow'
        temp_df.loc[temp_df[q_col]<=0,'qdir'] = 'outflow'
        
        temp_df['qgroup'] = (temp_df['qdir'] != temp_df['qdir'].shift()).cumsum()
        
        q_group_sums = temp_df.groupby('qgroup')[q_col].agg(np.sum)[1:-1] # drop incomplete tides
        # Dataset starts with an outflow to the harbor (low tide), start with first high tide/inflow
        q_tidal_net = q_group_sums.values[1:-1][::2]-np.abs(q_group_sums.values[2:][::2]) # HRE inflow- next outflow
        q_tidal_net_csum = np.cumsum(q_tidal_net)
        tide_mid_time = temp_df.groupby('qgroup')[q_col].agg(lambda x: x.index.values[int(x.index.values.shape[0]/2)])[1:-1].values
        
        # Calculate the proportion of tides within a time range (1yr?) that have net outflow - could go to zero, indicating run-away flooding
        q_df = pd.DataFrame({time_col:tide_mid_time[1:-1][::2],'qnetcsum':q_tidal_net_csum}) # time based on time of midpoint in high tide/inflow
        q_df['year'] = q_df[time_col].apply(lambda x: x.year)
        q_df['net_flow'] = 'impounded'
        q_df.loc[q_df['qnetcsum']<0,'net_flow'] = 'drained'
        flow_counts=q_df.groupby(['year','net_flow']).size() # number of tides in each year that is either able to fully drain (drained), or not (impounded)
        flow_ratio = (flow_counts/flow_counts.groupby(level=[0]).sum()).unstack()
        
        # min_df = temp_df.rolling(avg_window,axis=0,center=True,min_periods=int(avg_window/10)).min()
        # min_df = min_df.resample(new_dt).asfreq()
        mean_df = temp_df.rolling(avg_window,axis=0,center=True,min_periods=int(avg_window/10)).mean()
        mean_df = mean_df.resample(new_dt).asfreq()
        # max_df = temp_df.rolling(avg_window,axis=0,center=True,min_periods=int(avg_window/10)).max()
        # max_df = max_df.resample(new_dt).asfreq()
        
        # Percent change in rate - how quickly is the estuary hydrology changing?
        mean_df['dVol_p'] = (100*(mean_df[vol_col] - mean_df[vol_col].iloc[0])/mean_df[vol_col].iloc[0])
        mean_df['dArea_p'] = (100*(mean_df[a_col] - mean_df[a_col].iloc[0])/mean_df[a_col].iloc[0])
        # mean_df[['dVol_p','dArea_p']].plot()
        
        # Save annual rolling mean data for each model run separately
        out_mean = os.path.join(annual_dir,'{}_annualmeans.csv'.format(rname))
        if not os.path.isfile(out_mean):
            mean_df.to_csv(out_mean)
            
        # Store all flow_ratios
        if 'year' not in flow_ratio_list:
            flow_ratio_list.update({'year':flow_ratio.index.values})
        
        flow_ratio_list.update({'{}_{}'.format(rname,'drained'):flow_ratio['drained'],
                                '{}_{}'.format(rname,'impounded'):flow_ratio['impounded']})
        

out_flow_ratios = os.path.join(annual_dir,'all_flow_ratios_220426.csv')
fdr = pd.DataFrame(flow_ratio_list)
fdr.to_csv(out_flow_ratios,index=False)
                                
#%%
# This can be moved to a new script for plotting only.

val_range = np.linspace(0,1,101)
trange = np.hstack([fdr.index.values,2100])
X,Y = np.meshgrid(trange,val_range)

slr_scens = ['GMSL+0.3m','GMSL+0.5m','GMSL+1.0m']
hydro_vals = ['drained','impounded']

fig,ax = plt.subplots(len(hydro_vals),len(slr_scens))
ax = np.array(ax)

for islr,slr_scen in enumerate(slr_scens):
    slr_cols = [i for i in fdr.columns.values if slr_scen in i]
    for ih, hydro_val in enumerate(hydro_vals):
        hcols = [i for i in slr_cols if hydro_val in i]
        
        vals = fdr[hcols].values # for all scenarios
        vals[np.isnan(vals)] = 0.
        nmodels = vals.shape[1]
        tvals = np.tile(fdr.index.values[:,None],[1,nmodels]).ravel()
        vals = vals.ravel()
        
        # Drained
        H1,xedges,yedges = np.histogram2d(tvals,vals,
                                        bins=(trange,val_range))
        Hdr = H1.T
        Hdr[Hdr==0] = np.nan
        dr_cumsum = np.nancumsum(Hdr[::-1],axis=0)[::-1]/nmodels
        dr_cumsum[dr_cumsum==0] = np.nan
        
        # Cell centered
        xbins = xedges[:-1] + (xedges[1] - xedges[0]) / 2
        ybins = yedges[:-1] + (yedges[1] - yedges[0]) / 2
        
        if hydro_val == hydro_vals[0]:
            cmap = plt.cm.cividis_r
        else:
            cmap = plt.cm.cividis
        
        p1=ax[ih,islr].pcolormesh(X,Y,dr_cumsum,vmax=0.5,cmap=cmap)
        p2 = ax[ih,islr].contour(xbins,ybins,dr_cumsum,levels=[.05,.25,.5,.95],
                        colors='k')
        plt.clabel(p2,inline=False)
        # p1=ax.pcolormesh(X,Y,Hdr/nmodels,vmax=0.05)
        
        ax[ih,islr].set_title("{} {}, n={}".format(slr_scen,hydro_val,nmodels))
        if islr==(len(slr_scens)-1):
            plt.colorbar(p1,ax=ax[ih,islr],extend='max',shrink=0.85)
        
        # Save output info
        df1 = pd.DataFrame(dr_cumsum,columns=xedges[1:],index=yedges[1:])
        df1.index = np.round(df1.index ,decimals=2)
        
        save2dhist = os.path.join(annual_dir,'{}_{}_ratio_2dhist_220426.csv'.format(hydro_val,slr_scen))
        df1.to_csv(save2dhist)
        
        
        
#%%

drvals = fdr[fdr.columns.values[1:][::2]].values # for all scenarios
drvals[np.isnan(drvals)] = 0.
nmodels = drvals.shape[1]
tvals = np.tile(fdr.index.values[:,None],[1,nmodels]).ravel()
drvals = drvals.ravel()

# Drained
H1,xedges,yedges = np.histogram2d(tvals,drvals,
                                bins=(trange,val_range))
Hdr = H1.T
Hdr[Hdr==0] = np.nan
dr_cumsum = np.nancumsum(Hdr[::-1],axis=0)[::-1]/nmodels
dr_cumsum[dr_cumsum==0] = np.nan

# Cell centered
xbins = xedges[:-1] + (xedges[1] - xedges[0]) / 2
ybins = yedges[:-1] + (yedges[1] - yedges[0]) / 2

fig,ax = plt.subplots()
p1=ax.pcolormesh(X,Y,dr_cumsum,vmax=0.5)
p2 = ax.contour(xbins,ybins,dr_cumsum,levels=[.05,.25,.5,.95],
                colors='k')
plt.clabel(p2,inline=False)
# p1=ax.pcolormesh(X,Y,Hdr/nmodels,vmax=0.05)
plt.colorbar(p1,ax=ax,extend='max')



        
        
