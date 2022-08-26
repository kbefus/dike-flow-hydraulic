# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:12:06 2022

Purpose: Organize and prepare sea level elevations

@author: kbefus
"""

import os,sys
import pandas as pd
import numpy as np
from datetime import datetime,timedelta,timezone
import matplotlib.pyplot as plt

pytides_path = r'path\to\pytides'
sys.path.insert(1,pytides_path)

from pytides.tide import Tide as pyTide # https://github.com/sam-cox/pytides

#%% Workspace and load datasets

wdir = r'path\to\dir'

wl_col = 'waterlevel_mnavd88' # marine water levels
inland_wl_col = '{}_diked'.format(wl_col) # diked water level (i.e., inside/inland of dike)
tcol = 'datetime_utc'

maxtime = 2120

# USGS Cheq site: https://waterdata.usgs.gov/nwis/uv?site_no=011058798
# USGS 011058798 HERRING R AT CHEQUESSETT NECK RD AT WELLFLEET, MA
# Load Chequesset Dike Road 5 minute time series
cheq_fname_obs = os.path.join(wdir,'USGS_CHEQNeck_200227to220227.csv')
cheq_df = pd.read_csv(cheq_fname_obs)
cheq_df[tcol] = pd.to_datetime(cheq_df[tcol])
cheq_df.set_index(tcol,inplace=True,drop=True)


# Load Boston sea level rise forecasts, from https://geoport.usgs.esipfed.org/terriaslc/
bslr_fname = os.path.join(wdir,'boston_scenarios_usgsmapper.csv')
bslr_df = pd.read_csv(bslr_fname)
bslr_df['date'] = pd.to_datetime(bslr_df['date'],format='%Y',utc=True)
bslr_df.set_index('date',inplace=True)
bslr_df2 = bslr_df.truncate(after='{}-1-1 00:00:00+00:00'.format(maxtime))

#%% Resample to new timestep
dt = timedelta(days=5)#cheq_df.index.to_series().diff()[1]
bslr_new = bslr_df2.resample(dt).asfreq()
bslr_new = bslr_new.interpolate(method='polynomial',order=2)
#%%
# # Load Boston tide gauge hourly time series
tide_fname2 = os.path.join(wdir,'NOAA_gauge_8443970_utc_navd88_19210501to20220227.csv')
b_df = pd.read_csv(tide_fname2)
b_df[tcol] = pd.to_datetime(b_df[tcol])
b_df.set_index(tcol,inplace=True,drop=True)

b_2000_mean = b_df['{}-1-1 00:00:00+00:00'.format(1995):'{}-1-1 00:00:00+00:00'.format(2005)][wl_col].mean()
b_2021_mean = b_df['{}-1-1 00:00:00+00:00'.format(2020):'{}-1-1 00:00:00+00:00'.format(2022)][wl_col].mean()


blevels = b_df[wl_col]

b_annual_means = blevels.rolling(24*365,center=True,min_periods=1000).mean()
b_5y_means = blevels.rolling(24*365*5,center=True,min_periods=10000).mean()#.apply(lambda x: np.nanmean(x))
b_monthly_means = blevels.rolling(24*30,center=True,min_periods=100).mean()

# Add mean tide elevation to create forecasts in NAVD88
bslr_df3 = bslr_df2 + b_2000_mean


#%% Create Cheq tidal constituents model
cheq_df = cheq_df.dropna(subset=[wl_col])
ctide_model = pyTide.decompose(cheq_df[wl_col],cheq_df.index)
#%%
# Extend timeseries to 2100
start_date = datetime(2020,1,1,tzinfo=timezone.utc)
time_rangeh = (datetime(2100,1,1,tzinfo=timezone.utc)-start_date).days*24 # to hours
dt_hr = 30/60 # 30 minutes to hrs
new_time = pyTide._times(start_date,np.arange(-dt_hr,time_rangeh+dt_hr,dt_hr)) # some reason doesn't like to start at 0 for this time range
cheq_tide_2100 = ctide_model.at(new_time)

# Account for starting in 2020, instead of 2000
# For Boston, https://tidesandcurrents.noaa.gov/sltrends/sltrends_station.shtml?id=8443970
# trend from NOAA = 2.87 +/- 0.15 mm/yr
slr_rate = (b_2021_mean-b_2000_mean)/21. # 4.79 mm/yr

c_mean_2021 = cheq_df['{}-1-1 00:00:00+00:00'.format(2020):'{}-1-1 00:00:00+00:00'.format(2022)][wl_col].mean()

col1 = 'cheq_tide_noslr'
c2100_df = pd.DataFrame(cheq_tide_2100[1:,None],index=new_time[1:],columns=[col1])

# Add to slr scenarios
bslr_ctime = bslr_df2.reindex(new_time).interpolate(method='polynomial',order=2).reindex(new_time)
bslr_ctime.dropna(inplace=True)

# Add slr curve to cheq_tides, add mean 2021 level
# match_time = datetime(2020,3,1,tzinfo=timezone.utc)
# cheq_tide_match = cheq_df[cheq_df.index.to_pydatetime() == match_time][wl_col].values[0]
cslr_cols = ['cheq_tide_{}'.format('+'.join(i.split(' ')[1:4])) for i in bslr_ctime.columns.values]
# Add tides to slr curves - approximate relative response in Boston == Cheq
# Make relative to Jan 1, 2020
c2100_df[cslr_cols] = (bslr_ctime.values-bslr_ctime.values[0]) + c2100_df[col1].values[:,None]
c2100_df.index.name = tcol
out_c2100 = os.path.join(wdir,'CHEQ_TidePredictions_20200101to21000101.csv.gz')
c2100_df[cslr_cols].to_csv(out_c2100,compression='gzip')

#%%
plot=True
slr_names = ['Boston GMSL +0.3 m','Boston GMSL +0.5 m','Boston GMSL +1.0 m']
slr_colors = plt.cm.plasma(np.arange(3)/3)
if plot:
    fig,ax = plt.subplots()
    # bslr_df3.plot(ax=ax)
    
    # Plot slr curves as fills
    bslr_array = bslr_df3.values
    bslr_time = bslr_df3.index
    i=0
    for iname,name in enumerate(slr_names):
        ax.fill_between(bslr_time,bslr_array[:,i+1],bslr_array[:,i+2],alpha=0.5,
                        color=slr_colors[iname],edgecolor='none')
        ax.plot(bslr_time,bslr_array[:,i],label=name,lw=2,color=slr_colors[iname])
        i+=3
    
    
    
    b_monthly_means.plot(ax=ax,label='Monthly mean',color='lightgray')
    b_annual_means.plot(ax=ax,label='Annual mean',lw=2,color='b')
    b_5y_means.plot(ax=ax,label='5 year mean',lw=2,color='k')
    
    ax.legend()
    ax.set_xlim([datetime(1921,1,1),datetime(2100,1,1)])
    ax.set_ylim([-0.5,1.5])
    ax.grid()
    ax.set_axisbelow(True)

