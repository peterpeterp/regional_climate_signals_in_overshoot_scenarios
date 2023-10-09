#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[1]:


import sys,glob,datetime,os,gc,cftime
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

sys.path.append('/home/u/u290372/scripts/preprocesseing_levante/')
import _pre_oneRun as _pre_oneRun


# ACCESS-ESM1-5 CESM2-WACCM CanESM5 EC-Earth3
#        EC-Earth3-Veg-LR FGOALS-g3 GFDL-ESM4 GISS-E2-1-G
#        IPSL-CM6A-LR MIROC-ES2L MIROC6 MPI-ESM1-2-LR
#        MRI-ESM2-0 UKESM1-0-LL

# In[2]:


try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    indicator = 'tas'
    realm = 'Amon'
    esm = 'CanESM5'
    scenario = 'ssp534-over' 
    
except:
    import argparse
    parser = argparse.ArgumentParser(description='look it up')
    parser.add_argument('-i', '--indicator')  
    parser.add_argument('-s', '--scenario')  
    parser.add_argument('-m', '--esm')  
    parser.add_argument('-r', '--realm')  
    args = vars(parser.parse_args())
    print(args)
    indicator = args['indicator']   
    scenario = args['scenario']   
    esm = args['esm']   
    realm = args['realm']
print(esm, indicator)


# In[ ]:


if os.path.isdir('/work/uc1275/u290372/overshoot/gridded_period_averages/%s/' %(indicator)) == False:
    os.system('mkdir /work/uc1275/u290372/overshoot/gridded_period_averages/%s/' %(indicator))


# In[3]:


dummy = _pre_oneRun.oneRun(scenario=scenario, esm=esm, run='*', realm=realm, indicator=indicator, source='CMIP6')
dummy.get_files()
MSRs = np.unique([fl.split(realm+'_')[-1].split('_g')[0] for fl in dummy._scen_files])
SMRs = ['_'.join([msr.split('_')[1], msr.split('_')[0], msr.split('_')[2]]) for msr in MSRs]


# In[4]:


l = []
for smr in SMRs:
    scenario,esm,run = smr.split('_')
    fl = '/work/uc1275/u290372/overshoot/regional/%s/Amon/tas/global/CMIP6_Amon_tas_%s_global_annual-mean.nc' %('CMIP6', smr)
    gmt = xr.load_dataset(fl)['tas']
    gmt -= gmt.max()
    if np.all(np.isfinite(gmt.loc[2030:2080])):
        peak_year = gmt.year.values[np.nanargmax(gmt)]
        # closest years to -0.2 ,0.1, peak, -0.1, -0.2
        candidate_years = np.array([
            gmt.loc[:peak_year].year.values[np.nanargmin(np.abs(gmt.loc[:peak_year] + 0.2))],
            gmt.loc[:peak_year].year.values[np.nanargmin(np.abs(gmt.loc[:peak_year] + 0.1))],
            peak_year,
            gmt.loc[peak_year:].year.values[np.nanargmin(np.abs(gmt.loc[peak_year:] + 0.1))],
            gmt.loc[peak_year:].year.values[np.nanargmin(np.abs(gmt.loc[peak_year:] + 0.2))]
        ])   

        # check whether candidate years are close enough
        relevant_years = candidate_years.copy() + 0.0
        for i,yr,v in zip(range(5),candidate_years,[-0.2,-0.1,0,-0.1,-0.2]):
            if np.abs(gmt.loc[yr] - v) > 0.05:
                relevant_years[i] = np.nan

        # only continue if the required warming levels are there
        if len(relevant_years) > 3:
            full_grid = _pre_oneRun.oneRun(scenario=scenario,esm=esm,run=run,realm=realm,indicator=indicator, source='CMIP6')
            full_grid.get_files()
            full_grid.open_data()

            y = full_grid._data.groupby('time.year').mean('time')

            ps = xr.DataArray(dims=['SMR','period','lat','lon'], coords=dict(SMR=[smr], period=['-0.2b','-0.1b','oshoot','-0.1a','-0.2a'], lat=y.lat.values, lon=y.lon.values))
            for i,yr in enumerate(relevant_years):
                if np.isfinite(yr):
                    ps.loc[smr][i,:,:] = y.loc[yr-15:yr+15].mean('year')
            l.append(ps)
            xr.Dataset({indicator:xr.concat(l, dim='SMR')}).to_netcdf(
                '/work/uc1275/u290372/overshoot/gridded_period_averages/%s/%s_%s_31year_periods.nc' %(indicator, esm, scenario))
            


# In[ ]:




