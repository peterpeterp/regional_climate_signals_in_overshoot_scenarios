import sys,glob,datetime,os,gc,importlib,pickle,cftime,re
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from distinctipy import distinctipy
from matplotlib.backends.backend_pdf import PdfPages

import cartopy

import itertools
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/home/u/u290372/scripts/')
from cartopy_features import *
import _regions_ar6wg1 
sys.path.append('/home/u/u290372/scripts/preprocesseing_levante/')
import _pre_oneRun

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


if realm == 'ETCCDI':
    search_strings = ['/work/uc1275/u290372/data/etccdi_indices_*/INDICATOR_mon_ESM_SCENARIO_RUN_*.nc']
else:
    search_strings = ['/pool/data/CMIP6/data/*/*/ESM/SCENARIO/RUN/REALM/INDICATOR/*/*/*',
                      '/work/uc1275/u290372/CMIP6/data/*/*/ESM/SCENARIO/RUN/REALM/INDICATOR/*/*/*']

SMRs = []
for search_string in search_strings:
    dummy = _pre_oneRun.oneRun(scenario=scenario,esm=esm,run='*',realm=realm,indicator=indicator, search_strings=search_strings, source='CMIP6')
    if 'ETCCDI' in indicator:
        SMRs += ['_'.join([msr.split('_')[-3], msr.split('_')[-4], msr.split('_')[-2]]) for msr in dummy.get_files(search_string)]
    else:
        MSRs = np.unique([fl.split(realm+'_')[-1].split('_g')[0] for fl in dummy.get_files(search_string)])
        SMRs += ['_'.join([msr.split('_')[1], msr.split('_')[0], msr.split('_')[2]]) for msr in MSRs]   
SMRs = np.unique(SMRs)       

for smr in SMRs:
    print(smr)
    scenario,esm,run = smr.split('_')
    full_grid = _pre_oneRun.oneRun(scenario=scenario,esm=esm,run=run,realm=realm,indicator=indicator, search_strings=search_strings, source='CMIP6')
    full_grid.open_data()
    if 'ssp' in scenario:
        full_grid.open_and_add_historical_data()
    if scenario == 'ssp534-over':
        full_grid.open_and_add_ssp585_data()
    if 'ssp' in scenario:
        full_grid._data = full_grid._data.loc[:'2150']
        
    #if full_grid.check_if_sufficient_years(np.arange(2000,2100)):
    nc_mask_land = xr.open_dataset('/work/uc1275/u290372/data/masks/CMIP6_ar6wg1/ar6wg1_%s_%s_overlap.nc' %(full_grid._esm, 'land'))
    nc_mask_ocean = xr.open_dataset('/work/uc1275/u290372/data/masks/CMIP6_ar6wg1/ar6wg1_%s_%s_overlap.nc' %(full_grid._esm, 'ocean'))
    for region_name in list(_regions_ar6wg1.ar6wg1_reg_info.keys()):
        # determine whether land or ocean
        if region_name in _regions_ar6wg1.ar6wg1_d['land']+_regions_ar6wg1.ar6wg1_d['mix']:
            mask = nc_mask_land[region_name]
            mask.name = mask.name+'-land'
        elif region_name in _regions_ar6wg1.ar6wg1_d['ocean']:
            mask = nc_mask_ocean[region_name]
            mask.name = mask.name+'-ocean'
        r = full_grid.copy()
        r.weighted_average_over_mask(mask)
        r.aggregateOverYear(months=range(1,13), aggregationFunction=np.mean)
        r._out_path += str(mask.name) + '/'
        r.write()
    nc_mask_land.close()        
    nc_mask_ocean.close()
        
'''

for model in ACCESS-ESM1-5 CESM2-WACCM CanESM5 EC-Earth3 EC-Earth3-Veg-LR FGOALS-g3 GFDL-ESM4 GISS-E2-1-G IPSL-CM6A-LR MIROC-ES2L MIROC6 MPI-ESM1-2-LR MRI-ESM2-0 UKESM1-0-LL;do sbatch job_4-4.txt overshoot/get_regional_averages.py -m $model -i abs550aer -s ssp119 -r AERmon; done;

for model in ACCESS-ESM1-5 CESM2-WACCM CanESM5 EC-Earth3 EC-Earth3-Veg-LR FGOALS-g3 GFDL-ESM4 GISS-E2-1-G IPSL-CM6A-LR MIROC-ES2L MIROC6 MPI-ESM1-2-LR MRI-ESM2-0 UKESM1-0-LL;do sbatch job_4-4.txt overshoot/get_regional_averages.py -m $model -i pr -s ssp534-over -r Amon; done;

'''

