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
sys.path.append('/home/u/u290372/scripts/cmip6_ensemble_scripts')
import _paths as _paths; importlib.reload(_paths)
import _regTS_ensemble
sys.path.append('/home/u/u290372/git-projects/regional_panels_on_map')
import regional_panels_on_map

oo = _regTS_ensemble.regTS_ensemble(region_name='MED-land', realm='Amon', indicator='pr', scenario='*' , esm='*', run='*')
oo.load_existing_object()
oo.check_data_usability()
oo.SMR_table()
print(oo._SMR_table[
                ['scenario', 'ESM', 'run', 'peak year', 'period before', 'period after', 'period length']
                    ].to_latex(
                        index=False, formatters={"name": str.upper}, float_format="{:.1f}".format)
     )