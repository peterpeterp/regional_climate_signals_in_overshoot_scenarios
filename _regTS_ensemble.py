import sys,glob,datetime,os,gc,importlib,pickle,cftime,re
import xarray as xr
import numpy as np
import pandas as pd
import random

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from distinctipy import distinctipy
except:
    pass
    
from collections import Counter

import itertools
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as statsmodels
from scipy.stats import percentileofscore
from scipy.optimize import curve_fit

from tqdm import tqdm
import _paths

unit_d = dict(tas='K', tasmax='K', tasmin='K', tnnETCCDI='K', txxETCCDI='K', tos='K', pr='mm', rx1dayETCCDI='mm', ua='m/s', va='m/s', abs550aer='1', siconc='%', snc='%', od550aer='1')

models = str('ACCESS-ESM1-5 CESM2-WACCM CanESM5 EC-Earth3 EC-Earth3-Veg-LR FGOALS-g3 GFDL-ESM4 GISS-E2-1-G IPSL-CM6A-LR MIROC-ES2L MIROC6 MPI-ESM1-2-LR MRI-ESM2-0 UKESM1-0-LL').split(' ')

def linear_split_model(x, peak, slope_b, slope_a):
    N2 = int(0.5*len(x))
    out = x.copy() * np.nan
    out[:N2] = x[:N2] * slope_b + peak
    out[N2:] = x[N2:] * slope_a + peak
    return out


class regTS_ensemble():
    def __init__(self, region_name, realm, indicator, scenario, esm, run, source='CMIP6', unit='unit', verbose = False):
        self._source = source
        self._indicator = indicator
        self._scenario = scenario
        self._esm = esm
        self._run = run
        self._realm = realm
        self._region_name = region_name
        self._verbose = verbose
        self._unit = unit
        
        if self._realm is None:
            self._realm = dict(tas='Amon', pr='Amon', abs550aer='AERmon', txxETCCDI='ETCCDI', tnnETCCDI='ETCCDI', rx1dayETCCDI='ETCCDI')[self._indicator]
        
        self._rolling_mean_window = 31
        
        self._tag_list =  [self._source, self._realm, self._indicator, self._scenario, self._esm, self._run, self._region_name] 
        self._in_path = '/work/uc1275/u290372/overshoot/regional/%s/%s/%s/%s/'  %(self._source, self._realm, self._indicator, self._region_name)
        
        self._out_path = '/work/uc1275/u290372/overshoot/regional_ensembles/%s/' %(self._indicator)
        self._file_name = self._out_path + '_'.join([t for t in self._tag_list if t != '*']) + '.pkl' 
        
        
        
    def set_cmap(self):
        if self._indicator in ['tas','txxETCCDI','tnnETCCDI']:
            self._cmap = matplotlib.colormaps['RdBu_r'].resampled(101)
        if self._indicator in ['pr','rx1dayETCCDI']:
            self._cmap = matplotlib.colormaps['BrBG'].resampled(101)
        if self._indicator in ['abs550aer']:
            self._cmap = matplotlib.colormaps['PiYG_r'].resampled(101)
        
    def load_existing_object(self):
        if os.path.isfile(self._file_name):
            with open(self._file_name, 'rb') as fl:
                self.__dict__ = pickle.load(fl)
            return True
        else:
            return False
        
    def save_object(self):
        os.system('mkdir -p ' + self._out_path)
        with open(self._file_name, 'wb') as fl:
            pickle.dump(self.__dict__, fl)
                
    def read_data(self):
        files = sorted(glob.glob(self._in_path + '_'.join(self._tag_list) + '.nc'))
        files = [fl for fl in files if 'piControl' not in fl]
        l = [xr.load_dataset(fl)[self._indicator].squeeze() for fl in files]
        self._y = xr.concat(l, dim='SMR', coords='minimal', compat='override')
        self._y = self._y.assign_coords(SMR = [fl.split(self._indicator+'_')[-1].split('_'+self._region_name)[0] for fl in files])
        
    def read_data_piControl(self):
        files = sorted(glob.glob(str(self._in_path + '_'.join(self._tag_list) +'.nc').replace(self._scenario,'*')))
        files = [fl for fl in files if 'piControl' in fl]
        l = [xr.load_dataset(fl)[self._indicator].squeeze() for fl in files]
        self._y_piC = xr.concat(l, dim='SMR', coords='minimal', compat='override')
        self._y_piC = self._y_piC.assign_coords(SMR = [fl.split(self._indicator+'_')[-1].split('_'+self._region_name)[0] for fl in files])
        
    def read_gmt(self):
        l,SMRs = [],[]
        for smr in self._y.SMR.values:
            fl = '/work/uc1275/u290372/overshoot/regional/%s/Amon/tas/global/CMIP6_Amon_tas_%s_global_annual-mean.nc' %(self._source, smr)
            if os.path.isfile(fl):
                l += [ xr.load_dataset(fl)['tas'].squeeze() ]
                SMRs += [smr]
        self._gmt = xr.concat(l, dim='SMR', coords='minimal', compat='override')
        self._gmt = self._gmt.assign_coords(SMR = SMRs)
        
    def adjust_gmt(self):
        self._gmt -= self._gmt.loc[:,1995:2014].mean('year')
        self._gmt += 0.85

    ######################
    # Timing of peak GMT #
    ######################
    
    def get_timing_of_peak_gmt(self):
        self._peak_years = xr.DataArray(dims=['SMR'], coords=dict(SMR=self._gmt.SMR.values))
        for smr in self._gmt.SMR.values:
            gmt = self._gmt.loc[smr].copy().rolling(year=self._rolling_mean_window, center=True).mean('year')            
            #self._peak_years.loc[smr] = np.round(gmt.year[np.abs(gmt - np.max(gmt)) < 0.025].values.mean())
            self._peak_years.loc[smr] = int(gmt.year.values[np.nanargmax(gmt.values)])

    ######################
    # Data usability checks #
    ######################
            
    def check_gmt_available(self, smr):
        return smr in self._gmt.SMR.values
       
    def check_piControl_available(self, smr):
        SMRs_pC = [s for s in self._y_piC.SMR.values if smr.split('_')[1] == s.split('_')[1]]
        return len(SMRs_pC) > 0
    
    def check_nans_in_period(self, smr, start_year=2020, end_year=2099, allowed_missing=2):
        return bool(np.sum(np.isfinite(self._y.loc[smr, str(start_year):str(end_year)]) == False) < allowed_missing and \
                    np.sum(np.isfinite(self._gmt.loc[smr, str(start_year):str(end_year)]) == False) < allowed_missing)
        
        
    def check_sufficient_years_after_peak(self, smr, n_years = 20):
        y = self._y.loc[smr].copy()
        # remove missing years
        y = y[np.isfinite(y)]
        if len(y) < n_years:
            return False
        # get length of period (with data) after peak - limited to 2100
        periodL = min([y.year.values[-1],2100]) - self._peak_years.loc[smr]
        return bool(periodL > n_years)

    def check_sufficient_cooling_after_peak(self, smr, cooling=0.1):
        g = self._gmt.loc[smr].copy().rolling(year=self._rolling_mean_window, center=True).mean('year')
        g = g[np.isfinite(g)]
        return bool(np.min(g.loc[self._peak_years.loc[smr]:] - g.max()) < - cooling)

    def check_data_usability(self):
        
        check_functions_d = {
            'GMT available' : self.check_gmt_available,
            'piControl available' : self.check_piControl_available,
            'data in relevant period' : self.check_nans_in_period,
            'sufficient years after peak' : self.check_sufficient_years_after_peak,
            'sufficient cooling after peak' : self.check_sufficient_cooling_after_peak,
        }
        
        self._SMR_table = pd.DataFrame(index = self._y.SMR.values, columns=list(check_functions_d.keys()))
        self._SMR_table['scenario'] = [smr.split('_')[0] for smr in self._SMR_table.index]
        self._SMR_table['ESM'] = [smr.split('_')[1] for smr in self._SMR_table.index]
        self._SMR_table['run'] = [smr.split('_')[2] for smr in self._SMR_table.index]
                             
        self._usable_smrs = []
        for smr in self._y.SMR.values:
            scen,mod,run = smr.split('_')
            usable = []

            for check_name, check_func in check_functions_d.items():
                result = check_func(smr)
                self._SMR_table.loc[smr,check_name] = result
                usable += [result]
                if result == False:
                    break
                
            if np.all(usable):
                self._usable_smrs += [smr]
                

    def get_rid_of_uncomplete(self):        
        self._y = self._y.loc[self._usable_smrs]     
        self._gmt = self._gmt.loc[self._usable_smrs]     
        self._peak_years = self._peak_years.loc[self._usable_smrs] 
        
        
    def SMR_table(self):
        #self._SMR_table = pd.DataFrame(index = self._y.SMR.values, columns=list(check_functions_d.keys()))
        self._SMR_table['scenario'] = [smr.split('_')[0] for smr in self._SMR_table.index]
        self._SMR_table['ESM'] = [smr.split('_')[1] for smr in self._SMR_table.index]
        self._SMR_table['run'] = [smr.split('_')[2] for smr in self._SMR_table.index]        
        self._SMR_table['peak year'] = np.array(self._peak_years.loc[self._y.SMR.values], np.int)
        self._SMR_table['period length'] = 2100 - self._SMR_table['peak year']
        self._SMR_table['period before'] = ['%s-%s' %(p - l, p) for p,l in zip(self._SMR_table['peak year'],self._SMR_table['period length'])]
        self._SMR_table['period after'] = ['%s-%s' %(p, p + l) for p,l in zip(self._SMR_table['peak year'],self._SMR_table['period length'])]
                
        
    def exclude_models(self):
        to_exclude = ['NorESM2-LM', 'CNRM-ESM2-1']
        useful_smrs = []
        for smr in self._y.SMR.values:
            scen,mod,run = smr.split('_')  
            if mod not in to_exclude:
                useful_smrs.append(smr)
            
        self._y = self._y.loc[useful_smrs]     
        self._gmt = self._gmt.loc[useful_smrs]     
        self._peak = self._peak_years.loc[useful_smrs] 
        try:
            self._trend_vs_gmt = self._trend_vs_gmt.loc[useful_smrs] 
            self._lr = self._lr_perc.loc[useful_smrs] 
            self._lr = self._lr_slope.loc[useful_smrs] 
            self._mk = self._mk_perc.loc[useful_smrs] 
        except:
            pass
                                
    def get_models(self, filename=None, overwrite_colors_and_markers=False):   
        self._model_array = xr.DataArray([m.split('_')[1] for m in self._y.SMR.values], dims=['SMR'], coords=dict(SMR=self._y.SMR.values))
        self._models = np.unique([m.split('_')[1] for m in self._y.SMR.values])
        if filename is None:
            filename = _paths.dataPath + 'model_colors_markers_%s.pkl' %('.'.join([m[:2]+m[-3:] for m in sorted(self._models)]))
        if os.path.isfile(filename) and overwrite_colors_and_markers == False:
            with open(filename, 'rb') as fl:
                tmp = pickle.load(fl)
                self._color_d = tmp['color_d']
                self._marker_d = tmp['marker_d']
        else:
            print('choosing new colors and markers')
            self._color_d = {m:c for m,c in zip(self._models,distinctipy.get_colors(len(self._models)))}
            markers = itertools.cycle(['v','^','s','o','P','>','<','X','d','D'])
            self._marker_d = {m:next(markers) for m in self._models}
            with open(filename, 'wb') as fl:
                pickle.dump({'color_d':self._color_d, 'marker_d':self._marker_d}, fl)
        
    def plot_model_legend(self):
        fig,ax = plt.subplots(nrows=1, figsize=(3,1.7))
        for model in self._models:
            n = len(self.get_smrs_for_model(model))
            if np.isfinite(n):
                if n > 0:
                    ax.scatter([],[],color=self._color_d[model], marker=self._marker_d[model], edgecolor='k', label='%s (%s)' %(model, n), s=100)
        ax.legend(ncol=2)
        ax.axis('off')
        return fig,ax
        
    def get_smrs_for_model(self, model):
        smrs = []
        for smr in self._y.SMR.values:
            scen,mod,run = smr.split('_')
            if mod==model:
                smrs.append(smr)
        return smrs   

    def get_smrs_for_scenario_model(self, scenario, model):
        smrs = []
        for smr in self.get_smrs_for_model(model):
            scen,mod,run = smr.split('_')
            if scen==scenario:
                smrs.append(smr)
        return smrs   

    ######################
    # Trend analysis     #
    ######################

    def get_relevant_data_for_trend_analysis(self, smr, rolling_mean=False):
        scen,mod,run = smr.split('_')

        # get data of run
        y = self._y.loc[smr]
        gmt = self._gmt.loc[smr]
        
        # rolling mean ?
        if rolling_mean:
            y = y.rolling(year=self._rolling_mean_window, center=True).mean('year')
            gmt = gmt.rolling(year=self._rolling_mean_window, center=True).mean('year')
        
        # remove missing years
        y = y[np.isfinite(y)]
        gmt = gmt[np.isfinite(gmt)]
        y = y[np.isin(y.year, gmt.year)]
        gmt = gmt[np.isin(gmt.year, y.year)]
        
        # get length of period (with data) after peak - limited to 2100
        peak_year = int(self._peak_years.loc[smr])
        periodL = min([y.year.values[-1],2100]) - peak_year

        # get slope before
        y_before = y.loc[peak_year - periodL : peak_year]
        gmt_before = gmt.loc[peak_year - periodL : peak_year]
        
        # and after
        y_after = y.loc[peak_year : peak_year + periodL]
        gmt_after = gmt.loc[peak_year : peak_year + periodL]
        
        ## adjust to peak
        #y_peak = float(y_before.copy().values[-1])
        #gmt_peak = float(gmt_before.copy().values[-1])

        #y_before = y_before - y_peak
        #y_after = y_after - y_peak
        #gmt_before = gmt_before - gmt_peak
        #gmt_after = gmt_after - gmt_peak
        
        # get all piControl runs
        SMRs_pC = [smr for smr in self._y_piC.SMR.values if mod in smr] 
        yPC_list = []
        for smr_pc in SMRs_pC:
            yPC = self._y_piC.loc[smr_pc]
            # remove missing
            yPC = yPC[np.isfinite(yPC)]
            # remove trend 
            slope_PC = statsmodels.OLS(np.expand_dims(yPC.values, axis=-1),
                                       statsmodels.add_constant(yPC.year.values, prepend=False)).fit().params[0]
            yPC = yPC - yPC.year * slope_PC + np.mean(yPC.year * slope_PC)
            yPC_list.append(yPC)
            
        return y_before, y_after, gmt_before, gmt_after, periodL, yPC_list
    
    def check_whether_dep_on_gmt_is_significantly_different_one_SMR(self, smr, nBoot=10000):

        y_before, y_after, gmt_before, gmt_after, periodL, yPC_list = self.get_relevant_data_for_trend_analysis(smr, rolling_mean=True)

        y = xr.concat([y_before, y_after], dim='year')
        gmt_peak = xr.concat([gmt_before[-15:],gmt_after[:15]], dim='year').mean('year')
        gmt = xr.concat([gmt_before, gmt_after], dim='year')
        
        gmt -= gmt.max()
        
        y_peak, slope_before, slope_after = curve_fit(linear_split_model, gmt, y)[0]
        
        random_slopes_before = []
        for i in range(nBoot):
            indices = np.random.choice(range(len(y_before)), len(y_before), replace=True)
            try:
                slope,intercept = statsmodels.OLS(np.expand_dims(y_before.values[indices], axis=-1), 
                                                                  statsmodels.add_constant(gmt_before.values[indices], prepend=False)).fit().params
                random_slopes_before += [slope]
            except:
                random_slopes_before += [np.nan]
            
        random_slopes_before = np.array(random_slopes_before)

        if np.sum(np.isnan(random_slopes_before)) > 0:
            print(smr, np.sum(np.isnan(random_slopes_before)))
       
        return (slope_before, 
                slope_after, 
                random_slopes_before, 
                y,
                gmt,
                y_peak,
                int(len(y)/2),
               )



    def check_whether_dep_on_gmt_is_significantly_different(self):
        self._trend_vs_gmt = xr.DataArray(dims=['SMR','period'], coords=dict(SMR=self._y.SMR.values,
                                                                  period=['before','after']))
        self._slope_after_sig_different_from_before = xr.DataArray(dims=['SMR'], coords=dict(SMR=self._y.SMR.values))
        for smr in self._y.SMR.values:     
            #print(smr)
            (slope_before, 
                slope_after, 
                random_slopes_before, 
                y,
                gmt,
                y_peak,
                N_half,
               ) = self.check_whether_dep_on_gmt_is_significantly_different_one_SMR(smr)
            
            self._trend_vs_gmt.loc[smr] = [slope_before,slope_after]
            
            bounds = np.nanpercentile(random_slopes_before, [5,95])
            slope_after_could_be_found_in_before = (slope_after > bounds[0]) & (slope_after < bounds[1])

            self._slope_after_sig_different_from_before.loc[smr] = slope_after_could_be_found_in_before == False
       
    def analyse_lr_vs_nat_var_one_smr(self, smr):
        y_before, y_after, gmt_before, gmt_after, periodL, yPC_list = self.get_relevant_data_for_trend_analysis(smr)
        random_lrs_list = []
        for yPC in yPC_list:
            random_lrs = []
            N = len(yPC) - periodL
            for s in range(N):
                # get piControl period and perform linear regression
                yy = yPC[s:s+periodL]
                random_lrs.append(statsmodels.OLS(np.expand_dims(yy.values, axis=-1), 
                                                  statsmodels.add_constant(yy.year.values, prepend=False)).fit().params[0])
            random_lrs_list.append(np.array(random_lrs))

        lr_before = statsmodels.OLS(np.expand_dims(y_before.values, axis=-1), 
                                                  statsmodels.add_constant(y_before.year.values, prepend=False)).fit().params[0]
        lr_after = statsmodels.OLS(np.expand_dims(y_after.values, axis=-1), 
                                                  statsmodels.add_constant(y_after.year.values, prepend=False)).fit().params[0]
        lr_perc_before = percentileofscore(np.concatenate(random_lrs_list), lr_before, kind='rank')
        lr_perc_after = percentileofscore(np.concatenate(random_lrs_list), lr_after, kind='rank')
        
        return lr_before, lr_after, random_lrs_list, lr_perc_before, lr_perc_after
    
    def analyse_lr_vs_nat_var(self):        
        self._lr_perc = xr.DataArray(dims=['SMR','period'], coords=dict(SMR=self._y.SMR.values, period=['before','after']))
        self._lr_slope = xr.DataArray(dims=['SMR','period'], coords=dict(SMR=self._y.SMR.values, period=['before','after']))
        
        for smr in self._y.SMR.values:
            lr_before, lr_after, random_lrs_list, lr_perc_before, lr_perc_after = self.analyse_lr_vs_nat_var_one_smr(smr)
            self._lr_perc.loc[smr,'before'] = lr_perc_before
            self._lr_perc.loc[smr,'after'] = lr_perc_after
            self._lr_slope.loc[smr,'before'] = lr_before
            self._lr_slope.loc[smr,'after'] = lr_after
        
    def evaluate_whether_trend_exceeds_from_natVar(self, model, period):
        smrs = self.get_smrs_for_model(model) 
        slope_perc = self._lr_perc.loc[smrs,period] 
        sig = np.abs(slope_perc - 50) > self._trend_threshold
        sign = np.sign(slope_perc - 50)
        c = Counter(sign[sig].values)
        N = sign.shape[0]
        
        # for "large" ensembles:
        if N >= 4:
            if Counter(sign.values)[1] >= 0.75 * N:
                return 1
            elif Counter(sign.values)[-1] >= 0.75 * N:
                return -1
            elif sig.sum() <= 0.25 * N:
                return 0
            else:
                return np.nan
            
        else:
            if c[1] >= 0.75 * N:
                return 1
            elif c[-1] >= 0.75 * N:
                return -1        
            elif sig.sum() <= 0.25 * N:
                return 0
            else:
                return np.nan
    
    def evaluate_whether_dependence_on_GMT_is_significantly_different(self, model):
        smrs = self.get_smrs_for_model(model) 
        isDifferent = self._slope_after_sig_different_from_before.loc[smrs]
        isLarger = np.sign(self._trend_vs_gmt.loc[smrs,'after'] - self._trend_vs_gmt.loc[smrs,'before']) 
        c = Counter(isLarger.values[np.array(isDifferent.values, bool)])
        N = isDifferent.shape[0]
        #print(model)
        #print(isDifferent.values)
        #print(isLarger.values)
        #print(c, N)
        
        if c[1] >= 0.75 * N:
            result = True
        elif c[-1] >= 0.75 * N:
            result = True
        else:
            result = False
        
        #print(result)
        return result
    
    def reversed(self): return 'reversed'
    def inside_nat_var(self): return 'inside natural variability'
    def continued_increase(self): return 'continued increase'
    def continued_decrease(self): return 'continued decrease'
    def increase_part_reversed(self): return 'increase partially reversed'
    def decrease_part_reversed(self): return 'decrease partially reversed'
    def increase_overcomp(self): return 'increase over compensated'
    def decrease_overcomp(self): return 'decrease over compensated'
        
    def define_behaviour_dict(self):        
        self._behaviour_dict = {
            self.continued_decrease(): dict(c = self._cmap(10), ec = 'none', h = ''),  
            self.reversed() : dict(c = (0.6367205267327445, 0.9903725858012087, 0.5389780920684211), ec = 'none', h = ''),  
            self.continued_increase() : dict(c = self._cmap(90), ec = 'none', h = ''),  
            self.increase_part_reversed() : dict(c = 'white', ec = self._cmap(70), h = ''),  
            self.increase_overcomp() : dict(c = self._cmap(30), ec = 'white', h = ''),  
            self.decrease_overcomp() : dict(c = self._cmap(70), ec = 'white', h = ''),  
            self.decrease_part_reversed() : dict(c = 'white', ec = self._cmap(30), h = ''),  
            self.inside_nat_var() : dict(c = '#CFCFCF', ec = 'none', h = ''),  
        }
        
        self._behaviour_seq = [
            self.continued_decrease(), 
            self.decrease_part_reversed(), 
            self.increase_overcomp(), 
            self.reversed(),
            self.decrease_overcomp(),
            self.increase_part_reversed(),
            self.continued_increase(),
            self.inside_nat_var(), 
        ]

    def classify_behaviour_more_less(self, model):
        robust_trend, slope = [],[]
 
        rob_b = self.evaluate_whether_trend_exceeds_from_natVar(model, 'before')
        rob_a = self.evaluate_whether_trend_exceeds_from_natVar(model, 'after')
        rob = '%sI%s' %(rob_b, rob_a)

        sig_different = self.evaluate_whether_dependence_on_GMT_is_significantly_different(model)
        
        smrs = self.get_smrs_for_model(model)
        dep_b = self._trend_vs_gmt.loc[smrs,'before'].median('SMR')
        dep_a = self._trend_vs_gmt.loc[smrs,'after'].median('SMR')
           
        if rob_a == False and rob_b==False:
            return self.inside_nat_var()
        if sig_different == False:
            return self.reversed()
        if sig_different:
            if np.sign(dep_a) != np.sign(dep_b):
                if np.sign(dep_b) > 0:
                    return self.continued_increase()
                if np.sign(dep_b) < 0:
                    return self.continued_decrease()
            if np.sign(dep_a) == np.sign(dep_b):
                if np.abs(dep_b) > np.abs(dep_a):
                    if np.sign(dep_b) > 0:
                        return self.increase_part_reversed()
                    if np.sign(dep_b) < 0:
                        return self.decrease_part_reversed()
                if np.abs(dep_b) < np.abs(dep_a):
                    if np.sign(dep_b) > 0:
                        return self.increase_overcomp()
                    if np.sign(dep_b) < 0:
                        return self.decrease_overcomp()
        
        return None
    
        assert False, 'behavior not classifiable?'
            
        
    def get_behaviours(self):
        so = {trend_type:[] for trend_type in self._behaviour_seq}
        for model in self._models:
            behavior = self.classify_behaviour_more_less(model)
            if behavior is not None:
                so[behavior].append(model)
        co = {k:len(v) for k,v in so.items()}
        return so, co, np.sum(list(co.values()))
        
    def plot_behaviour_pie(self, ax=None, markers=False, threshold=45):
        self.define_behaviour_dict()
        self._trend_threshold = threshold

        if ax is None:
            fig,ax = plt.subplots(nrows=1, subplot_kw=dict(projection='polar'), figsize=(4,4))
        ax.axis('off')
        ax.set_rlim(0,1)
                
        so, co, N = self.get_behaviours()
        angle_direction = -1
        phi = 0.5*np.pi
        phi += angle_direction * 2 * np.pi/N * 0.5 * len(so[self.inside_nat_var()])
        
        for i in self._behaviour_seq:
            trend_count,models = co[i],so[i]
            d = self._behaviour_dict[i]
            phi0 = phi

            runs = [len(self.get_smrs_for_model(model)) for model in models]
            for model in np.array(models)[np.argsort(runs)[::-1]]:
                if len(self.get_smrs_for_model(model)) >= 4:
                    r = 0.85
                else:
                    r = 0.66
                                
                ax.fill_between(np.linspace(phi,phi+angle_direction*2*np.pi/N,100),0,r, facecolor=d['c'], linewidth=0.0)
                ax.fill_between(np.linspace(phi,phi+angle_direction*2*np.pi/N,100),0,r, facecolor='none', edgecolor='k', linewidth=0.4)
                #ax.plot(np.linspace(phi,phi + angle_direction *2* np.pi/N,5),[r]*5,color='k',linewidth=1)
                if d['ec'] != 'none':
                    ax.fill_between(np.linspace(phi,phi+angle_direction*2*np.pi/N,100),0,r*0.66, facecolor=d['ec'], linewidth=0.0)

                
                
                phi_ = phi + angle_direction *2 * np.pi/N
                if markers:
                    ax.scatter([phi+angle_direction*1*np.pi/N], [r+0.1], marker=self._marker_d[model], color=self._color_d[model], 
                                  s=100, edgecolor='k', zorder=2, clip_on=False)  
                
                #if r == 0.85:
                #    ax.plot([phi,phi],[r,0.66],color='k',linewidth=1)
                #    ax.plot([phi+angle_direction*2*np.pi/N]*2,[r,0.66],color='k',linewidth=1)
                phi = phi + angle_direction *2* np.pi/N
            
    def plot_vs_gmt(self, smrs, ax=None):
        
        if ax is None:
            fig,ax = plt.subplots(nrows=1, figsize=(4,3), dpi=100)
        
        all_smrs = []
        for smr in smrs:
            all_smrs += [self.check_whether_dep_on_gmt_is_significantly_different_one_SMR(smr, 1)]
        
        ylim,xlim = [],[]
        for l in all_smrs:
            slope_before, slope_after, random_slopes, y, gmt, y_peak, N_half = l
            usable = gmt >= -0.3
            ylim += list(y[usable] - y_peak)
            xlim += list(gmt[usable])
            ax.plot(gmt[:N_half], y[:N_half] - y_peak, color='grey', linewidth=0.5)
            ax.plot(-gmt[N_half:], y[N_half:] - y_peak, color='grey', linewidth=0.5)
            ax.plot([-0.3,0], np.array([-0.3,0]) * slope_before, color='darkmagenta', linewidth=0.5, linestyle=':')
            ax.plot([0,0.3], np.array([0,-0.3]) * slope_after, color='darkcyan', linewidth=0.5, linestyle=':')

        ax.plot([-0.3,0], np.array([-0.3,0]) * np.nanpercentile([l[0] for l in all_smrs], 50), color='darkmagenta', linewidth=2)
        ax.plot([0,0.3], np.array([0,-0.3]) * np.nanpercentile([l[1] for l in all_smrs], 50), color='darkcyan', linewidth=2)
        ax.plot([0,0.3], np.array([0,-0.3]) * np.nanpercentile([l[0] for l in all_smrs], 50), color='darkmagenta', linewidth=2, linestyle='--')

        ax.axvline(x=0, color='k', zorder=0)
        ax.axhline(y=0, color='k', zorder=0)
        ax.set_xlabel('GMT relative to peak warming [K]')
        ax.set_xticks(np.arange(-0.3,0.3,0.1).round(1))            
        ax.set_xticklabels(-np.abs(np.arange(-0.3,0.3,0.1).round(1)))
        ax.set_ylabel('%s anomaly relative to\npeak warming in [%s]' %(self._indicator,self._unit))
        ax.set_ylim(np.nanmin(ylim),np.nanmax(ylim))
        ax.set_xlim(np.nanmin(xlim),-np.nanmin(xlim))
        plt.tight_layout()
        return ax
   
    
    def plot_trend_hists(self, smr):
        (slope_before, slope_after, random_slopes, y, gmt, y_peak, N_half) = \
            self.check_whether_dep_on_gmt_is_significantly_different_one_SMR(smr, nBoot=1000)

        fig,ax = plt.subplots(nrows=1, figsize=(4,3))
        
        vals = list(random_slopes) + [slope_after,slope_before]
        maxabs = np.max(np.abs(vals)) * 1.5
        norm = matplotlib.colors.Normalize(vmin=-maxabs, vmax=maxabs)
        for l,h in zip(np.linspace(-maxabs,maxabs,20)[1:],np.linspace(-maxabs,maxabs,20)[:-1]):
            ax.axvspan(l,h, color=self._cmap(norm(np.mean([l,h]))))

        ax.hist(random_slopes, color='gray', bins=100, alpha=1, density=True)
        ax.axvline(slope_before, color='darkmagenta', linewidth=3)
        ax.axvline(slope_after, color='darkcyan', linewidth=3)
        #ax.axvline(np.percentile(random_slopes, [95]), color='gray')
        #ax.axvline(np.percentile(random_slopes, [5]), color='gray')

        ax.set_xlim(np.min([np.min(vals)*0.5,np.min(vals)*1.5]) , np.max(vals)* 1.2)
        ax.set_xlabel('change in %s per degree of GMT warming [%s]' %(self._indicator,self._unit))
        ax.set_ylabel('probability density')
        plt.tight_layout()
        return ax
    
    def plot_vs_gmt_scatter(self, trend_threshold=45, ax=None, explainers=False, zoomed=False):
        self.define_behaviour_dict()
        self._trend_threshold=trend_threshold
        if ax is None:
            fig,ax = plt.subplots(nrows=1, figsize=(4,3), dpi=100)
        ax.set_ylabel('change in %s per degree of warming [%s / K]' %(self._indicator, self._unit))
        ax.set_xlabel('change in %s per degree of cooling [%s / K]' %(self._indicator, self._unit))
        ax.plot([0,1], [0,1], color='gray', transform=ax.transAxes, zorder=0)
        ax.axhline(0, color='gray')
        ax.axvline(0, color='gray')
        for mod in self._models:
            beh = self.classify_behaviour_more_less(mod)
            ec = {True:'k', False:'w'}[self.evaluate_whether_dependence_on_GMT_is_significantly_different(mod)]
            alpha=1
            slope_before = np.median(self._trend_vs_gmt.loc[self.get_smrs_for_model(mod),'before'])
            slope_after = np.median(self._trend_vs_gmt.loc[self.get_smrs_for_model(mod),'after'])
            ax.scatter(slope_after, slope_before,
                       marker=self._marker_d[mod], color=self._color_d[mod], edgecolor=ec, alpha=alpha, s=100, zorder=20, label=mod+{True:'*', False:''}[self.evaluate_whether_dependence_on_GMT_is_significantly_different(mod)])
            
        if zoomed:
            maxabs = np.nanpercentile(np.abs(np.append(before,after)), 90)
        else:
            maxabs = np.max(np.abs(list(ax.get_ylim()) + list(ax.get_xlim())))
        ax.set_xlim(-maxabs,maxabs)
        ax.set_ylim(-maxabs,maxabs)
        ax.set_xticklabels([-t for t in ax.get_xticks().round(2)])

        #SN = self._trend_before_likely_SN_ratio
        SN = 0.5
        ax.fill_between([0,maxabs], [0,0], [0,maxabs], facecolor=self._cmap(30), edgecolor='k', linewidth=0.0, zorder=0)
        ax.fill_between([0,maxabs], [0,0], [-maxabs,-maxabs], facecolor=self._cmap(10), edgecolor='k', linewidth=0.0, zorder=0)
        ax.fill_between([-maxabs,0], [-maxabs,0], [-maxabs,-maxabs], facecolor=self._cmap(30), edgecolor='k', linewidth=0.0, zorder=0)
        ax.fill_between([0,-maxabs], [0,0], [0,-maxabs], facecolor=self._cmap(70), edgecolor='k', linewidth=0.0, zorder=0)
        ax.fill_between([-maxabs,0], [0,0], [maxabs,maxabs], facecolor=self._cmap(90), edgecolor='k', linewidth=0.0, zorder=0)
        ax.fill_between([0,maxabs], [maxabs,maxabs], [0,maxabs], facecolor=self._cmap(70), edgecolor='k', linewidth=0.0, zorder=0)
        
        if explainers:
            ax.annotate('decrease after peak\nis stronger than\nincrease before peak', rotation=-90,   fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3",fc=self._cmap(30), ec="none", lw=2),
               xy=(1.1,1.1), xycoords='axes fraction', ha='left', va='top', annotation_clip=False)
            arrowprops = dict(arrowstyle= '-|>', color='k', lw=1, ls='-')
            ax.annotate('', xy=(1.05, 0.5), xytext=(1.05, 0.9), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops, annotation_clip=False)
            ax.annotate('decrease before\nand after peak', color='white', fontsize=8, rotation=-90,
                        bbox=dict(boxstyle="round,pad=0.3",fc=self._cmap(10), ec="none", lw=2),
                       xy=(1.1,0.45), xycoords='axes fraction', ha='left', va='top', annotation_clip=False)
            ax.annotate('', xy=(1.05, 0), xytext=(1.05, 0.5), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops, annotation_clip=False)

            ax.annotate('increase before\nand after peak', color='white',  fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3",fc=self._cmap(90), ec="none", lw=2),
                       xy=(0.25,1.1), xycoords='axes fraction', ha='center', va='bottom', annotation_clip=False)
            ax.annotate('', xy=(0, 1.05), xytext=(0.5, 1.05), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops, annotation_clip=False)
            ax.annotate('increase before peak\nis stronger than\ndecrease before peak', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3",fc=self._cmap(70), ec="none", lw=2),
                       xy=(0.55,1.1), xycoords='axes fraction', ha='left', va='bottom', annotation_clip=False)
            ax.annotate('', xy=(0.5, 1.05), xytext=(0.9, 1.05), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops, annotation_clip=False)
        
        return ax

    
    def plot_scatter_diff_at(self, d_gmt=0.2, trend_threshold=45, ax=None, relative=False, explainers=False, zoomed=False, check_robustness=False):
        self.define_behaviour_dict()
        self._trend_threshold=trend_threshold
        if ax is None:
            fig,ax = plt.subplots(nrows=1, figsize=(4,4), dpi=100)
        before = self._trend_vs_gmt.loc[:,'before'].groupby(self._model_array).median('SMR')
        before = before  * (d_gmt)
        after = self._trend_vs_gmt.loc[:,'after'].groupby(self._model_array).median('SMR') 
        after = after  * (d_gmt)
        if relative:
            ref = self._y.loc[:,2000:2030].mean('year').groupby(self._model_array).median('SMR')
            before = before / ref * 100
            after = after / ref * 100
            ax.set_ylabel('change in %s for %s warming\nin %s or current %s' %(self._indicator, d_gmt, '%', self._indicator))
            ax.set_xlabel('change in %s for %s cooling\nin %s or current %s' %(self._indicator, d_gmt, '%', self._indicator))
        else:
            ax.set_ylabel('change in %s for %s warming [%s]' %(self._indicator, d_gmt, self._unit))
            ax.set_xlabel('change in %s for %s cooling [%s]' %(self._indicator, d_gmt, self._unit))
        ax.plot([0,1], [0,1], color='gray', transform=ax.transAxes, zorder=0)
        ax.axhline(0, color='gray')
        ax.axvline(0, color='gray')
        for mod in self._models:
            beh = self.classify_behaviour_more_less(mod)
            if ('?' in beh or 's' in beh) and check_robustness:
                alpha=0.5
                ec = 'gray'
            else:
                alpha = 1
                ec = 'k'
            ax.scatter(after.loc[mod], before.loc[mod], marker=self._marker_d[mod], color=self._color_d[mod], edgecolor=ec, alpha=alpha, s=100, zorder=20)
            
        if zoomed:
            maxabs = np.nanpercentile(np.abs(np.append(before,after)), 90)
        else:
            maxabs = np.max(np.abs(list(ax.get_ylim()) + list(ax.get_xlim())))
        ax.set_xlim(-maxabs,maxabs)
        ax.set_ylim(-maxabs,maxabs)
        ax.set_xticklabels([-t for t in ax.get_xticks().round(2)])
        ax.fill_between([0.5,1], [0.5,0.5], [0.5,0.75], facecolor=self._cmap(40), edgecolor='none', linewidth=0.0, transform=ax.transAxes, zorder=0)
        ax.fill_between([0.5,1], [0.5,0.5], [0,0], facecolor=self._cmap(30), edgecolor='none', linewidth=0.0, transform=ax.transAxes, zorder=0)
        ax.fill_between([0.25,0.5], [0,0.5], [0,0], facecolor=self._cmap(40), edgecolor='none', linewidth=0.0, transform=ax.transAxes, zorder=0)
        ax.fill_between([0.5,0], [0.5,0.5], [0.5,0.25], facecolor=self._cmap(60), edgecolor='none', linewidth=0.0, transform=ax.transAxes, zorder=0)
        ax.fill_between([0,0.5], [0.5,0.5], [1,1], facecolor=self._cmap(70), edgecolor='none', linewidth=0.0, transform=ax.transAxes, zorder=0)
        ax.fill_between([0.5,0.75], [1,1], [0.5,1], facecolor=self._cmap(60), edgecolor='none', linewidth=0.0, transform=ax.transAxes, zorder=0)
        
        if explainers:
            ax.annotate('decrease after peak\nis twice as strong as\nincrease before peak', 
                bbox=dict(boxstyle="round,pad=0.3",fc=self._cmap(40), ec="none", lw=2),
               xy=(1.1,0.75), xycoords='axes fraction', ha='left', va='top', annotation_clip=False)
            arrowprops = dict(arrowstyle= '-|>', color='k', lw=1, ls='-')
            ax.annotate('', xy=(1.05, 0.5), xytext=(1.05, 0.75), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops, annotation_clip=False)
            ax.annotate('decrease before\nand after peak',
                        bbox=dict(boxstyle="round,pad=0.3",fc=self._cmap(30), ec="none", lw=2),
                       xy=(1.1,0.35), xycoords='axes fraction', ha='left', va='top', annotation_clip=False)
            ax.annotate('', xy=(1.05, 0), xytext=(1.05, 0.5), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops, annotation_clip=False)

            ax.annotate('decrease before\nand after peak',
                        bbox=dict(boxstyle="round,pad=0.3",fc=self._cmap(70), ec="none", lw=2),
                       xy=(0.25,1.1), xycoords='axes fraction', ha='center', va='bottom', annotation_clip=False)
            ax.annotate('', xy=(0, 1.05), xytext=(0.5, 1.05), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops, annotation_clip=False)
            ax.annotate('decrease after peak\nis half as strong as\nincrease before peak',
                        bbox=dict(boxstyle="round,pad=0.3",fc=self._cmap(60), ec="none", lw=2),
                       xy=(0.55,1.1), xycoords='axes fraction', ha='left', va='bottom', annotation_clip=False)
            ax.annotate('', xy=(0.5, 1.05), xytext=(0.75, 1.05), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops, annotation_clip=False)
        
        return ax

    
    def plot_diff_after_before(self, d_gmt=0.2, trend_threshold=45, ax=None, relative=False, explainers=False, zoomed=False, check_robustness=False):
        self.define_behaviour_dict()
        self._trend_threshold=trend_threshold
        if ax is None:
            fig,ax = plt.subplots(nrows=1, figsize=(1,4), dpi=100)
        before = self._trend_vs_gmt.loc[:,'before'].groupby(self._model_array).median('SMR')
        before = before * (d_gmt)
        after = self._trend_vs_gmt.loc[:,'after'].groupby(self._model_array).median('SMR') 
        after = after * (d_gmt)
        if relative:
            ref = self._y.loc[:,2000:2030].mean('year').groupby(self._model_array).median('SMR')
            before = before / ref * 100
            after = after / ref * 100
            ax.set_ylabel('change in %s due to an overshoot of %sK\n in %s of current %s' %(self._indicator, d_gmt, '%', self._indicator)) 
        else:
            ax.set_ylabel('change in %s due to an overshoot of %sK [%s]' %(self._indicator, d_gmt, self._unit)) 
        values = before - after
        values = values.sortby(values)
        swarm = sns.swarmplot(y=values, ax=ax, alpha=0)
        xy = [xy for xy in swarm.get_children() if isinstance(xy, matplotlib.collections.PathCollection)][0]
        xx,yy = xy.get_offsets()[:,0],xy.get_offsets()[:,1]
        xx = xx / np.abs(xx).max()
        for x,y,model in zip(xx,yy,values.group.values):
            ax.scatter(x, y, marker=self._marker_d[model], color=self._color_d[model], edgecolor='k', s=80, clip_on=False, zorder=9)
        ax.set_xlim(-1.4,1.4)
        ax.axhline(0, color='gray', zorder=0)
        maxabs = np.max(np.abs(list(ax.get_ylim())))
        ax.set_ylim(-maxabs, maxabs)
        
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_xticks([])
        return ax

