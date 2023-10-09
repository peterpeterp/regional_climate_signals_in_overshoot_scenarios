import sys,glob,datetime,os,cftime,importlib,gc
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
#import geopandas
#import regionmask


import _paths as _paths; importlib.reload(_paths)


unit_d = dict(tas='K', tasmax='K', tasmin='K', tnnETCCDI='K', txxETCCDI='K', tos='K', pr='mm', rx1dayETCCDI='mm', ua='m/s', va='m/s')


class oneRun():
    def __init__(self,scenario,model,run,**kwargs):
        self._scenario = scenario
        self._model = model
        self._run = run
        self._SMR = scenario+'_'+model+'_'+run

        defaults = dict(
            level=None,
            time_format=None,
            mask=None,
            shape_for_mask=None,
            closest_grid=None,
            extent=None,
            regrid_prec=1,
            verbose=False,
            realm=None,
            preprocessing_function=None,
        )
        defaults.update(kwargs)
        for k,v in defaults.items():
            self.__dict__['_'+k] = v
            
        defaults.update(kwargs)
        for k,v in defaults.items():
            self.__dict__['_'+k] = v    
        
        self._history = ''
        
    def inherit_attributes_from_ensemble(self, ensembleObject):
        for k,v in ensembleObject.__dict__.items():
            if k in ['_'+w for w in \
                     ['indicator','scenario','verbose','level','time_format','mask_search_path','realm',
                      'closest_grid', 'cutoff_years', 'required_years', 'preprocessing_function',
                      'region_name','shape_for_mask','regrid_prec',
                      'extent','regrid','months','overwrite','source']]:
                self.__dict__[k] = v

        
    def get_files(self, data_paths = ['/work/uc1275/u290372/CMIP6/data/','/pool/data/CMIP6/data/']):            
        SMRs = []
        self._hist_files, self._ssp585_files = [],[]
        if self._source == 'CMIP6':
            if self._indicator in ['tas','pr','ua','va','uas','sos','tos','snc', 'tasmax', 'tasmin']:
                if self._realm is None:
                    if self._indicator in ['tas','pr','ua','va','uas']:
                        realm = 'Amon'
                    elif self._indicator in ['sos','tos']:
                        realm = 'Omon'
                    elif self._indicator in ['snc']:
                        realm = 'SImon'
                    self._realm = realm

                self._scen_files = []
                for path in data_paths:
                    self._scen_files += glob.glob(path+'/*/*/%s/%s/%s/%s/%s/*/*/*'
                                             %(self._model,self._scenario,self._run,self._realm,self._indicator))

                if 'ssp' in self._scenario:
                    self._hist_files = []
                    for path in data_paths:
                        self._hist_files += glob.glob(path+'/*/*/%s/%s/%s/%s/%s/*/*/*'
                                             %(self._model,'historical',self._run,self._realm,self._indicator))
                if 'ssp534-over' == self._scenario:
                    self._ssp585_files = []
                    for path in data_paths:
                        self._ssp585_files += glob.glob(path+'/*/*/%s/%s/%s/%s/%s/*/*/*'
                                             %(self._model,'ssp585',self._run,self._realm,self._indicator))

                SMRs += ['_'.join([fl.split('/')[i] for i in [-7,-8,-6]]) for fl in self._scen_files]

            if self._indicator in ['rx1dayETCCDI','txxETCCDI','tnnETCCDI']:
                self._scen_files = glob.glob('/work/uc1275/u290372/data/etccdi_indices_*/'+self._indicator+'_mon_'+self._model+'_'+self._scenario+'_'+self._run+'_*.nc')
                if 'ssp' in self._scenario:
                    self._hist_files = glob.glob('/work/uc1275/u290372/data/etccdi_indices_*/'+ self._indicator+ '_mon_'+ self._model+ '_'+'historical'+'_'+self._run+'_*.nc')
                    self._ssp585_files = glob.glob('/work/uc1275/u290372/data/etccdi_indices_*/'+ self._indicator+ '_mon_'+ self._model+'_'+'ssp585'+'_'+self._run+'_*.nc')
                SMRs += ['_'.join([fl.split('/')[-1].split('_')[i] for i in [3,2,4]]) for fl in self._scen_files 
                               if len(fl.split('_')) > 4]

            if self._indicator in ['cddETCCDI']:
                self._scen_files = glob.glob('/work/uc1275/u290372/data/etccdi_indices_*/' +self._indicator+'_yr_'+self._model+'_'+self._scenario+'_'+self._run+'_*.nc')
                self._hist_files = glob.glob('/work/uc1275/u290372/data/etccdi_indices_*/' +self._indicator+'_yr_'+self._model+'_'+'historical'+'_'+self._run+'_*.nc')
                self._ssp585_files = glob.glob('/work/uc1275/u290372/data/etccdi_indices_*/' +self._indicator+'_yr_'+self._model+'_'+'ssp585'+'_'+self._run+'_*.nc')
                SMRs += ['_'.join([fl.split('/')[-1].split('_')[i] for i in [3,2,4]]) for fl in self._scen_files 
                               if len(fl.split('_')) > 4]

        if self._source == 'mesmer':
            self._scen_files = glob.glob('/work/uc1275/u290372/data/mesmer/%s/%s/%s_mon_%s_%s_%s_g025.nc' \
                                         %(self._model, self._scenario, self._indicator, self._model, self._scenario, self._run))
            self._hist_files = glob.glob('/work/uc1275/u290372/data/mesmer/%s/%s/%s_mon_%s_%s_%s_g025.nc' \
                                         %(self._model, 'historical', self._indicator, self._model, 'historical', self._run))
            SMRs += ['_'.join([fl.split('/')[-1].split('_')[i] for i in [3,2,4]]) for fl in self._scen_files 
                               if len(fl.split('_')) > 4]
            
        return SMRs

    def regrid(self, var, fl):
        asd
        return var

    def treat_data(self, var, fl):
        # this is the main preprocessing
        # it has to be in a separate function because it is applied to each subfile
        if self._level is not None:
            for levName in ['lev','plev']:
                if levName in var.dims:
                    if self._verbose: print(levName,var[levName].attrs)
                    unit = var[levName].attrs['units']
                    if 'Pa' in unit:
                        if unit == 'hPa':
                            self._level /= 100
                        if self._verbose: print('available pressure levels in ',unit, var[levName].values)
                        var = var.sel({levName:self._level}, method='nearest')
                        if self._verbose: print('used pressure levels in ',unit, var[levName].values)
                    else:
                        return None
                else:
                    if self._verbose: print(var.dims)
        
        if self._regrid:
            if self._verbose: print('regridding')
            ds_regrid = xr.Dataset({
                'lon': (['lon'], np.arange(self._extent[0],self._extent[1]+self._regrid_prec, self._regrid_prec,'int')),
                'lat': (['lat'], np.arange(self._extent[2],self._extent[3]+self._regrid_prec, self._regrid_prec,'int')),})
            filename='/work/uc1275/u290372/data/regriding_weights/%s_%s_%s.nc' \
                                             %(self._model,self._realm,'x'.join([str(i) for i in self._extent]))            
            if os.path.isfile(filename) and self._overwrite == False:
                regridder = xe.Regridder(var, ds_regrid, 'bilinear', ignore_degenerate=True, periodic=True, weights=filename)
            else:
                regridder = xe.Regridder(var, ds_regrid, 'bilinear', ignore_degenerate=True, periodic=True)
                regridder.to_netcdf(filename)
            var = regridder(var).compute()
        
        if 'lat' not in var.dims and 'latitude' in var.dims:
            var = var.rename(dict(latitude='lat', longitude='lon'))
        if 'lat' not in var.dims and self._indicator != 'tas':
            if self._verbose: print('regridding')
            fls_tas = glob.glob('/'.join(fl.split('/')[:-5]) + '/' + '/'.join(['Amon','tas','*','*','*']))
            if len(fls_tas) > 0:
                nc_tas = xr.open_dataset(fls_tas[0])
                ds_regrid = xr.Dataset({
                    'lon': (['lon'], nc_tas.lon.values),'lat': (['lat'], nc_tas.lat.values),})
                regridder = xe.Regridder(var, ds_regrid, 'bilinear', ignore_degenerate=True, periodic=True,
                                         filename='/work/uc1275/u290372/data/regriding_weights/%s_%s.nc' %(self._model,self._realm))
                var = regridder(var).compute()
        if 'lat' not in var.dims:
            if self._verbose: print('no lat')
            return None
        
        if self._closest_grid is not None:
            var = var.sel(self._closest_grid, method='nearest')
            var = var.squeeze()
        
        if self._extent is not None and self._regrid==False:
            mask_lon = (var.lon >= self._extent[0]) & (var.lon <= self._extent[1])
            mask_lat = (var.lat >= self._extent[2]) & (var.lat <= self._extent[3])
            var = var.where(mask_lon & mask_lat, drop=True)
            
        if self._shape_for_mask is not None:
            mask = regionmask.mask_geopandas(self._shape_for_mask, var)
            mask.values[np.isfinite(mask.values)] = 1 
            iys,ixs = np.where(mask == 1)
            var.values = var.values * mask.values
            var = var.isel(lat=np.unique(iys)).isel(lon=np.unique(ixs))
            
        if self._mask is not None:
            mask = xr.load_dataset(self._mask_search_path %(self._model))[self._region_name]
            #print(len(mask.lat) != len(var.lat) or len(mask.lon) != len(var.lon))
            if len(mask.lat) != len(var.lat) or len(mask.lon) != len(var.lon):
                var = self.regrid(var,fl)
            mask.values[mask.values > 0] = 1
            mask.values[mask.values < 1] = np.nan
            iys,ixs = np.where(mask == 1)
            var.values = var.values * mask.values
            var = var.isel(lat=np.unique(iys)).isel(lon=np.unique(ixs))
            # for most models the folliwing lines worked just fine
            # for MPI not, no idea why
            # var = var.where(mask > 0, drop=True)
            
            
        # subselect months
        if self._months is not None:
            month_mask = np.isin(var.time.dt.month, self._months)
            month_mask = xr.DataArray(month_mask, coords=dict(time=var.time), dims=['time'])
            var = var.where(month_mask, drop=True)
                
        return var

    def get_projections(self):
        self.get_files()
        self._data = None
        
        # interrupt if mask is required and missing
        if self._mask is not None:
            if os.path.isfile(self._mask['search_path'] %(self._model)) == False:
                if self._verbose: print('mask missing for %s' %(self._model))
                return 1
            
        if self._verbose:
            print(self._scen_files)
        if len(self._scen_files) > 0:
            files = sorted(self._hist_files) + sorted(self._scen_files)

            # check whether there will be enough years
            if self._required_years is not None:
                years = np.array([])
                for fl in files:
                    years = np.append(years, np.arange(int(''.join(fl.split('_')[-1][:4])),int(''.join(fl.split('_')[-1].split('-')[-1][:4]))+1))
            
                if np.sum(np.isin(years,self._required_years)) < len(self._required_years):
                    if self._verbose: print('not enough years')
                    print('not enough years')
            
            l,years = [],[]
            for fl in files:
                if self._verbose: print(fl)
                var = xr.open_dataset(fl)[self._indicator]
                if var is not None:
                    yrs = np.array([int(str(t)[:4]) for t in var.time.values])
                    if self._cutoff_years is not None:
                        var = var[np.isin(yrs,np.arange(self._cutoff_years[0],self._cutoff_years[1]+1,1))]
                    if var.time.shape[0] > 0:
                        var = self.treat_data(var,fl)
                        if self._preprocessing_function is not None:
                            var = self._preprocessing_function(var)
                        if var is not None:
                            if np.sum(np.isin(yrs, np.array(years).flatten())) < len(yrs):
                                l.append(var)
                                years += [int(str(t)[:4]) for t in var.time.values]
                                
            if 'ssp' in self._scenario:
                # in ssp534-over the years 2015-2039 have to be taken from ssp585
                missing_years = np.array([yr for yr in np.arange(2015,2050,1,'int') if yr not in np.array(years)])
                if self._verbose: print('missing years', missing_years)
                if 2020 in missing_years and self._scenario == 'ssp534-over':
                    for fl in self._ssp585_files:
                        var = xr.open_dataset(fl)[self._indicator]
                        var_years = np.array([int(str(t)[:4]) for t in var.time.values])
                        var = var[np.isin(var_years,missing_years)]
                        if var.shape[0] > 0:
                            var = self.treat_data(var,fl)
                            l.append(var)
                            if self._verbose: print('adding years from ssp585',np.unique([int(str(t)[:4]) for t in var.time.values]))

                # this is needed because 2265 is not supported by np.datetime64 0_o
                time_formats = list(set([type(o.time.values[0]) for o in l]))
                if self._time_format is not None or len(time_formats) != 1:
                    if self._time_format is None:
                        self._time_format = [f for f in time_formats if f != np.datetime64][0]
                    for i,tmp in enumerate(l):
                        new_time = np.array([self._time_format(int(str(t)[:4]),int(str(t)[5:7]),int(str(t)[8:10])) for t in tmp.time.values])
                        l[i] =  tmp.assign_coords(time = new_time)
                                    

            if 'abrupt' in self._scenario or self._scenario == 'piControl':
                if len(l) > 0:
                    first_year = int(str(l[0].time.values[0])[:4])
                    for i,tmp in enumerate(l):
                        # there seem to be different standards for the time axis in this scenario
                        new_time = np.array([self._time_format(int(str(t)[:4]) -first_year, int(str(t)[5:7]), 15) 
                                             for t in tmp.time.values])
                        l[i] =  tmp.assign_coords(time = new_time)

            # subselect months
            
                        
            # concat all
            if len(l) > 0:
                out = xr.concat(l, dim='time', coords='minimal', compat='override')
                out = out.sortby('time')
                out = out.drop_duplicates('time', keep='first')
                #if self._required_years is not None:
                #    if np.sum(np.isin(np.unique(out.time.dt.year),self._required_years)) < len(self._required_years):
                #        if self._verbose: print('not enough years')
                #        return 1
                out = out.expand_dims(dim='SMR', axis=0)
                out = out.assign_coords(SMR=[self._SMR])
                self._data = out
                return 0 
        
        if self._verbose: print('something went wrong')
        return 1
    
    #def aggregateOverYear(self,months=range(1,13), aggregationFunction=np.mean):
    #    month_mask = np.isin(self._data.time.dt.month, months)
    #    month_mask = xr.DataArray(month_mask, coords=dict(time=self._data.time), dims=['time'])
    #    self._data = aggregationFunction(self._data.where(month_mask, drop=True).groupby('time.year'))
    #    self._history += 'months%s' %('-'.join([str(m) for m in sorted(months)]))

    