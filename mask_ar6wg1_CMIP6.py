#!/usr/bin/env python
# coding: utf-8
import sys,os,glob
import xarray as xr
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.io.shapereader as shapereader
from shapely.geometry import mapping, Polygon, MultiPolygon, asShape, Point
from shapely.ops import cascaded_union, unary_union

import argparse

parser = argparse.ArgumentParser(description='average scenario over region')
parser.add_argument('-m', '--model')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('-a', '--additional_mask', default='none')
args = vars(parser.parse_args())
print(args)

model = args['model']

shapefile=shapereader.Reader('/work/uc1275/u290372/data/shapefiles/ar6wg1/IPCC-WGI-reference-regions-v4.shp').records()
ar6wg1_polygons = {}
for item in shapefile:
    shape,region=item.geometry,item.attributes
    name=u''+region['Acronym']
    try:
        tmp=MultiPolygon(shape)
    except:
        tmp=Polygon(shape)
    if name not in ar6wg1_polygons.keys():
        ar6wg1_polygons[name] = tmp
    else:
        ar6wg1_polygons[name] = cascaded_union((ar6wg1_polygons[name],tmp))



print(model)
outFile = '/work/uc1275/u290372/data/masks/CMIP6_ar6wg1/ar6wg1_%s_%s_overlap.nc' %(model,args['additional_mask'])
if os.path.isfile(outFile) == False or args['overwrite']:
    fls = glob.glob('/pool/data/CMIP6/data/*/*/%s/*/*/fx/sftof/*/*/*.nc' %(model))
    fls += glob.glob('/pool/data/CMIP6/data/*/*/%s/*/*/fx/sftlf/*/*/*.nc' %(model))
    sft = fls[0].split('/')[-4]
    land_ocean = xr.load_dataset(fls[0])[sft]
    if land_ocean.attrs['units'] == '%':
        land_ocean /= 100.
    if sft == 'sftof':
        land_ocean = -1 * (land_ocean -1)
    
    lat = land_ocean.lat
    lon = land_ocean.lon
        
    # loop over the grid to get grid polygons
    grid_polygons = xr.DataArray(np.empty((len(lat),len(lon)),dtype=Polygon),
                                coords={'lat':lat,'lon':lon}, 
                                dims=['lat','lon'])
    dx = np.abs(np.diff(lon,1))[0]/2.
    dy = np.abs(np.diff(lat,1))[0]/2.

    for x in lon:
        x1 = x-dx
        x2 = x+dx
        if lon.max() > 180 and x-dx > 180 :
            x1 = x-dx -360
            x2 = x+dx -360
        if lon.max() > 180 and x-dx < 180 and x+dx > 180:
            x1 = x-dx
            x2 = 180
        for y in lat:
            y1 = y-dy
            y2 = y+dy
            grid_polygons.loc[y,x] = Polygon([(x1,y1),(x1,y2),(x2,y2),(x2,y1)])

    ar6wg1_overlap = {}
    for region_name in ar6wg1_polygons.keys():
        #print(region_name)
        poly = ar6wg1_polygons[region_name]
    
        overlap = xr.DataArray(coords={'lat':lat,'lon':lon}, dims=['lat','lon'])

        # only use land or ocean cells
        if args['additional_mask'] == 'ocean':
            ys,xs = np.where(land_ocean.values < 0.5)
        if args['additional_mask'] == 'land':
            ys,xs = np.where(land_ocean.values > 0.5)
        if args['additional_mask'] == 'none':
            ys,xs = np.where(np.isfinite(land_ocean.values))

        for x,y in zip(lon[xs].values,lat[ys].values):
            overlap.loc[y,x] = grid_polygons.loc[[y],[x]].values[0,0].intersection(poly).area / \
                               grid_polygons.loc[[y],[x]].values[0,0].area

        if np.sum(overlap) > 0:
            overlap.values[overlap==0]=np.nan
        
        ar6wg1_overlap[region_name] = overlap
        xr.Dataset(ar6wg1_overlap).to_netcdf(outFile)

        if region_name == 'WAF':
            fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5),
                              subplot_kw={'projection': cartopy.crs.PlateCarree()})
            im = ax.pcolormesh(overlap.lon, overlap.lat, overlap.values, shading='auto',
                                   cmap='plasma', transform=cartopy.crs.PlateCarree(), zorder=2)
            plt.savefig(outFile.replace('.nc','_%s.png' %(region_name)))

