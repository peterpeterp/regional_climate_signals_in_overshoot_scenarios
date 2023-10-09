import sys,glob,datetime,os,gc,importlib,pickle,cftime
import xarray as xr
import numpy as np

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

import cartopy
sys.path.append('/home/u/u290372/scripts/')
from cartopy_features import *
import cartopy.io.shapereader as shapereader
from shapely.geometry import mapping, Polygon, MultiPolygon, asShape, Point
from shapely.ops import unary_union

shapefile=shapereader.Reader('/work/uc1275/u290372/data/shapefiles/ar6wg1/IPCC-WGI-reference-regions-v4.shp').records()
polygons = {}
for item in shapefile:
    shape,region=item.geometry,item.attributes
    name=u''+region['Acronym']
    try:
        tmp=MultiPolygon(shape)
    except:
        tmp=Polygon(shape)
    if name not in polygons.keys():
        polygons[name] = tmp
    else:
        polygons[name] = unary_union((polygons[name],tmp))
        
        
centroids = {}
for region,polygon in polygons.items():
    centroids[region] = list(np.array(polygons[region].centroid).flatten())
centroids['NPO'][0] = -150
centroids['EPO'][0] = -150
centroids['SPO'][0] = -150
centroids['ESAF'][1] += 4
centroids['MDG'][1] -= 4
centroids['ARO'][1] -= 5
centroids['ARO'][0] += 8

ar6wg1_d = dict(ocean = ['ARO', 'ARS', 'BOB', 'EAO', 'EIO',
                         'EPO', 'NAO', 'NPO', 'SAO',
                         'SIO', 'SOO', 'SPO'],
                land = ['ARP', 'CAF', 'CAU', 'CEU', 'CNA', 'EAN',
                        'EAS', 'EAU', 'ECA', 'EEU', 'ENA', 'ESAF',
                        'ESB', 'GIC', 'MDG', 'NAU', 'NCA', 'NEAF',
                        'NEN', 'NES', 'NEU', 'NSA', 'NWN', 'NWS', 'NZ',
                        'RAR', 'RFE', 'SAH', 'SAM', 'SAS', 'SAU',
                        'SCA', 'SEAF', 'SES', 'SSA', 'SWS',
                        'TIB', 'WAF', 'WAN', 'WCA', 'WCE', 'WNA',
                        'WSAF', 'WSB', 'EAN', 'WAN'],
                mix = ['CAR', 'SEA', 'MED'],
                )

#ar6wg1_reg_info = {}
#for region in centroids.keys():
#    ar6wg1_reg_info[region] = {'color':xx,'pos_off':(0,0)}
    
l1 = '#b38e69'
l2 = '#522a02'
l3 = '#805225'
l4 = '#b88070'
l4 = '#E9C6BB'

o1 = '#8ba9e8'
o2 = '#142d61'
o3 = '#3e5075'
    
ar6wg1_reg_info = {'GIC': {'color': l1, 'pos_off': (0, 0)},
                 'NWN': {'color': l2, 'pos_off': (0, 0)},
                 'NEN': {'color': l3, 'pos_off': (0, 0)},
                 'WNA': {'color': l3, 'pos_off': (0, 0)},
                 'CNA': {'color': l1, 'pos_off': (0, +5)},
                 'ENA': {'color': l2, 'pos_off': (0, 0)},
                 'NCA': {'color': l2, 'pos_off': (0, 0)},
                 'SCA': {'color': l1, 'pos_off': (0, 0)},
                 'CAR': {'color': l3, 'pos_off': (0, 0)},
                   
                 'NWS': {'color': l2, 'pos_off': (-5, 0)},
                 'NSA': {'color': l1, 'pos_off': (0, 0)},
                 'NES': {'color': l4, 'pos_off': (0, 0)},
                 'SAM': {'color': l3, 'pos_off': (0, -3)},
                 'SWS': {'color': l1, 'pos_off': (0, 0)},
                 'SES': {'color': l2, 'pos_off': (+5, -5)},
                 'SSA': {'color': l3, 'pos_off': (0, 0)},
                   
                 'NEU': {'color': l3, 'pos_off': (0, 0)},
                 'WCE': {'color': l2, 'pos_off': (+5, 0)},
                 'EEU': {'color': l1, 'pos_off': (0, 0)},
                 'RAR': {'color': l4, 'pos_off': (0, 0)},
                 'WSB': {'color': l2, 'pos_off': (0, 0)},
                 'ESB': {'color': l3, 'pos_off': (+10, +5)},
                 'RFE': {'color': l1, 'pos_off': (0, 0)},
                 'WCA': {'color': l4, 'pos_off': (+5, 0)},
                 'ECA': {'color': l1, 'pos_off': (+10, 0)},
                 'TIB': {'color': l3, 'pos_off': (0, +3)},
                 'EAS': {'color': l2, 'pos_off': (0, 0)},
                 'SAS': {'color': l1, 'pos_off': (-2, -3)},
                 'SEA': {'color': l3, 'pos_off': (0, 0)},
                   
                 'MED': {'color': l3, 'pos_off': (-10, 0)},
                 'SAH': {'color': l1, 'pos_off': (+5, 0)},
                 'WAF': {'color': l3, 'pos_off': (0, 0)},
                 'CAF': {'color': l2, 'pos_off': (0, 0)},
                 'NEAF': {'color': l3, 'pos_off': (+3, 0)},
                 'SEAF': {'color': l1, 'pos_off': (-3, -5)},
                 'WSAF': {'color': l3, 'pos_off': (-5, 0)},
                 'ESAF': {'color': l4, 'pos_off': (-2, -5)},
                 'MDG': {'color': l2, 'pos_off': (+3, 0)},
                 'ARP': {'color': l2, 'pos_off': (0, +3)},

                 'NAU': {'color': l1, 'pos_off': (+3, +5)},
                 'CAU': {'color': l2, 'pos_off': (-10, 0)},
                 'EAU': {'color': l4, 'pos_off': (0, 0)},
                 'SAU': {'color': l1, 'pos_off': (0, -5)},
                 'NZ': {'color': l3, 'pos_off': (0, 0)},
                 
                 'EAN': {'color': l1, 'pos_off': (0, 0)},
                 'WAN': {'color': l2, 'pos_off': (0, 0)},
                   
                 'ARO': {'color': o1, 'pos_off': (+20, -5)},
                 'NPO': {'color': o2, 'pos_off': (-120, 0)},
                 'EPO': {'color': o1, 'pos_off': (-90, 0)},
                 'SPO': {'color': o3, 'pos_off': (-30, 0)},
                 'NAO': {'color': o2, 'pos_off': (0, 0)},
                 'EAO': {'color': o1, 'pos_off': (0, 0)},
                 'SAO': {'color': o3, 'pos_off': (0, 0)},
                 'ARS': {'color': o2, 'pos_off': (-3, 0)},
                 'BOB': {'color': o3, 'pos_off': (+3, -5)},
                 'EIO': {'color': o1, 'pos_off': (0, -3)},
                 'SIO': {'color': o3, 'pos_off': (0, 0)},
                 'SOO': {'color': o2, 'pos_off': (0, 0)}
                  }

for k,v in ar6wg1_reg_info.items():
    v['edge'] = 'k'

    
def add_land_ocean_flag(region_name):
    if region_name in ar6wg1_d['land']+ar6wg1_d['mix']:
        return region_name + '-land'
    elif region_name in ar6wg1_d['ocean']:
        return region_name + '-ocean'

    
    
    
    
if __name__ == '__main__':
    plt.close()
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5), subplot_kw={'projection': cartopy.crs.Robinson()})
    for region in ar6wg1_d['land']:
        ax.add_geometries([polygons[region]], color='green', alpha=0.2, crs=cartopy.crs.PlateCarree())
    for region in ar6wg1_d['ocean']:
        ax.add_geometries([polygons[region]], color='blue', alpha=0.2, crs=cartopy.crs.PlateCarree())
    for region in ar6wg1_d['mix']:
        ax.add_geometries([polygons[region]], color='m', alpha=0.2, crs=cartopy.crs.PlateCarree())
    for region,centroid in centroids.items():
        ax.annotate(region, xy=centroid, xycoords=cartopy.crs.PlateCarree()._as_mpl_transform(ax), ha='center', va='center')
    plt.savefig('/work/uc1275/u290372/data/shapefiles/ar6wg1/regions.png', bboxes='tight', dpi=300)


