#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Wed May 26 14:30:07 2021
# @author: Xiaodong Ming

"""
remove_block
===========
To do:
    remove overhead buildings on DEM
Input:
    1. DEM file (map unit:m)
    2. shape file marking overhead building with a quadrangle
    
Example

# shape_file = '/Users/ming/Dropbox/Python/Data/overhead_buildings.shp'
# dem_file = '/Users/ming/Dropbox/Python/RiverChannel/CA1_5m.tif'
shape_file = 'file:///Users/ming/OneDrive - Loughborough University/Tyneside/SHP/OverheadBuildings.shp'
dem_file = '/Users/ming/OneDrive - Loughborough University/Tyneside/DSM_DTM_BM_merged.tif'
dem_new, channel_dict_list = remove_overhead_buildings(dem_file, shape_file)

#%%
dem_obj = Raster(dem_file)
shape_polygon = fiona.open(shape_file)

#%% for one feature
one_feature = shape_polygon[0]
line_xy = one_feature['geometry']['coordinates'][0]
line_xy = np.array(line_xy)
bound = get_polygon_bound(line_xy, expand_edge=dem_obj.cellsize*2)
dem_clip = dem_obj.rect_clip(bound)
channel_dict = construct_river_section(dem_clip, line_xy)
if 'adjust_val' in one_feature['properties'].keys():
    adjust_val = one_feature['properties']['adjust_val']
else:
    adjust_val = None
dem_new = get_new_z(dem_clip, channel_dict, adjust_val)

#%% 
fig, ax = dem_obj.hillshade()
for channel_dict in channel_dict_list:
    pts_coords = channel_dict['polygon']
    ax.plot(pts_coords[:, 0], pts_coords[:, 1], 'r') 
# ax.plot(channel_dict['bank0'][:, 0], channel_dict['bank0'][:, 1], '*')
# ax.plot(channel_dict['bank1'][:, 0], channel_dict['bank1'][:, 1], '*') 
# ax.plot(channel_dict['cross0'][:, 0], channel_dict['cross0'][:, 1], 's') 
# ax.plot(channel_dict['cross1'][:, 0], channel_dict['cross1'][:, 1], 's') 
#%%
pts_coords = shape_polygon[0]['geometry']['coordinates'][0]
pts_coords = np.array(pts_coords)
#%%
pts_coords = channel_dict['polygon']
fig, ax = dem_clip.mapshow()
ax.plot(pts_coords[:, 0], pts_coords[:, 1], 'r*')    

"""

#%% load files
import fiona
import copy
import numpy as np
from . import Raster
from numpy.linalg import norm
from .channel_geometry import point2segment, remove_duplicate_rows
from .channel_geometry import distance_p2l, in_channel
from .convert_coords import xy2bc
#%
def remove_overhead_buildings(dem_obj, shape_polygon):
    """ remove overhead buildings on DEM file
    
    Parameters
    ----------
    dem_obj : a Raster object for DEM
    shape_polygon : a Collection object of fiona module 
        or a string for polygon shapefile.
        Each polygon must consists of four points, namely a quadrangle.
        If the polygon has property 'adjust_val', then all the cell inside the
        polygon will be adjusted according to the given value
    
    Returns
    -------
    dem_new : a Raster object with rectified DEM

    """
    if type(dem_obj) is str:
        dem_obj = Raster(dem_obj)
    dem_obj = copy.deepcopy(dem_obj)
    if type(shape_polygon) is str:
        shape_polygon = fiona.open(shape_polygon)
    # rasterize shapefile
    # ind_array = dem_obj.rasterize(shape_polygon)
    channel_dict_list = []
    N = 0
    for one_feature in shape_polygon:
        if one_feature['geometry'] is None:
            continue
        line_xy = one_feature['geometry']['coordinates'][0]
        line_xy = np.array(line_xy)
        bound = get_polygon_bound(line_xy, expand_edge=dem_obj.cellsize*2)
        dem_clip = dem_obj.rect_clip(bound)
        channel_dict = construct_river_section(dem_clip, line_xy)
        if 'adjust_val' in one_feature['properties'].keys():
            adjust_val = one_feature['properties']['adjust_val']
        else:
            adjust_val = None
        dem_new = get_new_z(dem_clip, channel_dict, adjust_val)
        dem_new.paste_on(dem_obj)
        channel_dict_list.append(channel_dict)
        print(N)
        N = N+1
    return dem_obj, channel_dict_list

def get_polygon_bound(polygon_xy, expand_edge=None):
    """ get (left, right, bottom, top) of a polygon feature
    """
    left = polygon_xy[:, 0].min()
    right = polygon_xy[:, 0].max()
    bottom = polygon_xy[:, 1].min()
    top = polygon_xy[:, 1].max()
    if expand_edge is not None:
        bound = (left-expand_edge, right+expand_edge, 
                 bottom-expand_edge, top+expand_edge)
    else:
        bound = (left, right, bottom, top)
    return bound

def get_cells_on_line1(dem_obj, segment):
    """ find cells across a line
    the distance between the cell centre and the land is no larger than half
    cellsize
    dem_obj: a Raster object for DEM
    segment: a 2X2 array providing the coords of two points
    return 

    """
    cells_x, cells_y = dem_obj.to_points()
    cells_xy = np.c_[cells_x.flatten(), cells_y.flatten()]
    # find ending cells of the line section
    d0 = norm(cells_xy - segment[0], axis=1)
    d1 = norm(cells_xy - segment[1], axis=1)
    ind_ends = np.logical_or(d0 <= d0.min(), d1 <= d1.min())
    # exclude cells far from the line section
    d2 = distance_p2l(np.array(segment[0:2]), cells_xy)
    _, ind_pos = point2segment(cells_xy, np.array(segment[0:2]))
    ind_inline = np.logical_and(d2 <= dem_obj.cellsize/2, ind_pos)
    ind_online = np.logical_or(ind_ends, ind_inline)
    cell_xyz = np.c_[cells_xy[ind_online], 
                     dem_obj.array.flatten()[ind_online]]
    # sort points
    d3 = norm(cells_xy[ind_online]-segment[0], axis=1)
    ind = d3.argsort()
    cell_xyz = cell_xyz[ind]

    return cell_xyz

def get_cells_on_line(dem_clip, segment):
    """ find cells across a line
    the distance between the cell centre and the land is no larger than half
    cellsize
    dem_clip: a Raster object for DEM
    segment: a 2X2 array providing the coords of two points
    return 

    """
    one_line = {"type": "LineString", 
                "coordinates": [segment[0], segment[1]]}
    shape_lines = [one_line]
    ind_array = dem_clip.rasterize(shape_lines)
    cells_x, cells_y = dem_clip.to_points()
    cell_xyz = np.c_[cells_x[ind_array], cells_y[ind_array],
                     dem_clip.array[ind_array]]
    d = norm(cell_xyz[:, :2]-segment[0], axis=1)
    ind = d.argsort()
    cell_xyz = cell_xyz[ind]
    return cell_xyz

def construct_river_section(dem_clip, polygon_xy):
    """ construct a river channel based on polygon points (4) 
    
    Parameters
    ----------
    dem_clip : a Raster object for DEM
    polygon_xy : TYPE

    Returns
    -------
    channel_dict : dict with five keys, showing the channel line points 

    """
    polygon_xy = np.array(polygon_xy)
    polygon_xy = remove_duplicate_rows(polygon_xy)
    if polygon_xy.shape[0] > 4: # if the polygon is not quadrangle
        channel_dict = {'polygon':polygon_xy}
    else:
        cl0 = polygon_xy[[0, 1]]
        cl0_xyz = get_cells_on_line(dem_clip, cl0)
        cl1 = polygon_xy[[3, 2]]
        cl1_xyz = get_cells_on_line(dem_clip, cl1)
        bl0 = polygon_xy[[0, 3]]
        bl0_xyz = get_cells_on_line(dem_clip, bl0)
        bl1 = polygon_xy[[1, 2]]
        bl1_xyz = get_cells_on_line(dem_clip, bl1)
        # identify crossline and bankline according to the median z value
        # crossline should have a lower median z value
        z_c = np.concatenate([cl0_xyz[:,2], cl1_xyz[:,2]])
        z_b = np.concatenate([bl0_xyz[:,2], bl1_xyz[:,2]])
        if np.median(z_c) < np.median(z_b):
            bline0 = bl0_xyz[:, :2]
            bline1 = bl1_xyz[:, :2]
            cline0 = cl0_xyz
            cline1 = cl1_xyz
        else:
            bline0 = cl0_xyz[:, :2]
            bline1 = cl1_xyz[:, :2]
            cline0 = bl0_xyz
            cline1 = bl1_xyz
        # construct river section    
        all_points = np.r_[bline0, cline1[:, :2], np.flipud(bline1),
                               np.flipud(cline0[:, :2])]
        channel_dict = {'bank0':bline0, 'bank1':bline1,
                        'cross0':cline0, 'cross1':cline1,
                        'polygon':all_points}
    return channel_dict

def get_new_z(dem_obj, channel_dict, adjust_val):
    cells_x, cells_y = dem_obj.to_points()
    cells_xy = np.c_[cells_x.flatten(), 
                     cells_y.flatten()]
    cells_z = dem_obj.array.flatten()
    ind = in_channel(channel_dict, cells_xy, radius=0.00001)
    xy_cells_vec = cells_xy[ind]
    z_cells_vec = cells_z[ind]
    if adjust_val is None:
        bs, cs, zs = xy2bc(xy_cells_vec, channel_dict, 
                           max_error=dem_obj.cellsize*2)
        z_cells_vec[~np.isnan(zs)] = zs[~np.isnan(zs)]
    else:
        z_cells_vec = z_cells_vec+adjust_val
    cells_z[ind] = z_cells_vec
    dem_obj.array = np.reshape(cells_z, dem_obj.array.shape)
    return dem_obj

def main():
    print('Package to show grid data')

if __name__=='__main__':
    main()
