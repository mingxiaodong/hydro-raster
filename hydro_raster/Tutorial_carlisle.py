#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Mar 18 16:14:45 2021
# @author: Xiaodong Ming

"""
Tutorial_carlisle
===========
To do:
    use the Eden river cross sections in Carlisle to generate bed terrain

"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from convert_coords import cross_section2grid_elevation
from hipims_io import Raster
#% load mat data
mat_fname = 'CarlisleRiverChannelData.mat'
mat_contents = sio.loadmat(mat_fname)
channelBoundaryLines = mat_contents['channelBoundaryLine']
crossSectionLines = mat_contents['crossSectionLine']
channelBoundary = mat_contents['channelBoundary']
#% Get bankline points
points_x = channelBoundaryLines[0]['X'][0]
points_y = channelBoundaryLines[0]['Y'][0]
bankline0 = np.c_[points_x.flatten(), points_y.flatten()]
bankline0 = bankline0[~np.isnan(bankline0[:, 0])]

points_x = channelBoundaryLines[1]['X'][0]
points_y = channelBoundaryLines[1]['Y'][0]
bankline1 = np.c_[points_x.flatten(), points_y.flatten()]
bankline1 = bankline1[~np.isnan(bankline1[:, 0])]

#% Get crossline points
crossline_list = [a[0] for a in crossSectionLines]

#%% show banklines and cross section lines with directions of their vertex
fig, ax = plt.subplots(1)
ax.plot(bankline0[:, 0], bankline0[:, 1], 'r')
ax.plot(bankline1[:, 0], bankline1[:, 1], 'r')
for n in np.arange(len(crossline_list)):
    line_data = crossline_list[n]
    ax.plot(line_data[:, 0], line_data[:, 1], marker='.')
    ax.annotate(str(n), xy=line_data[0][:2])
ax.set_aspect('equal', adjustable='box')
#%% create terrain data from cross section lines and bank lines
array, header = cross_section2grid_elevation(bankline0, bankline1, 
                                             crossline_list, cellsize=2,
                                             max_error=0.1)
#%% show interpolated river topography with source lines
obj_dem = Raster(array=array, header=header)
fig, ax = obj_dem.mapshow()
ax.plot(bankline0[:, 0], bankline0[:, 1], 'r')
ax.plot(bankline1[:, 0], bankline1[:, 1], 'r')
for n in np.arange(len(crossline_list)):
    line_data = crossline_list[n]
    # ax.plot(line_data[:, 0], line_data[:, 1], 'k-')
#%% show preprocessed data: sorted and trimmed lines
from demo import draw_arrow
from channel_geometry import preprocessing_data
bl0, bl1, cl_sort = preprocessing_data(bankline0, bankline1, crossline_list)
fig, ax = plt.subplots(1)
ax.plot(bl0[:, 0], bl0[:, 1], 'r')
draw_arrow(ax, bankline0, width=0.2)
ax.plot(bl1[:, 0], bl1[:, 1], 'r')
draw_arrow(ax, bankline1, width=0.2)
for n in np.arange(len(cl_sort)):
    line_data = cl_sort[n]
    ax.plot(line_data[:, 0], line_data[:, 1], 'b:')
    draw_arrow(ax, line_data, width=0.2)
    ax.annotate(str(n), xy=line_data[0][:2])
ax.set_aspect('equal', adjustable='box')

#%% Show splitted river channels
from channel_geometry import split_channel
from channel_geometry import break_bankline, discretize_river_section
from channel_geometry import match_break_points_by_relative_pos
# from channel_geometry import match_break_points
river_sections = split_channel(cl_sort, bl0, bl1)
max_error = 2
fig, ax = plt.subplots(1)
for n in [19, 20, 21, 22]: #np.arange(len(river_sections)): 
    bl0_sec = river_sections[n]['bank0']
    bl1_sec = river_sections[n]['bank1']
    cl0 = river_sections[n]['cross0']
    cl1 = river_sections[n]['cross1']
    _, bl0_broken = break_bankline(bl0_sec, max_error)
    _, bl1_broken = break_bankline(bl1_sec, max_error)
    bl0_broken_e, bl1_broken_e = match_break_points_by_relative_pos(bl0_broken, 
                                                                    bl1_broken)
    # bl0_broken_e, bl1_broken_e = match_break_points(bl0_broken, bl1_broken)
    river_polys = discretize_river_section(bl0_broken_e, bl1_broken_e)
    ax.plot(bl0_broken[:, 0], bl0_broken[:, 1], '--',
                marker='o', mfc='None') #, c='b'
    ax.plot(bl1_broken[:, 0], bl1_broken[:, 1], '--', 
                marker='o', mfc='None') #, c='b'
    ax.plot(bl0_broken_e[:, 0], bl0_broken_e[:, 1], '--',
                marker='*', mfc='None') #, c='b'
    ax.plot(bl1_broken_e[:, 0], bl1_broken_e[:, 1], '--', 
                marker='*', mfc='None') #, c='b'
    ax.plot(cl0[:, 0], cl0[:, 1], 'k-', linewidth=0.5) #, c='b'
    ax.plot(cl1[:, 0], cl1[:, 1], 'k-', linewidth=0.5) #, c='b'
    
    # xy_anno = np.median(river_sections[n]['polygon'], axis=0)
    # ax.annotate(str(n), xy=xy_anno)
    # for i in np.arange(bl1_broken.shape[0]):
    #     xy_p = bl1_broken[i]
    #     ax.annotate(str(i), xy=xy_p)   
    #%
    for geom in river_polys.geoms:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=0.5)
ax.set_aspect('equal', adjustable='box')
fname = 'split_channel_err'+str(max_error)+'m_'+'.png'
# fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=2000)
#%% Show relative bc coordinates of river points
from channel_geometry import get_bounding_box
from convert_coords import interp_one_reach, generate_cells_xy
fig, ax = plt.subplots(1)

cellsize = 5
for n in [17, 18, 19, 20, 21, 22, 23, 24, 25]:
    one_reach = river_sections[n]
    bounding_box = get_bounding_box(one_reach['polygon'])
    x_grid, y_grid, header = generate_cells_xy(bounding_box, cellsize)
    bs, cs, zs = interp_one_reach(one_reach, x_grid, y_grid, 
                                  max_error=0.1)
    ax.plot(one_reach['polygon'][:, 0], one_reach['polygon'][:, 1], 'k-')
    ax.contour(x_grid, y_grid, bs, 10, cmap='winter')
    ax.contour(x_grid, y_grid, cs, 10, cmap='autumn')
ax.set_aspect('equal', adjustable='box')