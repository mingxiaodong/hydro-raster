#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Mon Dec  7 12:55:34 2020
# @author: Xiaodong Ming

"""
tutorial
===========
To do:
    [objective of this script]

"""
#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from convert_coords import cross_section2grid_elevation
from demo_functions import draw_arrow

# load and show input bankline and cross-section lines
with open('sample/sample_data.pkl', 'rb') as f:
    sample_data = pickle.load(f)
bankline0 = sample_data['bankline0']
bankline1 = sample_data['bankline1']
crossline_list = sample_data['crosslines']
#%% Composite function to convert cross lines to DEM grid cells
z_grid, header = cross_section2grid_elevation(bankline0, bankline1, 
                                              crossline_list, 
                                              cellsize=0.2, max_error=0.1)
#%% show results with bank lines and cross lines
import hipims_io as hp
dem_obj = hp.Raster(array=z_grid, header=header)
fig, ax = dem_obj.hillshade()
ax.plot(bankline0[:, 0], bankline0[:, 1], 'r')
ax.plot(bankline1[:, 0], bankline1[:, 1], 'r')
for n in np.arange(len(crossline_list)):
    line_data = crossline_list[n]
    ax.plot(line_data[:, 0], line_data[:, 1], marker='.')

#%% Step-by-step tutorial
import convert_coords as cc
import channel_geometry as cg
#%% show banklines and cross section lines with directions of their vertex
fig1, axs = plt.subplots(2, 1)
ax1 = axs[0]
ax2 = axs[1]
ax1.plot(bankline0[:, 0], bankline0[:, 1], 'r')
draw_arrow(ax1, bankline0, width=0.2)
ax1.plot(bankline1[:, 0], bankline1[:, 1], 'r')
draw_arrow(ax1, bankline1, width=0.2)
for n in np.arange(len(crossline_list)):
    line_data = crossline_list[n]
    ax1.plot(line_data[:, 0], line_data[:, 1], marker='.')
    draw_arrow(ax1, line_data, width=0.2)
    ax1.annotate(str(n), xy=line_data[0][:2])
    xy1d = cc.normalise_xy_coords(line_data[:, 0], line_data[:, 1])
    ax2.plot(xy1d, line_data[:, 2])
ax1.set_aspect('equal', adjustable='box')

#%% Sort bankline points to the same direction, trim and sort cross lines
bl0, bl1, cl_list = cg.preprocessing_data(bankline0, bankline1, crossline_list)
fig1, ax1 = plt.subplots(1)
ax1.plot(bl0[:, 0], bl0[:, 1], 'r')
draw_arrow(ax1, bl0, width=0.2)
ax1.plot(bl1[:, 0], bl1[:, 1], 'r')
draw_arrow(ax1, bl1, width=0.2)
for n in np.arange(len(cl_list)):
    line_data = cl_list[n]
    if line_data.size>0:
        ax1.plot(line_data[:, 0], line_data[:, 1], marker='.')
        draw_arrow(ax1, line_data, width=0.2)
        ax1.annotate(str(n), xy=line_data[0][:2])
ax1.set_aspect('equal', adjustable='box')

#%% split river polygon using crosslines
channel_list = cc.split_channel(cl_list, bl0, bl1)
fig, ax = plt.subplots(1)
n=0
for channel_dict in channel_list:
    ax.plot(channel_dict['bank0'][:, 0], channel_dict['bank0'][:, 1], 
            marker='o')
    draw_arrow(ax, channel_dict['bank0'], width=0.2)
    ax.plot(channel_dict['bank1'][:, 0], channel_dict['bank1'][:, 1], 
            marker='o')
    draw_arrow(ax, channel_dict['bank1'], width=0.2)
    ax.plot(channel_dict['cross0'][:, 0], channel_dict['cross0'][:, 1], 
            marker='.')
    draw_arrow(ax, channel_dict['cross0'], width=0.2)
    ax.annotate(str(n), xy=channel_dict['cross0'][0][:2])
    n=n+1
    ax.plot(channel_dict['cross1'][:, 0], channel_dict['cross1'][:, 1], 
            marker='.')
    draw_arrow(ax, channel_dict['cross1'], width=0.2)
ax.set_aspect('equal', adjustable='box')
#%% for each river channel section
max_error = 0.2
n = 0
channel_dict = channel_list[n]
bline0 = channel_dict['bank0']
bline1 = channel_dict['bank1']
# break bank line
_, break_points_0 = cg.break_bankline(bline0, max_error)
_, break_points_1 = cg.break_bankline(bline1, max_error)
fig, ax = plt.subplots(1)
ax.plot(bline0[:, 0], bline0[:, 1])
ax.plot(break_points_0[:, 0], break_points_0[:, 1], marker='o')
ax.plot(bline1[:, 0], bline1[:, 1])
ax.plot(break_points_1[:, 0], break_points_1[:, 1], marker='s')
#%% project the bank break points to the opposite bank
break_points_exp_0 = cc.expand_break_points(break_points_0,
                                             break_points_1[1:-1])
break_points_exp_1 = cc.expand_break_points(break_points_1,
                                             break_points_0[1:-1])
fig, ax = plt.subplots(1)

ax.plot(break_points_exp_0[:, 0], break_points_exp_0[:, 1], marker='o',
           mfc='none')
ax.scatter(break_points_0[:, 0], break_points_0[:, 1], marker='*')
ax.plot(break_points_exp_1[:, 0], break_points_exp_1[:, 1], marker='s',
           mfc='none')
ax.scatter(break_points_1[:, 0], break_points_1[:, 1], marker='*')

#%% discretize river channel to polygons
river_polys, breaks_0, breaks_1 = cc.discretize_river_channel(bline0, 
                                            bline1, max_error)
for geom in river_polys.geoms:
        xs, ys = geom.exterior.xy
        plt.fill(xs, ys, alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')
#%% create a river topography grid
grid_header = sample_data['header']
cellsize = grid_header['cellsize']
nrows = grid_header['nrows']
ncols = grid_header['ncols']
x00 = grid_header['xllcorner']+cellsize/2
y00 = grid_header['yllcorner']+nrows*cellsize-cellsize/2
x_vect = np.linspace(x00, x00+cellsize*ncols, num=ncols, endpoint=False)
y_vect = np.linspace(y00, y00-cellsize*nrows, num=nrows, endpoint=False)
x_grid, y_grid = np.meshgrid(x_vect, y_vect)
#%% show relative coordinates of grid cells
xy_cells_vec = np.c_[x_grid.flatten(), y_grid.flatten()]
bc_pos = cc.convert_coords_2d(xy_cells_vec, bline0, bline1, max_error)
fig, ax = plt.subplots(1)
points_b = bc_pos[:,0]
points_c = bc_pos[:,1]
points_c[points_c>0.5] = 1-points_c[points_c>0.5]
points_z = np.round(points_c*10)
z_grid = points_z.reshape(x_grid.shape)
ax.contourf(x_grid, y_grid, z_grid, 10, cmap='YlGnBu', alpha=1)
points_z = np.round(points_b*20)
z_grid = points_z.reshape(x_grid.shape)
ax.contourf(x_grid, y_grid, z_grid, 20, cmap='Reds', alpha=0.5)
ax.set_aspect('equal', adjustable='box')

#%%
bc_pos = cc.convert_coords_2d(xy_cells_vec, bline0, bline1, max_error)
bs = bc_pos[:, 0]
grid_b = bs.reshape(x_grid.shape)
cs = bc_pos[:, 1]
grid_c = cs.reshape(x_grid.shape)
cline0 = channel_dict['cross0']
cline1 = channel_dict['cross1']
zs = cc.bc_interp_2d(bs, cs, cline0, cline1)
z_grid = zs.reshape(x_grid.shape)

#%
# from mpl_toolkits.mplot3d import Axes3D
bline0_z = np.linspace(cline0[0, 2], cline1[0, 2], num=bline0.shape[0], 
                       endpoint=False)
bline1_z = np.linspace(cline0[-1, 2], cline1[-1, 2], num=bline1.shape[0], 
                       endpoint=False)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(bline0[:, 0], bline0[:, 1], bline0_z, 'r')
ax.plot(bline1[:, 0], bline1[:, 1], bline1_z, 'b')
ax.plot(cline0[:, 0], cline0[:, 1], cline0[:, 2], '-.')
ax.plot(cline1[:, 0], cline1[:, 1], cline1[:, 2], '--')
xs = xy_cells_vec[:, 0]
ys = xy_cells_vec[:, 1]
ax.scatter(xs, ys, zs, '.', s=1)
ax.plot_trisurf(xs, ys, zs, linewidth=0.2, color=(0,0,0,0), edgecolor='Grey')
ax.view_init(44, -57)
plt.tight_layout()
# fig.savefig('interpolated_trisurf', dpi=300)
#%%
header = sample_data['header']
array = cc.cross_section2grid_elevation(bankline0, bankline1, crossline_list, 
                                          header, max_error=0.2)

from hipims_io import Raster
dem = Raster(array=array, header=header)
dem.mapshow()
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
z_grid = dem.array
ax.scatter(x_grid, y_grid, z_grid, '.', s=1)
ax.plot_trisurf(x_grid.flatten(), y_grid.flatten(), z_grid.flatten(),
                linewidth=0.2, color=(0,0,0,0), edgecolor='Grey')
for crossline in crossline_list:
    ax.plot(crossline[:, 0], crossline[:, 1], crossline[:, 2], '*')
