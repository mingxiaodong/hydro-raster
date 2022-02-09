#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[file name]
To do:
    [Write the tasks to do in this script]
-----------------    
Created on Fri May  1 09:12:12 2020

@author: Xiaodong Ming
"""
import pickle
import interp_line2surface as itp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
#% Create a river channel
xmax0 = 50
xmax1 = xmax0-5
xmin0 = 0
xmin1 = xmin0+5
x0 = np.arange(0, xmax0)
y0 = np.sin(x0/5)*5+10
x1 = np.arange(xmin1, xmax0-5)
y1 = np.sin(x1/5)*5
line_t = np.c_[x0, y0] # top line
line_b = np.c_[x1, y1] # bottom line
cellsize = 0.1
xy_cells = np.meshgrid(np.arange(xmin0, xmax0, cellsize),
                       np.arange(y1.min(), y0.max(), cellsize))
del x0, y0, x1, y1, xmax0, xmin0, xmax1, xmin1

#%% Discretize river channel to polygons
max_error = 0.1 # maximum error in meter
river_polys, break_points_b, break_points_t = \
    itp.discretize_river_channel(line_t, line_b, max_error)
#% show discretized polygons
# itp.show_discretized_channel(line_t, line_b, max_error)
#% show bank discretization
itp.show_discretized_bank(line_b, max_error)
itp.show_discretized_bank(line_t, max_error)
#% show projections of break points
itp.show_projected_break_points(break_points_t, line_b, color='b')
itp.show_projected_break_points(break_points_b, line_t, color='r')
#% show midlle points of the division lines
mid_points = (break_points_b+break_points_t)/2
plt.plot(mid_points[:,0], mid_points[:,1], ':')
plt.gca().set_aspect('equal', adjustable='box')
fig = plt.gcf()
plt.show()
# fig.savefig('bank_break_points', dpi=300)

#%% Show relative coords of gridded points
# compare two methods of converting coords

xy_cells_vec = np.c_[xy_cells[0].flatten(), xy_cells[1].flatten()]
# get bank-cross coordiantes, two ways
bc_pos = itp.convert_coords_2d(xy_cells_vec, line_b, line_t, 
                            max_error, itp.get_relative_position)
bc_dis = itp.convert_coords_2d(xy_cells_vec, line_b, line_t, 
                            max_error, itp.get_relative_distance)

fig, ax = plt.subplots(2, 1)

points_b = bc_pos[:,0]
points_c = bc_pos[:,1]
points_c[points_c>0.5] = 1-points_c[points_c>0.5]
points_z = np.round(points_c*10)
grids_z = points_z.reshape(xy_cells[0].shape)
ax[0].contourf(xy_cells[0], xy_cells[1], grids_z, 10, cmap='YlGnBu', alpha=1)
points_z = np.round(points_b*20)
grids_z = points_z.reshape(xy_cells[0].shape)
ax[0].contourf(xy_cells[0], xy_cells[1], grids_z, 20, cmap='Reds', alpha=0.5)
ax[0].set_aspect('equal', adjustable='box')

points_b = bc_dis[:,0]
points_c = bc_dis[:,1]
points_c[points_c>0.5] = 1-points_c[points_c>0.5]
points_z = np.round(points_c*10)
grids_z = points_z.reshape(xy_cells[0].shape)
ax[1].contourf(xy_cells[0], xy_cells[1], grids_z, 10, cmap='YlGnBu', alpha=1)
points_z = np.round(points_b*20)
grids_z = points_z.reshape(xy_cells[0].shape)
ax[1].contourf(xy_cells[0], xy_cells[1], grids_z, 20, cmap='Reds', alpha=0.5)
ax[1].set_aspect('equal', adjustable='box')
plt.tight_layout()
# fig.savefig('two_coords_interp_compare', dpi=300)
"""

The conclusion of this part is that get_relative_position can better convert
xy coords to bc coords.

"""
#%%============= show points within a channel polygon
x0 = np.arange(0,15)
y0 = np.sin(x0/10)*5+10
x1 = np.arange(5,15)
y1 = np.sin(x1/10)*5

line_t = np.c_[x0, y0] # top line
line_b = np.c_[x1, y1] # bottom line
point00 = line_b[0]
point01 = line_b[-1]
point11 = line_t[-1]
point10 = line_t[0]
grids_xy = np.meshgrid(np.arange(0, 15, 0.2), np.arange(0, 15, 0.2))
points_xy = np.c_[grids_xy[0].flatten(), grids_xy[1].flatten()]
section_poly = path.Path([point00, point01, point11, point10])
ind = section_poly.contains_points(points_xy)
points_xy = points_xy[ind, :]
#% p.contains_points([(.5, .5)])
line_cros0 = np.array([point00, point10])
line_cros1 = np.array([point01, point11])
plt.plot(x0, y0)
plt.plot(x1, y1)
plt.plot(line_cros0[:,0], line_cros0[:,1], ':')
plt.plot(line_cros1[:,0], line_cros1[:,1], ':')
plt.scatter(points_xy[:,0], points_xy[:,1], s=1)
#plt.annotate('O', point_inters)
plt.gca().set_aspect('equal', adjustable='box')
