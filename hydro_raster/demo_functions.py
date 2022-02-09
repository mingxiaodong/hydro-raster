#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Wed Mar 24 00:08:04 2021
# @author: Xiaodong Ming

"""
demo_functions
===========
To do:
    [objective of this script]

"""
import matplotlib.pyplot as plt
import numpy as np
#%% display functions
# def show_discretized_bank(line_bank, max_error=5):
#     """ Plot discretized bank sections
#     """
#     break_ind, break_points = discretize_bank_line(line_bank, max_error)
#     plt.plot(line_bank[:,0], line_bank[:,1], 'b', linewidth=0.5)
#     plt.plot(break_points[:,0], break_points[:,1], '.')
#     for n in np.arange(1, break_points.shape[0]):
#         section_line = break_points[[n-1, n]]
#         plt.plot(section_line[:,0], section_line[:,1], 'g--')
#     plt.gca().set_aspect('equal', adjustable='box')
#     return break_ind

# def show_projected_break_points(points_xy, break_points, color='r'):
#     """ Show break points on one bank projecting to the other bank
#     """
#     points_prj2bank = []
#     for one_point in points_xy:
#         point_prj, _ = project_point2polyline(break_points,
#                                                           one_point)
#         points_prj2bank.append(point_prj)
#     points_prj2bank = np.array(points_prj2bank)
#     plt.plot(points_prj2bank[:,0], points_prj2bank[:, 1], '*', alpha=0.4)
#     prj_vectors = points_prj2bank - points_xy
#     plt.quiver(points_xy[:, 0], points_xy[:, 1], prj_vectors[:, 0],
#                prj_vectors[:, 1], scale_units='xy', scale=1, width=0.003,
#                color=color, alpha=0.4)

#%%
# fig, ax = plt.subplots(1)
# n = 0
# for n in np.arange(len(river_sections)):# 
#     channel_dict = river_sections[n]
#     ax.plot(channel_dict['bank0'][:, 0], channel_dict['bank0'][:, 1], 
#             marker='o') #, c='b'
#     ax.plot(channel_dict['bank1'][:, 0], channel_dict['bank1'][:, 1], 
#             marker='o') #, c='b'
#     ax.plot(channel_dict['cross0'][:, 0], channel_dict['cross0'][:, 1], 
#             marker='.') #, c='r'
#     ax.plot(channel_dict['cross1'][:, 0], channel_dict['cross1'][:, 1], 
#             marker='.')
#     all_points = channel_dict['polygon']
#     # ax.plot(all_points[:, 0], all_points[:, 1], marker='.', c='r')
#     ax.annotate(str(n), xy=np.median(all_points, axis=0))
# ax.set_aspect('equal', adjustable='box')

#%% plot processed banklines and crosslines
def draw_arrow(ax, line_data, width=0.2, color='k', len_n=1):
    x_a = line_data[-1, 0]
    y_a = line_data[-1, 1]
    dx_a = line_data[-1, 0]-line_data[-2, 0]
    dy_a = line_data[-1, 1]-line_data[-2, 1]
    ax.arrow(x_a, y_a, dx_a*len_n, dy_a*len_n, width=width, color=color)

def draw_banklines_crosslines(bl0, bl1, crossline_list, fname=None):
    """ Plot all banklines and corsslines
    """
    fig1, ax1 = plt.subplots(1)
    ax1.plot(bl0[:, 0], bl0[:, 1], 'r', marker='.')
    draw_arrow(ax1, bl0, width=1, color='b')
    ax1.plot(bl1[:, 0], bl1[:, 1], 'r', marker='.')
    draw_arrow(ax1, bl1, width=1, color='b')
    
    for n in np.arange(len(crossline_list)):
        line_data = crossline_list[n]
        if line_data.size>0:
            ax1.plot(line_data[:, 0], line_data[:, 1], marker='.')
            # draw_arrow(ax1, line_data, width=1, len_n=5)
            if n%2:
                xy = line_data[0][:2]
            else:
                xy = np.mean(line_data[:, :2], axis=0)
            ax1.annotate(str(n), xy=xy)

    if fname is not None:
        ax1.set_aspect('equal', adjustable='box')
        fig1.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=200)
    

#%%

# def show_discretized_channel(line_top, line_bottom, max_error):
#     """ Show discretized channel with polygons
#     call function discretize_river_channel
#     """
#     river_polys, _, _ = discretize_river_channel(line_top, line_bottom,
#                                                  max_error)
#     for geom in river_polys.geoms:
#         xs, ys = geom.exterior.xy
#         plt.fill(xs, ys, alpha=0.5)
#     plt.gca().set_aspect('equal', adjustable='box')
    
# def make_synthetic_crossline(cs, zs, p0=None, p1=None, num_of_stair=100):
#     """
#     make synthetic cross line using 1d relative coords of points
#     The number of points must be no less than 3. All these points must be
#     sorted and on the same straight line. 2-degree polyfit for 3 points, or
#     cubic interp1d for more than 3 points. If a start point(p0) and a end 
#     point(p1) are given, the function will return a 3-column array for xyz
#     coords. Otherwise, the return value will be a 2-column array for relative
#     coords and elevation

#     Parameters
#     ----------
#     cs : 1-col numpy array
#         1d relative coords.
#     zs : 1-col numpy array
#         elevation values.
#     num_of_stair: scalar
#         number of stairs (points) to the output cross line
#     p0 : xy coords of the start point on line section.
#     p1 : xy coords of the end point on line section.

#     Returns
#     -------
#     cz: 2-col numpy array
#         relative_position and z values of cross line points
#     xyz: 3-col numpy array for xyz coordiante of the cross line points
#         or 1-col numpy array only for z values.

#     """
#     cs_stairs = np.linspace(0, 1, num=num_of_stair+1, endpoint=True)
#     if zs.size >= 3:
#         line_fit = interpolate.interp1d(cs.flatten(), zs.flatten(),
#                                         kind='quadratic')
#     else:
#         raise ValueError('The number of points must be no less than 3')
#     z_stairs = line_fit(cs_stairs)
#     if (p0 is not None) and (p1 is not None):
#         x_stairs, y_stairs = get_xy_from_relative_position(p0, p1, cs_stairs)
#         xyz = np.c_[x_stairs, y_stairs, z_stairs]
#     else:
#         xyz = z_stairs
#     cz = np.c_[cs_stairs, z_stairs]
#     return cz, xyz
"""
#% Create a artificial river channel and cross section lines
# create bank lines
import pickle
xmax0 = 50
xmax1 = xmax0-5
xmin0 = 0
xmin1 = xmin0+5
x_bank_1 = np.arange(0, xmax0) # top line
y_bank_1 = np.sin(x_bank_1/5)*5+20
x_bank_0 = np.arange(xmin1, xmax0-5) # bottom line
y_bank_0 = np.sin(x_bank_0/5)*5+10
bankline0 = np.c_[x_bank_0, y_bank_0]
bankline1 = np.c_[x_bank_1, y_bank_1]
extent = (xmin0, xmax0, y_bank_0.min(), y_bank_1.max())
del x_bank_1, y_bank_1, x_bank_0, y_bank_0, xmax0, xmin0, xmax1, xmin1


# # upstream cross line

num_of_stair = 20
p0 = bankline0[0]
p1 = bankline1[0]
cs_given0 = np.array([0, 0.1, 0.3, 0.5, 1])
z_given0 = np.array([3, 2, 0, 1, 3])+5
_, cross_line0 = itp.make_synthetic_crossline(cs_given0, z_given0, p0, p1,
                                                      num_of_stair)
p0 = (14, 10)
p1 = (20, 17)
cs_given0 = np.array([0, 0.25, 0.5, 0.6, 0.9, 1])
z_given0 = np.array([3, 2, 1, 0, 2, 3])-2+5
_, cross_line1 = itp.make_synthetic_crossline(cs_given0, z_given0, p0, p1,
                                                      num_of_stair)


p0 = (30, 20)
p1 = (36.5, 14)
cs_given1 = np.array([0, 0.2, 0.5, 0.7, 1])
z_given1 = np.array([0,-2, -3, -2, 0])-0.5+5                          
_, cross_line2 = itp.make_synthetic_crossline(cs_given1, z_given1, p0, p1,
                                                      num_of_stair)

p0 = bankline0[-1]-0.5
p1 = bankline1[-1]+np.array([0.5, -1])
cs_given1 = np.array([0, 0.3, 0.6, 0.8, 1])
z_given1 = np.array([0,-2, -3, -2, 0])-2+5                      
_, cross_line3 = itp.make_synthetic_crossline(cs_given1, z_given1, p0, p1,
                                                      num_of_stair)

bankline1 = np.flipud(bankline1)
crossline_list = [cross_line0, cross_line1, cross_line2, cross_line3]
crossline_list = crossline_list[::-1]
header = {'ncols':120, 'nrows':60, 'cellsize':0.5, 
          'xllcorner':-5, 'yllcorner':0, 'NODATA_value':-9999}

sample_data = {'bankline0':bankline0, 'bankline1':bankline1,
                'crosslines':crossline_list, 'header':header}

with open('sample_data.pkl', 'wb') as f:
    pickle.dump(sample_data, f, pickle.HIGHEST_PROTOCOL)
"""