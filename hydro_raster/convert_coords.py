#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Mon Mar 29 16:05:14 2021
# @author: Xiaodong Ming

"""
convert_coords
===========
To do:
    [objective of this script]

"""
import numpy as np
from matplotlib import path
from numpy.linalg import norm
from scipy import interpolate
from hipims_io import Raster
from rasterio import mask
from .channel_geometry import discretize_river_section
from .channel_geometry import get_intersection, distance_p2l, break_bankline
from .channel_geometry import preprocessing_data, split_channel
from .channel_geometry import construct_channel_polygon, in_channel
from .channel_geometry import get_bounding_box
from .channel_geometry import match_break_points_by_relative_pos

#%%
def cross_section2grid_elevation(bankline0, bankline1, crossline_list,
                                 cellsize, bounding_box=None, max_error=None):
    # preprocess data: sort and trim
    bl0, bl1, cl_sort = preprocessing_data(bankline0, bankline1, crossline_list)
    # split river polygon using crosslines
    river_sections = split_channel(cl_sort, bl0, bl1)
    river_channel, _, _ = construct_channel_polygon(bl0, bl1, cl_sort[0], 
                                              cl_sort[-1])
    # create a river topography grid
    if bounding_box is None:
        bounding_box = get_bounding_box(river_channel)
    x_grid, y_grid, header = generate_cells_xy(bounding_box, cellsize)
    if max_error is None:
        max_error = cellsize*2
    x_valid, y_valid, ind_valid = extract_valid_xy(x_grid, y_grid, header, 
                                                   river_channel)
    z_valid = x_valid+np.nan
    n = 0
    for channel_dict in river_sections:
        bs, cs, zs = interp_one_reach(channel_dict, x_valid, y_valid, 
                                      max_error)
        ind = ~np.isnan(zs)
        z_valid[ind] = zs[~np.isnan(zs)]
        print(str(n)+'/'+str(len(river_sections)))
        n = n+1
    z_grid = x_grid+np.nan
    z_grid[ind_valid] = z_valid
    return z_grid, header

def generate_cells_xy(bounding_box, cellsize):
    """get xy coordiantes of the center of each cell 
    bounding_box: 2x2 array [[xmin, ymin], [xmax, ymax]] 
    cellsize: length of each cell
    """
    xmin = np.min(bounding_box[:, 0])
    xmax = np.max(bounding_box[:, 0])
    ymin = np.min(bounding_box[:, 1])
    ymax = np.max(bounding_box[:, 1])
    xllcorner = xmin-cellsize/2
    yllcorner = ymin-cellsize/2
    ncols = np.abs((xmax-xmin)/cellsize).astype('int')
    nrows = np.abs((ymax-ymin)/cellsize).astype('int')
    header = {'ncols':ncols,
              'nrows':nrows,
              'xllcorner':xllcorner, 
              'yllcorner':yllcorner,
              'cellsize':cellsize, 
              'NODATA_value':-9999}
    x00 = xllcorner+cellsize/2
    y00 = yllcorner+nrows*cellsize-cellsize/2
    x_vect = np.linspace(x00, x00+cellsize*ncols, num=ncols, endpoint=False)
    y_vect = np.linspace(y00, y00-cellsize*nrows, num=nrows, endpoint=False)
    x_grid, y_grid = np.meshgrid(x_vect, y_vect)
    return x_grid, y_grid, header

def extract_valid_xy(x_grid, y_grid, header, river_channel):
    # extract valid xy coordinates and return as a 2-col array plus a ind array
    # river_channel: path or numpy arrays providing xy of river channel vertex
    raster_obj = Raster(array=x_grid, header=header)
    ind_valid = rasterise_polygon(river_channel, raster_obj)
    x_grid_1d = x_grid[ind_valid]
    y_grid_1d = y_grid[ind_valid]
    return x_grid_1d, y_grid_1d, ind_valid

def rasterise_polygon(river_channel, raster_obj):
    # river_channel is a Path object
    # raster_obj: a Raster object
    if hasattr(river_channel, 'vertices'):
        xy_list = river_channel.vertices.tolist()
    else:
        xy_list = river_channel.tolist()
    shapes = [{'type':'Polygon', 
               'coordinates':[xy_list]}]
    ds_rio = raster_obj.to_rasterio_ds()
    out_image, _ = mask.mask(ds_rio, shapes) #, crop=True
    rasterized_array = out_image[0]
    rasterized_array[np.isnan(rasterized_array)] = ds_rio.nodata
    index_array = np.full(rasterized_array.shape, True)
    index_array[rasterized_array == ds_rio.nodata] = False
    ds_rio.close()
    return index_array

def interp_one_reach(channel_dict, x, y, max_error):
    """ interp xy points within one reach splitted by two cross-section lines
    Parameters
    ----------
    channel_dict : dictionary provide xy coords of two bank lines and two
        cross lines of one river reach.
    x, y : coords of xy cells within the reach
    max_error : maximum error to discritise the reach bank lines

    Returns
    -------
    bs : bank coords
    cs : cross coords
    zs : elevation values
    ind: index of valid cells within the reach 
    """
    cl0 = channel_dict['cross0']
    cl1 = channel_dict['cross1']
    bl0 = channel_dict['bank0']
    bl1 = channel_dict['bank1']
    xy_cells_vec = np.c_[x.flatten(), y.flatten()]
    b_vec = x.flatten()+np.nan
    c_vec = x.flatten()+np.nan
    z_vec = x.flatten()+np.nan
    ind = in_channel(channel_dict, xy_cells_vec, radius=0.00001)
    if ind.sum() > 0:
        xy_valid = xy_cells_vec[ind]
        b, c = xy2bc(xy_valid, bl0, bl1, max_error)
        z = interp_z_on_bc(b, c, cl0, cl1)
        b_vec[ind] = b
        c_vec[ind] = c
        z_vec[ind] = z
    bs = b_vec.reshape(x.shape)
    cs = c_vec.reshape(x.shape)
    zs = z_vec.reshape(x.shape)
    return bs, cs, zs

def xy2bc(points_xy, bl0_sec, bl1_sec, max_error,
          relative_pos=True):
    """Convert points_xy to bc coords defined within two river cross-sections
    points_xy: XY points to be converted
    line_bank0/1: points XY coords of two bank lines  between the crosslines
    max_error: maximum discretization error (map unit)
    relative_pos: yes to choose get_relative_position to convert xy to bc,
        otherwise, choose get_relative_distance
    one river section would be divided into several quadr-/tri- angle polygons
    coords transfer is processed within each polygon
    """
    points_bc = points_xy*0+np.nan
    _, bl0_broken = break_bankline(bl0_sec, max_error)
    _, bl1_broken = break_bankline(bl1_sec, max_error)
    if relative_pos:
        convert_fun = get_relative_position
    else:
        convert_fun = get_relative_distance
    bl0_broken_e, bl1_broken_e = match_break_points_by_relative_pos(bl0_broken, bl1_broken)
    river_polys = discretize_river_section(bl0_broken_e, bl1_broken_e)
    b_coords_div = get_b_coords_of_bank_divisions(bl0_broken_e, bl1_broken_e)
    n = 1
    for geom in river_polys.geoms:
        xs, ys = geom.exterior.xy
        point00 = bl0_broken_e[n-1]
        point01 = bl0_broken_e[n]
        point10 = bl1_broken_e[n-1]
        point11 = bl1_broken_e[n]
        polygon_n = path.Path([point00, point01, point11, point10])       
        # add a small radius to include points on edges
        ind = polygon_n.contains_points(points_xy, radius=0.00001)
        if ind.sum()>0:
            points_xy_in = points_xy[ind, :]
            b_coords, c_coords = convert_fun(points_xy_in,
                                          point00, point01, point10, point11)
            # convert local B coords to global
            b0 = b_coords_div[n-1]
            b1 = b_coords_div[n]
            b_coords = b_coords*(b1-b0)+b0
            points_bc[ind, :] = np.c_[b_coords, c_coords]
        n = n+1
    bs = points_bc[:, 0]
    cs = points_bc[:, 1]
    return bs, cs

def get_b_coords_of_bank_divisions(bl0_broken_e, bl1_broken_e):
    """ Get b coordinates of bank division lines within one river section
    """
    mid_points = (bl0_broken_e+bl1_broken_e)/2
    mid_gaps = mid_points[1:]-mid_points[:-1]
    mid_gaps = norm(mid_gaps, axis=1)
    mid_gaps_cumsum = np.cumsum(mid_gaps)
    b_coords = mid_gaps_cumsum/mid_gaps_cumsum.max()
    b_coords = np.insert(b_coords, 0, 0)
    return b_coords

def get_relative_distance(points_xy, point00, point01, point10, point11):
    """ Convert a Cartesian coordinates to a relative coordinates (B, C)
    points_xy: 2-col array for X and Y coordinates
    points position: point00, point01, point10, point11
    Return:
        coords_B: relative coordiantes along bank line
        coords_C: relative coordiantes along cross line
    """
    line_bank0 = np.array([point00, point01])
    line_bank1 = np.array([point10, point11])
    line_cros0 = np.array([point00, point10])
    line_cros1 = np.array([point01, point11])
    # axis along bank
    d2cros0 = distance_p2l(line_cros0, points_xy)
    d2cros1 = distance_p2l(line_cros1, points_xy)
    coords_B = d2cros0/(d2cros0+d2cros1)
    # axis along cross
    d2bank0 = distance_p2l(line_bank0, points_xy)
    d2bank1 = distance_p2l(line_bank1, points_xy)
    coords_C = d2bank0/(d2bank0+d2bank1)
    return coords_B, coords_C

def get_relative_position(points_xy, point00, point01, point10, point11):
    """ Find relative position of points_xy in a polygon
    via intersections of sides
    """
    line_bank0 = np.array([point00, point01])
    line_bank1 = np.array([point10, point11])
    line_cros0 = np.array([point00, point10])
    line_cros1 = np.array([point01, point11])
    ip_bank = get_intersection(line_bank0, line_bank1) #bank lines intersection
    ip_cros = get_intersection(line_cros0, line_cros1) #cross lines intersection
    # axis along bank
    if np.isnan(ip_bank.flat[0]): # parallel lines use projected distance
        d1 = distance_p2l(line_cros0, points_xy)
        d2 = distance_p2l(line_cros1, points_xy)
    else:
        d1, d2 = relative_pos_between2lines(points_xy, line_cros0, line_cros1,
                                            ip_bank)
    coords_B = d1/(d1+d2)
    # axis along cross section
    if np.isnan(ip_cros.flat[0]): # parallel lines use projected distance
        d1 = distance_p2l(line_bank0, points_xy)
        d2 = distance_p2l(line_bank1, points_xy)
    else:
        d1, d2 = relative_pos_between2lines(points_xy, line_bank0, line_bank1,
                                            ip_cros)
    coords_C = d1/(d1+d2)
    return (coords_B, coords_C)

def relative_pos_between2lines(points, side0, side1, intersection):
    """ Get relative position of points between two line sections
    sides1/sides2: two opposite sides of a quadrangle
    intersection：the intersection of the other two sides
    """
    point_num = points.shape[0] 
    intersection = np.array(intersection).flatten()
    intersection_rep = np.array([intersection,]*point_num)
    connect_lines = np.dstack([intersection_rep, points])
    connect_lines = connect_lines.transpose(2,1,0)
    if norm(side0[0]-side0[1])<=0.000001: # (nearly) triangle
        points_inters1 = np.array([side0[0].flatten()]*point_num)
    else:
        points_inters1 = get_intersection(connect_lines, side0)
        if np.array(points_inters1).size>2:
            points_inters1 = np.array(points_inters1).transpose(1,0)
    if norm(side1[0]-side1[1])<=0.000001: #(nearly) triangle
        points_inters2 = np.array([side1[0].flatten()]*point_num)
    else:
        points_inters2 = get_intersection(connect_lines, side1)
        if np.array(points_inters2).size>2:
            points_inters2 = np.array(points_inters2).transpose(1,0)
    d1 = norm(points_inters1-points, axis=1)
    d2 = norm(points_inters2-points, axis=1)
    return d1, d2

def interp_z_on_bc(bs, cs, cz0, cz1):
    """
    generate z values of relative bc coords according to cross line z values
    Parameters
    ----------
    bs : 1-col numpy array
        b(ank) coordinates.
    cs : 1-col numpy array
        c(ross) coordinates.
    cz0 : 2-col numpy array or 3-col numpy array(x,y,z)
        c(ross) and z coordinates of the start cross line.
    cz1 : 2-col numpy array or 3-col numpy array(x,y,z)
        c(ross) and z coordinates of the end cross line.

    Returns
    -------
    zs: 1-col numpy array
        interpolated z value of points defined by bs and cs
    """
    if cz0.shape[1]==3:
        cz = normalise_xy_coords(cz0[:, 0], cz0[:, 1])
        cz0 = np.c_[cz, cz0[:, 2]]
        cz = normalise_xy_coords(cz1[:, 0], cz1[:, 1])
        cz1 = np.c_[cz, cz1[:, 2]]
    c_interp0 = interpolate.interp1d(cz0[:, 0], cz0[:, 1], fill_value="extrapolate")
    c_interp1 = interpolate.interp1d(cz1[:, 0], cz1[:, 1], fill_value="extrapolate")
    z0 = c_interp0(cs)
    z1 = c_interp1(cs)
    zs = z0*(1-bs)+z1*bs
    return zs

def normalise_xy_coords(x, y):
    """ Convert cross line points to 1d relative coords cs
    convert xy coordiantes of points on one straignt line to relative coords
    and return values ranging 0 to 1, which cooresponding to position 
    starting from the first point and ending at the last point

    Parameters
    ----------
    x : 1-col numpy array
        x coords.
    y : 1-col numpy array
        y coords.

    Returns
    -------
    cs : 1-col numpy array
        1d relative coords ranging 0 to 1.

    """
    x.shape = (x.size, 1)
    y.shape = (y.size, 1)
    p0 = np.array([x[0], y[0]]).reshape((1,2)) # start point
    p1 = np.array([x[-1], y[-1]]).reshape((1,2)) # end point
    dist = np.linalg.norm(p1-p0, axis=1) # total length of the line
    d_points = np.linalg.norm(np.c_[x, y]-p0, axis=1) # distance to the start
    cs = d_points/dist
    return cs
