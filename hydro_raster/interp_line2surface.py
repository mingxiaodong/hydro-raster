#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interp_line2surface
To do:
    [Write the tasks to do in this script]

Input requirements:
    1. all river bank line points should be sorted towards the same flow 
        direction: upstream to downstream or the opposite
    2. all cross section points should be sorted from one bank to the other
    
General information:

    1. A b-c (bank-crossline) coordiante system:
        point10--------------bank1-------------point11
           |                                      |
           |                                      |
        cross0                                  cross1
           |                                      |
           |                                      |
        point00--------------bank0-------------point01

Universal variable names:
    line_poly: 2-col array, 1st col is x coords, 2nd col is y coords
        a polygonal line defined by a series of points (more than two)
    line_straight: 2x2 array, 1st col is x coords, 2nd col is y coords
        a straight line defined only by two points
    line_section: 2x2 array, 1st col is x coords, 2nd col is y coords
        a straight line between two points
    p0, p1, ..., pN: 2-element array provide x,y coords of one point
    points: 2-col array providing x(1st) and y(2nd) coords of multiple points
    
-----------------    
Created on Wed Apr 29 12:48:33 2020
@author: Xiaodong Ming
"""
#%%
import numpy as np
from numpy.linalg import norm
from matplotlib import path
from shapely import geometry
from scipy import interpolate
from channel_geometry import get_intersection, in_channel, trim_bankline
from channel_geometry import nearest_points
#%% 
def cross_section2grid_elevation(bankline0, bankline1, crossline_list,
                                 grid_header, max_error=None):
    
    # split river polygon using crosslines
    channel_list = split_river_polygon(bankline0, bankline1, crossline_list)
    # create a river topography grid
    cellsize = grid_header['cellsize']
    nrows = grid_header['nrows']
    ncols = grid_header['ncols']
    x00 = grid_header['xllcorner']+cellsize/2
    y00 = grid_header['yllcorner']+nrows*cellsize-cellsize/2
    x_vect = np.linspace(x00, x00+cellsize*ncols, num=ncols, endpoint=False)
    y_vect = np.linspace(y00, y00-cellsize*nrows, num=nrows, endpoint=False)
    x_grid, y_grid = np.meshgrid(x_vect, y_vect)
    if max_error is None:
        max_error = cellsize*2
    x_grid_1d = x_grid.flatten()
    y_grid_1d = y_grid.flatten()
    z_grid_1d = x_grid_1d+np.nan
    xy_1d = np.c_[x_grid_1d, y_grid_1d]
    for channel_dict in channel_list:
        ind = in_channel(channel_dict, xy_1d)
        xy_cells_vec = xy_1d[ind]
        if ind.sum() > 0:
            
            bc_cells = convert_coords_2d(xy_cells_vec, channel_dict['bank0'], 
                                         channel_dict['bank1'], max_error)
            bs = bc_cells[:, 0]
            cs = bc_cells[:, 1]
            zs = bc_interp_2d(bs, cs, channel_dict['cross0'],
                              channel_dict['cross1'])
            z_grid_1d[ind] = zs
    z_grid = np.reshape(z_grid_1d, x_grid.shape)
    return z_grid

#********************* Coordinates conversion ********************************
def get_split_lines_middle_coords(break_points_full_b, break_points_full_t):
    """ Get BC coordinates of channel split lines
    break_points_full_b/c: expanded break points from function 
        expand_break_points
    """
    mid_points = (break_points_full_b+break_points_full_t)/2
    mid_gaps = mid_points[1:]-mid_points[:-1]
    mid_gaps = norm(mid_gaps, axis=1)
    mid_gaps_cumsum = np.cumsum(mid_gaps)
    mid_bc_coords = mid_gaps_cumsum/mid_gaps_cumsum.max()
    mid_bc_coords = np.insert(mid_bc_coords, 0, 0)
    return mid_bc_coords

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
    if np.isnan(ip_bank[0]): # parallel lines use projected distance
        d1 = distance_p2l(line_cros0, points_xy)
        d2 = distance_p2l(line_cros1, points_xy)
    else:
        d1, d2 = relative_position_1d(points_xy, line_cros0, line_cros1,
                                       ip_bank)
    coords_B = d1/(d1+d2)
    # axis along cross section
    if np.isnan(ip_cros[0]): # parallel lines use projected distance
        d1 = distance_p2l(line_bank0, points_xy)
        d2 = distance_p2l(line_bank1, points_xy)
    else:
        d1, d2 = relative_position_1d(points_xy, line_bank0, line_bank1,
                                       ip_cros)
    coords_C = d1/(d1+d2)
    return (coords_B, coords_C)

def relative_position_1d(points, sides1, sides2, intersection):
    """ Get relative position of points between two line sections
    sides1/sides2: two opposite sides of a quadrangle
    intersection：the intersection of the other two sides
    """
    point_num = points.shape[0] 
    intersection = np.array(intersection).flatten()
    intersection_rep = np.array([intersection,]*point_num)
    connect_lines = np.dstack([intersection_rep, points])
    connect_lines = connect_lines.transpose(2,1,0)
    if norm(sides1[0]-sides1[1])<=0.000001: # (nearly) triangle
        points_inters1 = np.array([sides1[0].flatten()]*point_num)
    else:
        points_inters1 = get_intersection(connect_lines, sides1)
        points_inters1 = np.array(points_inters1).transpose(1,0)
    if norm(sides2[0]-sides2[1])<=0.000001: #(nearly) triangle
        points_inters2 = np.array([sides2[0].flatten()]*point_num)
    else:
        points_inters2 = get_intersection(connect_lines, sides2)
        points_inters2 = np.array(points_inters2).transpose(1,0)
    d1 = norm(points_inters1-points, axis=1)
    d2 = norm(points_inters2-points, axis=1)
    return d1, d2

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

def get_xy_from_relative_position(p0, p1, relative_position):
    """ Get xy coords of points on a line section defined by relative position
    Parameters
    ----------
    p0 : xy coords of the start point on line section.
    p1 : xy coords of the end point on line section.
    relative_position : numpy array
        relative_position ranging from 0 to 1.

    Returns
    -------
    x : numpy array
        x coords of the new points.
    y : numpy array
        y coords of the new points.
    """
    x = p0[0]+(p1[0]-p0[0])*relative_position
    y = p0[1]+(p1[1]-p0[1])*relative_position
    return x, y

def convert_coords_2d(points_xy, line_bank0, line_bank1, max_error,
                   conver_fun=get_relative_position):
    """Convert points_xy to bc coords defined within two river cross-sections
    points_xy: XY points to convert
    line_bank0/1: points XY coords of two bank lines  between the crosslines
    max_error: maximum discretization error (map unit)
    """
    points_bc = points_xy*0+np.nan
    line_b = line_bank0
    line_t = line_bank1
    _, breaks_t, breaks_b = discretize_river_channel(line_t, line_b, max_error)
    mid_bc_coords = get_split_lines_middle_coords(breaks_t, breaks_b)
    for n in np.arange(1, breaks_t.shape[0]):
        point00 = breaks_t[n-1]
        point01 = breaks_t[n]
        point10 = breaks_b[n-1]
        point11 = breaks_b[n]
        section_poly = path.Path([point00, point01, point11, point10])       
        # add a small radius to include points on edges
        ind = section_poly.contains_points(points_xy, radius=0.00001)
        if ind.sum()>0:
            points_xy_in = points_xy[ind, :]
#        conver_fun = cartesian2relative #uniformed_relative_position #
            points_B, points_C = conver_fun(points_xy_in,
                                          point00, point01, point10, point11)
        # conver local B coords to global
            B0 = mid_bc_coords[n-1]
            B1 = mid_bc_coords[n]
            points_B = points_B*(B1-B0)+B0
            points_bc[ind, :] = np.c_[points_B, points_C]
    return points_bc

#********************* Construct river channel *******************************
def discretize_bank_line(line_bank, max_error):
    """discretize bank line to straight line sections
    line_bank: X and Y coordinates of bank line points
    max_error: the max distance between line points to straight bank line
    Return:
        break_ind: index of the break points
    """
    break_ind = []
    ind_0 = 0 # start index of a bank section
    ind_1 = line_bank.shape[0] # end index of a bank section
    while ind_0 < ind_1:
        section_line = np.array([line_bank[ind_0], line_bank[ind_1-1]])
        points_xy = line_bank[ind_0:ind_1]
        distance = distance_p2l(section_line, points_xy)
        if distance.max() > max_error:
            ind_1 = distance.argmax()+ind_0
            ind_1 = ind_1.astype('int64')
        else:
            break_ind.append(ind_1)
            ind_0 = ind_1
            ind_1 = line_bank.shape[0]
    break_ind = np.r_[0, np.array(break_ind)-1]
    break_points = line_bank[break_ind]
    return break_ind, break_points

def discretize_river_channel(line_top, line_bottom, max_error):
    """discretize river channel to 3/4-side polygons
    Return:
        river_polys: shapely multi-polygon
        break_points_full_b: full break points on line_bottom
        break_points_full_t: full break points on line_top
    """
    _, break_points_t = discretize_bank_line(line_top, max_error)
    _, break_points_b = discretize_bank_line(line_bottom, max_error)
    points_xy_t = break_points_t[1:-1] # line_top[1:-1] # 
    points_xy_b = break_points_b[1:-1] # line_bottom[1:-1] # 
    # expand break points with projections from break points on the other bank
    # break_points_full_b and break_points_full_t should have the same size
    break_points_full_b = expand_break_points(break_points_b, points_xy_t)
    break_points_full_t = expand_break_points(break_points_t, points_xy_b)
    polygon_list = []
    for n in np.arange(1, break_points_full_b.shape[0]):
        point00 = break_points_full_b[n-1]
        point01 = break_points_full_t[n-1]
        point11 = break_points_full_t[n]
        point10 = break_points_full_b[n]
        polygon = geometry.Polygon([point00, point01, point11, point10])
        polygon_list.append(polygon)
    river_polys = geometry.MultiPolygon(polygon_list)
    return river_polys, break_points_full_b, break_points_full_t

def project_point2polyline(polyline, point_xy):
    """
    Project a point to the two nearest line sections and return index

    Parameters
    ----------
    polyline : 2-col array
        xy coords of points of a polyline.
    point_xy : 2-element array
        xy coords of the point to be projected.

    Returns
    -------
    point_projected : 2-element array
        xy coords of the projected point.
    ind_insert : int scalar
        position index of the projected point on the
            polyline points array.

    """
    ind_nearest2 = nearest_points(point_xy, polyline, 2)
    if ind_nearest2[0] == 0 or ind_nearest2[0] == polyline.shape[0]-1:
        ind_nearest = ind_nearest2[1]
    else:
        ind_nearest = ind_nearest2[0]
    
    p0 = polyline[ind_nearest-1]
    p1 = polyline[ind_nearest]
    point_prj_0, out0 = project_point2line_section(p0, p1, point_xy)
    out0 = out0[0]
    p0 = polyline[ind_nearest]
    p1 = polyline[ind_nearest+1]
    point_prj_1, out1 = project_point2line_section(p0, p1, point_xy)
    out1 = out1[0]
    # select the best projected point
    if out0:
        if out1: # outside both two sections, select the nearest break point
            point_projected = polyline[ind_nearest]
            ind_insert = ind_nearest
        else: # project on the second section
            point_projected = point_prj_1
            ind_insert = ind_nearest+1
    else: # out0 is false
        if out1: # project on the first section
            point_projected = point_prj_0
            ind_insert = ind_nearest
        else: # onside both two sections, select the nearest projected point
            if norm(point_prj_0-point_xy) > norm(point_prj_1-point_xy):
                point_projected = point_prj_1
                ind_insert = ind_nearest+1
            else:
                point_projected = point_prj_0
                ind_insert = ind_nearest
    return point_projected, ind_insert


def expand_break_points(break_points, points_to_project):
    """ Expand bank break points with points projected from the other bank
    break_points: to define break points of bank line sections, 
        sequence mattters
    points_to_project: coordinates of points on the other bank
    Return:
        break_points_expand
    """
#    break_points_exp = break_points+0 # break points expanded
    point_prj_list = []
    ind_insert_list = []
    for one_point in points_to_project:
        point_prj, ind_insert = project_point2polyline(
                break_points, one_point)
        point_prj_list.append(point_prj.flatten())
        ind_insert_list.append(ind_insert)
    break_points_expand = np.insert(break_points, ind_insert_list,
                                    point_prj_list, axis=0)
    return break_points_expand

def split_river_polygon(bankline0, bankline1, crossline_list):
    """
    bankline: bankline points sorted from upstream to downstream
    crossline0, crossline1: crossline points

    """
    channel_list = []
    for n in np.arange(len(crossline_list)-1):
        cline0 = crossline_list[n]
        cline1 = crossline_list[n+1]
        bline0 = trim_bankline(bankline0, cline0, cline1)
        bline1 = trim_bankline(bankline1, cline0, cline1)
        channel_dict = {'bank0':bline0, 'bank1':bline1,
                        'cross0':cline0, 'cross1':cline1}
        channel_list.append(channel_dict)
    return channel_list
    
#***************** Interpolation and calculation *****************************

def bc_interp_2d(bs, cs, cz0, cz1):
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


#%%*****************Geometry calculation *************************************


def two_lines_intersect(line0, line1):
    """ Check if two sections intersect
    line0, line2: 2X2 array, each row represents a point of line
    Return: True|False
    """
    line = geometry.LineString(line0)
    other = geometry.LineString(line1)
    return(line.intersects(other))
    
def distance_p2l(line_straight, points):
    """Calculate the distance from a point to a line defined by two points
    line_a: 2x2 array, 1st col is X and 2nd col is Y
    points: x and y coordinates of a point, can also be a 2-col array for 
        more than one points
    """
    p1 = line_straight[0, 0:2]
    p2 = line_straight[1, 0:2]
    if norm(p2-p1)==0:
        distance = norm(points-p1, axis=1)
    else:
        distance = np.cross(p2-p1, points-p1, axisb=1)/norm(p2-p1)
    distance = np.abs(distance)
    return distance

def project_point2line_section(p0, p1, points):
    """
    Project a point to a line section
    Returns
    -------
    point_prj : 2-element array
        coordinates of the projected point.
    outside : logical
        to indicate whether the projected point is outside the line section.

    """
    line_vect = np.array(p0)-np.array(p1)
    unit_vect = line_vect/norm(line_vect)
    dot_values = np.dot(points-p0, unit_vect)
    if dot_values.size > 1:
        point_prj = p0+unit_vect*dot_values[:, None]
    else:
        point_prj = p0+unit_vect*dot_values
        point_prj = point_prj.reshape(1,2)
    dis2end = np.c_[norm(point_prj-p0, axis=1), norm(point_prj-p1, axis=1)]
    outside = np.max(dis2end, axis=1)>norm(p1-p0)
    
    return point_prj, outside


def point_position_on_linesection(point, line_section):
    """
    Find the relative position of one point to a section of line which pass
    through one point
    
    Parameters
    ----------
    p : x,y coords of a point
    line_section : x,y coords of two points defining a line pass point p
    
    Returns
    -------
    ind_pos : -1|0|1, out on side of the first point(-1), within the line
        section(0), out on side of the last point(1)

    """
    p = np.array(point).flatten()
    d_p_p0 = norm(p-line_section[0])
    d_p_p1 = norm(p-line_section[1])
    d_p0_p1 = norm(line_section[0]-line_section[1])
    if np.logical_and(d_p_p0 < d_p_p1, d_p_p1 > d_p0_p1):
        ind_pos = -1 # out on p0 side
    elif np.logical_and(d_p_p1 < d_p_p0, d_p_p0 > d_p0_p1):
        ind_pos = 1  # out on p1 side
    else:
        ind_pos = 0
    return ind_pos


