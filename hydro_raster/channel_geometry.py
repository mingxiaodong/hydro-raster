#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue Mar 23 19:48:38 2021
# @author: Xiaodong Ming

"""
channel_geometry
===========
To do:
    deal with polyline and points, sort bank lines, cross-section lines
    construct closed polygon with bank lines and cross lines
    identify geometry relations
"""
import numpy as np
from matplotlib import path
from numpy.linalg import norm
from shapely import geometry
from .coords_interp import line_1d_2d
def preprocessing_data(bankline0, bankline1, crossline_list):
    """ Preprocess raw data
    0.  Remove vertex without xy coordiantes (nans in points-array)
    1.	Unify the direction of two bank lines
    2.	Identify the first and the last cross-section lines, referring to the 
            bank line direction
    3.	Trime bank lines in their two ends
    4.  Cut outside vertices of cross-section lines and unify their directions
    5.	Sort the list of cross-section lines towards the bank line direction

    Parameters
    ----------
    bankline0 : 2-col array, xy coords of bank line vertex
    bankline1 : 2-col array, xy coords of bank line vertex
    crossline_list : list of 2-col arrays, xy coords of cross line vertex

    Returns
    -------
    bl0, bl1, cl_list_sort: sorted and trimmed bank lines and cross lines

    """
    # 0 Remove nan value vertex
    bl0 = bankline0[~np.isnan(bankline0[:, -1]), :]
    bl1 = bankline1[~np.isnan(bankline1[:, -1]), :]
    cl_list = []
    for crossline in crossline_list:
        crossline = crossline[~np.isnan(crossline[:, -1]), :]
        cl_list.append(crossline)
    # 1 Sort banklien vertex towards the same stream direction
    bl1 = sort_line_points(bl0[0], bl1) # first point of bl0 is the reference
    # 2 Find the first and last cross line
    ind_cl0, ind_cl1 = identify_crossline_ends(bl0, bl1, cl_list)
    # 3 Trime bank lines in their two ends
    bl0 = trim_bankline(bl0, cl_list[ind_cl0], cl_list[ind_cl1])
    bl1 = trim_bankline(bl1, cl_list[ind_cl0], cl_list[ind_cl1])
    # 4 Trime and sort cross line
    river_channel, cl0_cut, cl1_cut = construct_channel_polygon(bl0, bl1, 
                                            cl_list[ind_cl0], cl_list[ind_cl1])
    for ele in sorted([ind_cl0, ind_cl1], reverse = True):
        del cl_list[ele]
    cl_list_cut = trim_crossline_points(river_channel, cl_list)
    cl_list_cut.insert(0, cl0_cut)
    cl_list_cut.insert(-1, cl1_cut)
    # 5 Sort the list of cross-section lines towards the bank line directio
    cl_list_sort = sort_crossline_list(cl_list_cut, bl0)

    return bl0, bl1, cl_list_sort

def sort_line_points(point_refer, bankline):
    """Sort (bank) line points refer to the original point
    point_refer: 2-element array, the origin reference point of bank lines
    """
    bankline = bankline[~np.isnan(bankline[:,0]), :]
    p0 = bankline[0]
    p1 = bankline[-1]
    if norm(point_refer-p0) > norm(point_refer-p1):
        bankline_sorted = np.flipud(bankline)
    else:
        bankline_sorted = bankline+0
    return bankline_sorted

def identify_crossline_ends(bl0, bl1, crossline_list):
    """Identify the first and the last cross-section lines, referring to the 
            bank line direction
    Parameters
    ----------
    bl0 and bl1 are bankline vertices sorted towards the same direction
    crossline_list: contains middle point of each crossline
    Returns
    -------
    ind_cl0, ind_cl1: index of the first / last crossline

    """
    cl_mid_list = [] # middle point of each cross line
    for crossline in crossline_list:
        cl_mid = crossline[[0, -1], :2]
        cl_mid = cl_mid.mean(axis=0)
        cl_mid_list.append(cl_mid)
    cl_mid = np.array(cl_mid_list)
    point_ref = (bl0[0]+bl1[0])/2
    distance = norm(cl_mid-point_ref, axis=1)
    ind = distance.argsort()
    ind_cl0 = ind[0] # index of the first cross line
    point_ref = (bl0[-1]+bl1[-1])/2
    distance = norm(cl_mid-point_ref, axis=1)
    ind = distance.argsort()
    ind_cl1 = ind[0] # index of the alst cross line
    return ind_cl0, ind_cl1

def trim_bankline(bline, cline0, cline1):
    """make the ends of bankline be connected with the crosslines in two ends 
    bline: bankline
    cline0: crossline on the beginning
    cline1: crossline on the ending
    """
    # the left end
    straight_line = cline0[[0, -1], :2]
    inter_point, inter_ind = find_inter_point_on_polyline(bline, straight_line,
                                                          True)
    if inter_ind is None:
        b_segment = bline[0:2]
        inter_point = get_intersection(b_segment, straight_line)
        c_ind = nearest_points(inter_point, cline0[:, :2])
        bline = np.r_[cline0[c_ind, :2].reshape(1,2), bline]
    else:
        bline = bline[inter_ind:]
        bline[0] = inter_point    
    
    # the right end
    straight_line = cline1[[0, -1], :2]
    inter_point, inter_ind = find_inter_point_on_polyline(bline, straight_line,
                                                          True)
    if inter_ind is None:
        b_segment = bline[-2:]
        inter_point = get_intersection(b_segment, straight_line)
        c_ind = nearest_points(inter_point, cline1[:, :2])
        bline = np.r_[bline, cline1[c_ind, :2].reshape(1,2)]
    else:
        bline = bline[:inter_ind+1]
        bline[-1] = inter_point
    indexes = np.unique(bline, return_index=True, axis=0)[1]
    bline = [bline[i] for i in sorted(indexes)]# remove duplicated points
    bline = np.array(bline)
    return bline

def construct_channel_polygon(bl0, bl1, cl0, cl1):
    """ Cut two cross lines and create a Path object for river channel
    bl0, bl1 are sorted and trimmed banklines, cl0 and cl1 are the first and
    the last crossline without cutting
        point10--------------bank1-------------point11
           |                                      |
           |                                      |
        cross0                                  cross1
           |                                      |
           |                                      |
        point00--------------bank0-------------point01
    """
    ind_c00 = nearest_points(bl0[0], cl0[:, :2])
    ind_c10 = nearest_points(bl1[0], cl0[:, :2])
    ind_c01 = nearest_points(bl0[-1], cl1[:, :2])
    ind_c11 = nearest_points(bl1[-1], cl1[:, :2])
    if ind_c00 > ind_c10:
        cl0_cut = cl0[ind_c10:ind_c00+1]
        cl0_cut = np.flipud(cl0_cut)
    else:
        cl0_cut = cl0[ind_c00:ind_c10+1]
    if ind_c01 > ind_c11:
        cl1_cut = cl1[ind_c11:ind_c01+1]
        cl1_cut = np.flipud(cl1_cut)
    else:
        cl1_cut = cl1[ind_c01:ind_c11+1]
    cl0_cut[0, :2] = bl0[0]
    cl0_cut[-1, :2] = bl1[0]
    cl1_cut[0, :2] = bl0[-1]
    cl1_cut[-1, :2] = bl1[-1]
    xy_points = np.r_[bl0, cl1_cut[:, :2], 
                      np.flipud(bl1), np.flipud(cl0_cut[:, :2])]
    channel_points = path.Path(xy_points)
    return channel_points, cl0_cut, cl1_cut

def trim_crossline_points(river_channel, crossline_list):
    cl_list_cut = []
    N=0
    for crossline in crossline_list:
        points_xy = crossline[:, :2]
        ind = river_channel.contains_points(points_xy, radius=-0.001)
        crossline = crossline[ind, :]
        if ind.sum()==0:
            print(N)
            raise ValueError('crossline fully cut, check the radius value')
        N = N+1
        # remove repeat
        crossline = remove_duplicate_rows(crossline)
        cl_list_cut.append(crossline)
    return cl_list_cut
    
def sort_crossline_list(cl_list_cut, bline):
    
    # find two central points of cross line
    ind_b_list = []
    cl_mid_list = []
    for crossline in cl_list_cut:
        cl_mid = np.mean(crossline[:, :2], axis=0)
        ind_2 = nearest_points(cl_mid, crossline[:, :2], 2)
        cl_segm = crossline[ind_2, :2]
        _, ind_b = find_inter_point_on_polyline(bline, cl_segm)
        if ind_b is None:
            ind_b = nearest_points(cl_mid, bline, 1)
        ind_b_list.append(ind_b)
        cl_mid_list.append(cl_mid)
    # check duplicated ind
    cl_mid_list = np.array(cl_mid_list)
    ind_b_list = np.array(ind_b_list).astype('float64')
    u, c = np.unique(np.array(ind_b_list), return_counts=True)
    dup_ind = u[c>1] # duplicated ind
    if dup_ind.size>0:
        for ind_b_dup in dup_ind:
            ind_b_dup = int(ind_b_dup)
            cl_mid_dup = cl_mid_list[ind_b_list == ind_b_dup]
            b_point = bline[ind_b_dup].reshape(1,2)
            b_point = np.repeat(b_point, cl_mid_dup.shape[1], axis=0)
            d_cl2bl =  norm(cl_mid_dup-b_point, axis=1)
            add_v = d_cl2bl.argsort()
            add_v = add_v*(0.5/add_v.max())
            ind_b_list[ind_b_list == ind_b_dup] = ind_b_dup+add_v
            
    ind_sort = np.array(ind_b_list).argsort()
    cl_sort = [cl_list_cut[ind] for ind in ind_sort]
    return cl_sort

def split_channel(cl_list, bl0, bl1):
    """ Split river channel to river sections by cross-section lines
    and eidt end points of each line to make them connected with each other
    all input lines must be trimmed and sorted before
    Return:
        channel_list Dict with five keys:
        channel_dict = {'bank0':bline0, 'bank1':bline1,
                        'cross0':cline0, 'cross1':cline1,
                        'polygon':all_points
        point10--------------bank1-------------point11
            |                                      |
            |                                      |
        cross0                                  cross1
            |                                      |
            |                                      |
        point00--------------bank0-------------point01
    """
    bl0 = bl0+0
    bl1 = bl1+0
    channel_list = []
    for n in np.arange(len(cl_list)-1):
        cline0 = cl_list[n]
        cl_0 = cline0[[0, -1], :2]
        cline1 = cl_list[n+1]
        cl_1 = cline1[[0, -1], :2]
        p00, ind00 = find_inter_point_on_polyline(bl0, cl_0, avoid_none=True)
        p10, ind10 = find_inter_point_on_polyline(bl1, cl_0, avoid_none=True)
        p11, ind11 = find_inter_point_on_polyline(bl1, cl_1, avoid_none=True)
        p01, ind01 = find_inter_point_on_polyline(bl0, cl_1, avoid_none=True)
        # change the xy cooords of crossline endpoints to intersections
        cline0[0][:2]  = p00
        cline0[-1][:2] = p10
        cline0 = line_flatten(cline0)
        cline1[0][:2]  = p01
        cline1[-1][:2] = p11
        cline1 = line_flatten(cline1)
        bline0 = bl0[ind00:ind01+1]
        if bline0.shape[0]>1:
            if within_segment(p00, bline0[0:2]):
                bline0[0] = p00
            else:
                bline0 = np.r_[p00, bline0]
            if within_segment(p01, bline0[-2:]):
                bline0[-1] = p01
            else:
                bline0 = np.r_[bline0, p01]
        else:
            bline0 = np.r_[p00, bline0, p01]
        bline0 = remove_duplicate_rows(bline0)
        
        bline1 = bl1[ind10:ind11+1]
        if bline1.shape[0] > 1:
            if within_segment(p10, bline1[0:2]):
                bline1[0] = p10
            else:
                bline1 = np.r_[p10, bline1]
            if within_segment(p11, bline1[-2:]):
                bline1[-1] = p11
            else:
                bline1 = np.r_[bline1, p11]
        else:
            bline1 = np.r_[p10, bline1, p11]
        bline1 = remove_duplicate_rows(bline1)
        
        all_points = np.r_[bline0, cline1[:, :2], np.flipud(bline1),
                           np.flipud(cline0[:, :2])]
        channel_dict = {'bank0':bline0, 'bank1':bline1,
                        'cross0':cline0, 'cross1':cline1,
                        'polygon':all_points}
        channel_list.append(channel_dict)
    return channel_list

def line_flatten(polyline):
    # flatten polyline  as a straignt line refer to its two ends
    line_obj = line_1d_2d(polyline)
    if norm(polyline[0, :2] - polyline[-1, :2])==0:
        line_obj0 = line_1d_2d(polyline[[0, -2], :2])
    else:
        line_obj0 = line_1d_2d(polyline[[0, -1], :2])
    xy = line_obj0.d_to_xy(line_obj.d)
    polyline_new = line_obj.xy+0
    polyline_new[:, :2] = xy
    return polyline_new
    
#%% discretise river section
def break_bankline(bankline, max_error):
    """discretise bankline to a orderd line segments
    bankline: X and Y coordinates of bank line points
    max_error: the max distance between line points to straight bank line
    Return:
        break_ind: index of the break points
        broken_line: 2-col array
    """
    break_ind = []
    ind_0 = 0 # start index of a bank section
    ind_1 = bankline.shape[0] # end index of a bank section
    while ind_0 < ind_1:
        section_line = np.array([bankline[ind_0], bankline[ind_1-1]])
        points_xy = bankline[ind_0:ind_1]
        distance = distance_p2l(section_line, points_xy)
        if distance.max() > max_error:
            ind_1 = distance.argmax()+ind_0
            ind_1 = ind_1.astype('int64')
        else:
            break_ind.append(ind_1)
            ind_0 = ind_1
            ind_1 = bankline.shape[0]
    break_ind = np.r_[0, np.array(break_ind)-1]
    broken_line = bankline[break_ind]
    return break_ind, broken_line

def discretize_river_section(bl0_broken_e, bl1_broken_e):
    """discretize river channel to 3/4-side polygons
    river_polys = discretize_river_section(bl0_broken, bl1_broken)
    Return:
        river_polys: shapely multi-polygon
        break_points_full_b: full break points on line_bottom
        break_points_full_t: full break points on line_top
    """
    # expand break points with projections from break points on the other bank
    # break_points_full_b and break_points_full_t should have the same size
    # bl0_broken_expand, bl1_broken_expand = match_break_points_by_relative_pos(bl0_broken, 
    #                                                           bl1_broken)
    polygon_list = []
    for n in np.arange(1, bl0_broken_e.shape[0]):
        point00 = bl0_broken_e[n-1]
        point01 = bl1_broken_e[n-1]
        point11 = bl1_broken_e[n]
        point10 = bl0_broken_e[n]
        polygon = geometry.Polygon([point00, point01, point11, point10])
        polygon_list.append(polygon)
    section_polys = geometry.MultiPolygon(polygon_list)
    return section_polys

def match_break_points(bl0_broken, bl1_broken):
    """ Expand bank break points with points projected from the other bank
    break_points: to define break points of bank line sections, 
        sequence mattters
    points_to_project: coordinates of points on the other bank
    Return:
        break_points_expand
    """
    bl0_broken = remove_duplicate_rows(bl0_broken)
    bl1_broken = remove_duplicate_rows(bl1_broken)
    bl0_broken_e = bl0_broken+0
    bl1_broken_e = bl1_broken+0
    
    pts_repeat = []
    for xy in bl0_broken[1:-1]: # project bl0_broken to bankline1
        point_prj, ind_insert = project_point2polyline(bl1_broken_e, xy)
        if norm(point_prj-bl1_broken_e, axis=1).min()==0:
            pts_repeat.append(point_prj)
        else:
            bl1_broken_e = np.insert(bl1_broken_e, ind_insert, 
                                     point_prj.flatten(), axis=0)
    for pt_repeat in pts_repeat:
        ind_insert = np.where(norm(pt_repeat-bl1_broken_e, axis=1)==0)
        ind_insert = ind_insert[0][0]
        bl1_broken_e = np.insert(bl1_broken_e, ind_insert, 
                                     pt_repeat.flatten(), axis=0)
    
    pts_repeat = []
    for xy in bl1_broken[1:-1]: # project bl1_broken to bankline0
        point_prj, ind_insert = project_point2polyline(bl0_broken_e, xy)
        if norm(point_prj-bl0_broken_e, axis=1).min()==0:
            pts_repeat.append(point_prj)
        else:
            bl0_broken_e = np.insert(bl0_broken_e, ind_insert, 
                                     point_prj.flatten(), axis=0)
    for pt_repeat in pts_repeat:
        ind_insert = np.where(norm(pt_repeat-bl0_broken_e, axis=1)==0)
        ind_insert = ind_insert[0][0]
        bl0_broken_e = np.insert(bl0_broken_e, ind_insert, 
                                     pt_repeat.flatten(), axis=0)
    
    return bl0_broken_e, bl1_broken_e

def match_break_points_by_relative_pos(bl0_broken, bl1_broken):

    obj_0 = line_1d_2d(bl0_broken)
    obj_1 = line_1d_2d(bl1_broken)
    
    bl0_add = obj_0.d_to_xy(obj_1.d[1:-1])
    bl1_add = obj_1.d_to_xy(obj_0.d[1:-1])
    if bl0_add.size >= 2:
        bl0_e = np.r_[obj_0.xy, bl0_add]
        d0_all = np.concatenate([obj_0.d, obj_1.d[1:-1]])
        ind0 = d0_all.argsort()
        bl0_e = bl0_e[ind0]
    else:
        bl0_e = obj_0.xy
    
    if bl1_add.size >= 2:
        bl1_e = np.r_[obj_1.xy, bl1_add]
        d1_all = np.concatenate([obj_1.d, obj_0.d[1:-1]])
        ind1 = d1_all.argsort()
        bl1_e = bl1_e[ind1]
    else:
        bl1_e = obj_1.xy
    return bl0_e, bl1_e
    
def project_point2polyline(polyline, point_xy):
    """
    Project a point to the two nearest line sections and return index

    """
    num_p = polyline.shape[0]
    if num_p == 2: # only one segment
        ind_insert = 1
        point_prj, pos = point2segment(point_xy, polyline)
        if pos:
            point_projected = point_prj.reshape(1, 2)
        else:
            dis_pp = norm(polyline - point_xy, axis=1)
            if dis_pp[0] < dis_pp[1]:
                point_projected = polyline[0].reshape(1, 2)
            else:
                point_projected = polyline[1].reshape(1, 2)
    elif num_p > 2: # two segments
        if num_p == 3:
            ind_nearest = 1
        else:
            ind_nearest2 = nearest_points(point_xy, polyline, 2)
            ind_nearest = ind_nearest2[0]
            if ind_nearest == 0 or ind_nearest == num_p-1:
                ind_nearest = ind_nearest2[1]
            if ind_nearest >= num_p-1:
                ind_nearest = num_p-2
        segment = polyline[ind_nearest-1:ind_nearest+1]
        point_prj_0, pos0 = point2segment(point_xy, segment)
        segment = polyline[ind_nearest:ind_nearest+2]
        point_prj_1, pos1 = point2segment(point_xy, segment)
        # select the best projected point
        if not pos0: # outside
            if not pos1: # outside both two sections, select the nearest point
                # from the three points
                polyline_3pts= polyline[ind_nearest-1:ind_nearest+2]
                ind0 = nearest_points(point_xy, polyline_3pts)
                ind_insert = ind_nearest+ind0
                point_projected = polyline_3pts[ind0]
            else: # project on the second section
                point_projected = point_prj_1
                ind_insert = ind_nearest+1
        else: # onside
            if not pos1: # project on the first section
                point_projected = point_prj_0
                ind_insert = ind_nearest
            else: # onside both two sections, select the nearest projected point
                polyline_3pts= polyline[ind_nearest-1:ind_nearest+2]
                ind0 = nearest_points(point_xy, polyline_3pts)
                if ind0 == 0: # left
                    point_projected = point_prj_0
                    ind_insert = ind_nearest
                elif ind0 == 2: #right
                    point_projected = point_prj_1
                    ind_insert = ind_nearest+1
                else:                    
                    ind_insert = ind_nearest
                    point_projected = polyline_3pts[ind0]                
        point_projected = point_projected.reshape(1, 2)
    else:
        raise IOError('polyline must contains more than one points')
    

    return point_projected, ind_insert

def point2segment(points, segment):
    """ project point(s) to a line segment and identify its position  
    points: nx2 array
    segment: 2X2 array
    -------
    point_prj : nx2 array
        coordinates of the projected point.
    position: True|False
        indicate whether the projected point is in/outside the line segment.
        True: inside, False: outside 
    """
    p0 = segment[0]
    p1 = segment[1]
    line_vect = np.array(p0)-np.array(p1)
    if norm(line_vect)==0:
        unit_vect = line_vect*0
    else:
        unit_vect = line_vect/norm(line_vect)
    dot_values = np.dot(points-p0, unit_vect)
    if dot_values.size > 1:
        point_prj = p0+unit_vect*dot_values[:, None]
    else:
        point_prj = p0+unit_vect*dot_values
        point_prj = point_prj.reshape(1,2)
    dis2end = np.c_[norm(point_prj-p0, axis=1), norm(point_prj-p1, axis=1)]
    
    position = np.max(dis2end, axis=1) <= norm(p1-p0)
    if points.size == 2:
        position = position[0]
    return point_prj, position

#%% supporting functions
def find_inter_point_on_polyline(polyline, straight_line, cmpr_dist=False, 
                                 avoid_none=False):
    """find the intersection point between a straigt line and a poly line
    If there is no intersection, project the near end point of polyline to 
    line section
    
    Parameters
    ----------
    polyline : 2-col numpy array
        points of a polyline.
    straight_line : 2x2 numpy array
        points of a line section(only two points).
    cmpr_dist: whether compare the distances between straight_line and the 
        nearest crossed-segment and the length of straight_line

    Returns
    -------
    inter_point.
    inter_ind: index on the first point of the crossed segment

    """
    line_p1 = straight_line[0]
    line_p2 = straight_line[-1]
    inter_ind = crossed_segments_on_polyline(polyline, line_p1, line_p2, 
                                             cmpr_dist)
    if inter_ind is None:
        inter_point = None
    else:
        segment_p0 = polyline[inter_ind]
        segment_p1 = polyline[inter_ind+1]
        segment = polyline[inter_ind:inter_ind+2]
        if norm(segment_p0-segment_p1) == 0:
            inter_point = segment_p0
        else:
            # lines1 = np.r_[line_p1.reshape((1,2)), line_p2.reshape((1,2))]
            inter_point = get_intersection(straight_line, segment)
    if avoid_none & (inter_ind is None): # choose the nearest point
        point_mid = np.mean(straight_line, axis=0)
        inter_ind = nearest_points(point_mid, polyline, 1)
        inter_point = polyline[inter_ind]
    return inter_point, inter_ind

def crossed_segments_on_polyline(polyline_xy, line_p1, line_p2, 
                                 cmpr_dist=False):
    # line_p1, line_p2: two points of a straight line
    # polyline_xy: xy coordinates of polyline
    # line_p1 = np.array(line_p1).reshape((1,2))
    # line_p2 = np.array(line_p2).reshape((1,2))
    segments_p0 = polyline_xy[:-1, :]
    segments_p1 = polyline_xy[1:, :]
    side_values0 = which_side_to_line(line_p1, line_p2, segments_p0)
    side_values1 = which_side_to_line(line_p1, line_p2, segments_p1)
    ind = np.where(side_values0*side_values1 <= 0)
    ind = ind[0]
    if ind.size>0:
        # fine the nearest segment
        line_mid = (line_p1+line_p2)/2
        segments_p0 = segments_p0[ind, :]
        segments_p1 = segments_p1[ind, :]
        segments_mid = (segments_p0+segments_p1)/2
        distance = norm(segments_mid-line_mid, axis=1)
        ind_sort = distance.argsort()
        ind_crossed = ind[ind_sort[0]]
        if cmpr_dist:
            # distance between straight line midpoint and polyline_xy
            cl = np.r_[line_p1.reshape(1,2), line_p2.reshape(1,2)]
            s_crssd = polyline_xy[ind_crossed:ind_crossed+2]
            inter_point = get_intersection(cl, s_crssd)
            d2bline = norm(inter_point-line_mid)
            # length of the straight line
            d_cline = norm(line_p1-line_p2)
            if d2bline > d_cline: 
                #it means the crossed segment is far away, the crossline does not
                # cross banline in the nearest position
                ind_crossed = None
    else:
        ind_crossed = None
    return ind_crossed

def which_side_to_line(line_p1, line_p2, points_xy):
    A, B, C = get_line_coefs(line_p1, line_p2)
    # A * x + B * y = C
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    values = A*x + B*y - C
    values[values < 0] = -1
    values[values > 0] = 1
    return values
    
def get_line_coefs(p1, p2):
    """ Get line coefs of two points
    p1/p2: [2*N array] x and y coordinates 
    A * x + B * y = C
    """
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p2[0]*p1[1] - p1[0]*p2[1])
    return A, B, C

def get_intersection(lines1, lines2):
    """ Get the intersection of two straight line or lines 
    lines1/lines2: 2*2*N array, 
        axis 0: point 1~2
        axis 1: x, y
        axis 2: line 1~N
    if lines2 contains only one line while lines1 contains more, then lines2 
        will be duplicated to match the size of lines1
    # A1 * x + B1 * y = C1
    # A2 * x + B2 * y = C2
    """
    # for line 1, 2xN array
    p1 = lines1[0, 0:2] # first point of line
    p2 = lines1[1, 0:2] # second point of line
    L1 = get_line_coefs(p1, p2)
    # for line 2
    p1 = lines2[0, 0:2]
    p2 = lines2[1, 0:2] 
    L2 = get_line_coefs(p1, p2)
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    D = D.flatten()
    D[D==0] = np.nan
    x = Dx / D
    y = Dy / D
    if np.array([x, y]).size == 2:
        inter_point = np.array([x, y]).reshape(1, 2)
    else:
        inter_point = (x, y)
    return inter_point

def nearest_points(point_xy, poly_points, num=1):
    """
    find the nearest point on a line to a given point 
    Returns
    -------
    ind : index of the first nearest point.
    """
    distance = norm(poly_points-point_xy, axis=1)
    ind_sort = distance.argsort()
    if num==1:
        ind_nearest = ind_sort[0]
    else:
        ind_nearest = ind_sort[:num]
    return ind_nearest

def in_channel(channel_dict, xy_points, radius=-0.1):
    if 'polygon' in channel_dict.keys():
        poly_points = channel_dict['polygon']
    else:
        bank0 = channel_dict['bank0']
        bank1 = np.flipud(channel_dict['bank1'])
        cross0 = np.flipud(channel_dict['cross0'])
        cross1 = channel_dict['cross1']
        poly_points = np.r_[bank0, cross1[:, :2], bank1, cross0[:, :2]]
    section_poly = path.Path(poly_points)
    ind = section_poly.contains_points(xy_points, radius=radius)
    return ind

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

def remove_duplicate_rows(data_array):
    """ Remove duplicated points and preserve its order
    """
    u, ind = np.unique(data_array, axis=0, return_index=True)
    array_u = u[np.argsort(ind)]
    return array_u

def within_segment(point_xy, line_segment):
    dis2end = [norm(point_xy-line_segment[0]), 
               norm(point_xy-line_segment[1])]
    if np.max(dis2end)>norm(line_segment[0]-line_segment[1]):
        ind_logical = False
    else:
        ind_logical = True
    return ind_logical

def get_bounding_box(river_channel, margin=0.1):
    if hasattr(river_channel, 'get_extents'):
        xy = river_channel.get_extents().get_points()
    else:
        xy = np.array(river_channel)
    x_min = np.min(xy[:, 0])
    x_max = np.max(xy[:, 0])
    x_range = x_max-x_min
    y_min = np.min(xy[:, 1])
    y_max = np.max(xy[:, 1])
    y_range = y_max-y_min
    left = x_min-margin*x_range
    right = x_max+margin*x_range
    bottom = y_min-margin*y_range
    top = y_max+margin*y_range
    bounding_box = np.array([[left, bottom], [right, top]] )
    return bounding_box
    
    
    
    