#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Mar 25 22:19:33 2021
# @author: Xiaodong Ming

"""
coords_interp
===========
To do:
    [objective of this script]

"""
import numpy as np
from numpy.linalg import norm

class line_1d_2d:
    def __init__(self, polyline):
        polyline = remove_duplicate_rows(polyline)
        # polyline = polyline[:, :2]
        p0 = polyline[:-1]
        p1 = polyline[1:]
        self.xy = polyline
        self.num_vert = polyline.shape[0] #number of vertice
        self.lengths = norm(p0-p1, axis=1)
        self.dx = p1[:, 0]-p0[:, 0]
        self.dy = p1[:, 1]-p0[:, 1]
        self.length_total = self.lengths.sum()
        d = self.lengths.cumsum()/self.length_total
        self.d = np.insert(d, 0, 0)
        self.dd = self.d[1:]-self.d[:-1]
    
    def xy_to_1d(self, xy_points):
        if xy_points.size == 2:
            xy_points = [xy_points]
        d_points = []
        for xy0 in xy_points:
            ind_n = nearest_points(xy0, self.xy, num=2)
            ind_0 = ind_n.min()
            verts = self.xy[ind_0:ind_0+2, :]
            point_prj, pos = point2segment(xy0, verts)
            if not pos:
                ind_0 = ind_n.max()
            verts = self.xy[ind_0:ind_0+2, :]
            d2 = norm(verts-xy0, axis=1)
            d_r = d2[0]/d2.sum()
            d0 = self.d[ind_0] + d_r*(self.d[ind_0+1]-self.d[ind_0])
            d_points.append(d0)
        d_points = np.array(d_points)
        return d_points
    
    def d_to_xy(self, d_points):
        #d>=0 and d<=1
        d_points = np.array(d_points)
        # if d_points.size == 1:
        #     d_points = [d_points]
        xy_list = []
        for n in np.arange(d_points.size):
            d0 = d_points[n]
            if d0 <= 0:
                x = self.xy[0, 0]
                y = self.xy[0, 1]
            elif d0>=1:
                x = self.xy[-1, 0]
                y = self.xy[-1, 1]
            else:
                ind  = np.where(self.d >= d0)[0]
                ind = ind[0]-1
                d_ratio = (d0-self.d[ind])/self.dd[ind]
                x = self.xy[ind, 0]+d_ratio*self.dx[ind]
                y = self.xy[ind, 1]+d_ratio*self.dy[ind]
            xy_list.append([x, y])
        xy = np.array(xy_list)
        if xy.size == 2:
            xy = xy.reshape(1,2)
        return xy
        
def get_line_coefs(p0, p1):
    """ Get line coefs of two points
    p1/p2: [2*N array] x and y coordinates 
    A * x + B * y = C
    """
    if p0.size==2:
        p0 = p0.reshape(1,2)
    if p1.size==2:
        p1 = p1.reshape(1,2)
    A = p0[:, 1] - p1[:, 1]
    B = p1[:, 0] - p0[:, 0]
    C = p1[:, 0]*p0[:, 1] - p1[:, 0]*p0[:, 1]
    return A, B, C

def remove_duplicate_rows(data_array):
    """ Remove duplicated points and preserve its order
    """
    u, ind = np.unique(data_array, axis=0, return_index=True)
    array_u = u[np.argsort(ind)]
    return array_u

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