#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Wed Nov  6 14:33:36 2019

"""

spatial_analysis
================

functions to analyse data in raster and/or feature datasets to replace ArcGridDataProcessing

Assumptions:
    * map unit is meter
    * its cellsize is the same in both x and y direction
    * its reference position is on the lower left corner of the southwest cell

To do:
    * read and write arc

-----------------

"""
__author__ = "Xiaodong Ming"
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rasterio as rio
import shapefile
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
def arc_header_read(file_name, header_rows=6):
    """ read the header of a asc file as a dictionary

    Args: 
        file_name: (string) file name
        header_rows: (int) number of header rows

    Return:
        dict: header: a dictionary with keys:

                ncols: (int) number of columns

                nrows: (int) number of rows

                xllcorner: (int/float) x-coordinate of the lower left corner of the lower left cell of the grid

                yllcorner: (int/float) y-coordinate of the lower left corner of the bottom left cell of the grid

                cellsize: (int/float) the length of one square cell

                NODATA_value: (int/float)|-9999 the value representing nodata cell

    """
    check_file_existence(file_name)
    # read header
    header = {} # store header information including ncols, nrows,...
    row_ite = 1
    if file_name.endswith('.gz'):
        with gzip.open(file_name, 'rt') as file_h:     
            for line in file_h:
                if row_ite <= header_rows:
                    line = line.split(" ", 1)
                    header_key = line[0]
                    header_value = float(line[1])
                    # convert key to lowercase except nodata_value
                    if len(header_key) < 12:
                        header_key = header_key.lower()
                    header[header_key] = header_value
                else:
                    break
                row_ite = row_ite+1
    else:
        # read header
        with open(file_name, 'rt') as file_h:
            for line in file_h:
                if row_ite <= header_rows:
                    line = line.split(" ", 1)
                    # convert key to lowercase except nodata_value
                    header_key = line[0]
                    header_value = float(line[1])
                    if len(header_key) < 12:
                        header_key = header_key.lower()
                    header[header_key] = header_value
                else:
                    break
                row_ite = row_ite+1
    header['ncols'] = int(header['ncols'])
    header['nrows'] = int(header['nrows'])
    return header

def arcgridread(file_name, header_rows=6, return_nan=True):
    """ Read ArcGrid format raster file

    Args:
        file_name: (str) the file name to read data
        header_rows: (int) the number of head rows of the asc file

    Return:
        numpy array: the data content
        dict: head of the raster to provide reference information of the grid
        tuple: outline extent of the grid (left, right, bottom, top)

    Note: 
        this function can also read compressed gz files

    """
    check_file_existence(file_name)
    header = arc_header_read(file_name, header_rows)
    array = np.loadtxt(file_name, skiprows=header_rows, dtype='float64')
    if return_nan:
        if 'NODATA_value' in header:
            array[array == header['NODATA_value']] = np.nan
    prj_file = file_name[:-4]+'.prj'
    if os.path.isfile(prj_file):
        with open(prj_file, 'r') as file:
            wkt = file.read()
    else:
        wkt = None
    return array, header, wkt

def arcgridwrite(file_name, array, header, compression=False):
    """ write gird data into a ascii file

    Args:
        file_name: (str) the file name to write grid data. A compressed file will automatically add a suffix '.gz'
        array: (int/float numpy array)
        header: (dict) to provide reference information of the grid
        compression: (logic) to inidcate whether compress the ascii file

    Example:

    .. code:: python

        gird = np.zeros((5,10))
        grid[0,:] = -9999
        grid[-1,:] = -9999
        header = {'ncols':10, 'nrows':5, 'xllcorner':0, 'yllcorner':0,
                  'cellsize':2, 'NODATA_value':-9999}
        file_name = 'example_file.asc'
        arcgridwrite(file_name, array, header, compression=False)
        arcgridwrite(file_name, array, header, compression=True)

    """
    array = array+0
    if not isinstance(header, dict):
        raise TypeError('bad argument: header')
    if file_name.endswith('.gz'):
        compression = True
    # create a file (gz or asc)
    if compression:
        if not file_name.endswith('.gz'):
            file_name = file_name+'.gz'
        file_h = gzip.open(file_name, 'wb')
    else:
        file_h = open(file_name, 'wb')
    file_h.write(b"ncols    %d\n" % header['ncols'])
    file_h.write(b"nrows    %d\n" % header['nrows'])
    file_h.write(b"xllcorner    %g\n" % header['xllcorner'])
    file_h.write(b"yllcorner    %g\n" % header['yllcorner'])
    file_h.write(b"cellsize    %g\n" % header['cellsize'])
    file_h.write(b"NODATA_value    %g\n" % header['NODATA_value'])
    array[np.isnan(array)] = header['NODATA_value']
    np.savetxt(file_h, array, fmt='%g', delimiter=' ')
    file_h.close()
    print(file_name + ' created')

def tif_read(file_name):
    """read tif file (only the 1st band) and return array, header, crs
    """
    with rio.open(file_name) as src:
        masked_array = src.read(1, masked=True)
        ras_meta = src.meta
    array = masked_array.data+0.0
    array[array == masked_array.fill_value] = np.nan
    ncols = ras_meta['width']
    nrows = ras_meta['height']
    geo_transform = ras_meta['transform']
    cellsize = geo_transform[0]
    x_min = geo_transform[2]
    y_max = geo_transform[5]
    xllcorner = x_min
    yllcorner = y_max-nrows*cellsize
    header = {'ncols':ncols, 'nrows':nrows,
              'xllcorner':xllcorner, 'yllcorner':yllcorner,
              'cellsize':cellsize, 'NODATA_value':ras_meta['nodata']}     
    crs = ras_meta['crs']
    return array, header, crs, ras_meta

def byte_file_read(file_name):
    """ Read file from a bytes object
    """
    # read header
    header = {} # store header information including ncols, nrows, ...
    num_header_rows = 6
    for _ in range(num_header_rows):
        line = file_name.readline()
        line = line.strip().decode("utf-8").split(" ", 1)
        header[line[0]] = float(line[1])
        # read value array
    array  = np.loadtxt(file_name, skiprows=num_header_rows, 
                        dtype='float64')
    array[array == header['NODATA_value']] = float('nan')
    header['ncols'] = int(header['ncols'])
    header['nrows'] = int(header['nrows'])
    return array, header

def read_shapefile_as_list(shp_name):
    # read shapefile as a list of dict using pyshp
    # return a list of dict giving geometry and a list of dict giving fields
    sf = shapefile.Reader(shp_name)
    shapes_all = sf.shapes()
    shape_dict_list = []
    for one_shape in shapes_all:
        shape_dict = {'type':one_shape.shapeTypeName, 
                      'coordinates':one_shape.points}
        shape_dict_list.append(shape_dict)
    records_all = sf.records()
    record_dict_list = [x.as_dict() for x in records_all]
    return shape_dict_list, record_dict_list

#%% Combine raster files
def combine_raster(asc_files, num_header_rows=6):
    """Combine a list of asc files to a DEM Raster

    Args:   
        asc_files: a list of asc file names. All raster files have the same 
        cellsize.
    """
    # default values for the combined Raster file
    xllcorner_all = []
    yllcorner_all = []
    extent_all =[]
    # read header
    for file in asc_files:
        header0 = arc_header_read(file, num_header_rows)
        extent0 = header2extent(header0)
        xllcorner_all.append(header0['xllcorner'])
        yllcorner_all.append(header0['yllcorner'])
        extent_all.append(extent0)
    cellsize = header0['cellsize']
    if 'NODATA_value' in header0.keys():
        NODATA_value = header0['NODATA_value']
    else:
        NODATA_value = -9999
    xllcorner_all = np.array(xllcorner_all)
    xllcorner = xllcorner_all.min()
    yllcorner_all = np.array(yllcorner_all)
    yllcorner = yllcorner_all.min()
    extent_all = np.array(extent_all)
    x_min = np.min(extent_all[:,0])
    x_max = np.max(extent_all[:,1])
    y_min = np.min(extent_all[:,2])
    y_max = np.max(extent_all[:,3])
#    extent = (x_min, x_max, y_min, y_max)
#    print(extent)
    nrows = int((y_max-y_min)/cellsize)
    ncols = int((x_max-x_min)/cellsize)
    header = header0.copy()
    header['xllcorner'] = xllcorner
    header['yllcorner'] = yllcorner
    header['ncols'] = ncols
    header['nrows'] = nrows
    header['NODATA_value'] = NODATA_value
    array = np.zeros((nrows ,ncols))+NODATA_value
    print(array.shape)
    for file in asc_files:
        array0, header0, _ = arcgridread(file, num_header_rows)
        extent0 = header2extent(header0)
        x0 = extent0[0]+header0['cellsize']/2
        y0 = extent0[3]-header0['cellsize']/2
        row0, col0 = map2sub(x0, y0, header)
        array[row0:row0+header0['nrows'],
              col0:col0+header0['ncols']] = array0
    array[array == header['NODATA_value']] = float('nan')
    extent = header2extent(header)
    return array, header, extent

#%% ------------------------Supporting functions-------------------------------
def check_file_existence(file_name):
    """ check whether a file exists
    """
    try:
        file_h = open(file_name, 'r')
        file_h.close()
    except FileNotFoundError:
        raise

def header2extent(header):
    """ convert a header dict to a 4-element tuple (left, right, bottom, top) 
    all four elements shows the coordinates at the edge of a cell, not center
    """
    left = header['xllcorner']
    right = header['xllcorner']+header['ncols']*header['cellsize']
    bottom = header['yllcorner']
    top = header['yllcorner']+header['nrows']*header['cellsize']
    extent = (left, right, bottom, top)
    return extent

def meta2header(ras_meta):
    """ Transfer rasterio meta object to a header dict
    """
    ncols = ras_meta['width']
    nrows = ras_meta['height']
    transform = ras_meta['transform']
    cellsize = transform[0]
    xllcorner = transform[2]
    yllcorner = transform[5] - cellsize*nrows
    header = {}
    header['ncols'] = ncols
    header['nrows'] = nrows
    header['xllcorner'] = xllcorner
    header['yllcorner'] = yllcorner
    header['cellsize'] = cellsize
    header['NODATA_value'] = ras_meta['nodata']
    return header

def shape_extent_to_header(shape, extent, nan_value=-9999):
    """ Create a header dict with shape and extent of an array
    """
    ncols = shape[1]
    nrows = shape[0]
    xllcorner = extent[0]
    yllcorner = extent[2]
    cellsize_x = (extent[1]-extent[0])/ncols
    cellsize_y = (extent[3]-extent[2])/nrows
    if cellsize_x != cellsize_y:
        raise ValueError('extent produces different cellsize in x and y')
    cellsize = cellsize_x
    header = {'ncols':ncols, 'nrows':nrows,
              'xllcorner':xllcorner, 'yllcorner':yllcorner,
              'cellsize':cellsize, 'NODATA_value':nan_value}
    return header

def map2sub(X, Y, header):
    """ convert map coordinates to subscripts of an array

        array is defined by a geo-reference header

    Args: 
        X: a scalar or numpy array of coordinate values
        Y: a scalar or numpy array of coordinate values

    Return: 
        numpy array: rows and cols in the array
    """
    # X and Y coordinate of the centre of the first cell in the array
    X = np.array(X)
    Y = np.array(Y)
    x0 = header['xllcorner']+0.5*header['cellsize']
    y0 = header['yllcorner']+(header['nrows']-0.5)*header['cellsize']
    rows = (y0-Y)/header['cellsize'] # row and col number starts from 0
    cols = (X-x0)/header['cellsize']
    if isinstance(rows, np.ndarray):
        rows = np.round(rows).astype('int64')
        cols = np.round(cols).astype('int64') #.astype('int64')
    else:
        rows = int(rows)
        cols = int(cols)
    return rows, cols

def sub2map(rows, cols, header):
    """convert subscripts of a matrix to map coordinates rows,
    cols: subscripts of the data matrix, starting from 0

    Args: 
        rows: rows in the array
        cols: cols in the array

    Returns: 
        X and Y coordinate values

    """
    #x and y coordinate of the centre of the first cell in the matrix
    if not isinstance(rows, np.ndarray):
        rows = np.array(rows)
        cols = np.array(cols)        
    extent = header2extent(header)
    left = extent[0] #(left, right, bottom, top)
    top = extent[3]
    X = left + (cols+0.5)*header['cellsize']
    Y = top  - (rows+0.5)*header['cellsize']  
    return X, Y

#% Extent compare between two Raster objects
def compare_extent(extent0, extent1):
    """Compare and show the difference between two Raster extents

    Args:
        extent0, extent1: objects or extent dicts to be compared
        displaye: whether to show the extent in figures

    Return:
        int: 0 extent0>=extent1; 1 extent0<extent1; 2 extent0 and extent1 have
        intersections

    """
    logic_left = extent0[0]<=extent1[0]
    logic_right = extent0[1]>=extent1[1]
    logic_bottom = extent0[2]<=extent1[2]
    logic_top = extent0[3]>=extent1[3]
    logic_all = logic_left+logic_right+logic_bottom+logic_top
    if logic_all == 4:
        output = 0
    elif logic_all == 0:
        output = 1
    else:
        output = 2
        print(extent0)
        print(extent1)
    return output
   
def extent2shape_points(extent):
    """Convert extent to a two-col numpy array of shape points
    """
    #extent = (left, right, bottom, top)
    shape_points = np.array([[extent[0], extent[2]], 
                             [extent[1], extent[2]], 
                             [extent[1], extent[3]], 
                             [extent[0], extent[3]]])
    return shape_points

def _adjust_map_extent(extent, relocate=True, scale_ratio=1):
    """
    Adjust the extent (left, right, bottom, top) to a new staring point and 
    new unit. extent values will be divided by the scale_ratio

    Example:
        if scale_ratio = 1000, and the original extent unit is meter, then the 
        unit is converted to km, and the extent is divided by 1000
    """
    if relocate:
        left = 0 
        right = (extent[1]-extent[0])/scale_ratio
        bottom = 0
        top = (extent[3]-extent[2])/scale_ratio
    else:
        left = extent[0]/scale_ratio
        right = extent[1]/scale_ratio
        bottom = extent[2]/scale_ratio
        top = extent[3]/scale_ratio
    return (left, right, bottom, top)

def _set_colorbar(ax,img,norm):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)
    y_tick_values = cax.get_yticks()
    boundary_means = [np.mean((y_tick_values[ii],y_tick_values[ii-1])) 
                        for ii in range(1, len(y_tick_values))]
    category_names = [(str(norm.boundaries[ii-1])+'~'+
                       str(norm.boundaries[ii]))
                      for ii in range(1, len(norm.boundaries))]
    category_names[0] = '<='+str(norm.boundaries[1])
    category_names[-1] = '>'+str(norm.boundaries[-2])
    cax.yaxis.set_ticks(boundary_means)
    cax.yaxis.set_ticklabels(category_names,rotation=0)
    return cax

def _set_color_legend(ax, norm, cmp,
                      loc='lower right', bbox_to_anchor=(1,0),
                      facecolor=None):
    category_names = [(str(norm.boundaries[ii-1])+'~'+
                       str(norm.boundaries[ii]))
                      for ii in range(1, len(norm.boundaries))]
    category_names[0] = '<='+str(norm.boundaries[1])
    category_names[-1] = '>'+str(norm.boundaries[-2])
    ii = 0
    legend_labels = {}
    for category_name in category_names:
        legend_labels[category_name] = cmp.colors[ii,]
        ii = ii+1
    patches = [Patch(color=color, label=label)
               for label, color in legend_labels.items()]
    ax.legend(handles=patches, loc=loc,
              bbox_to_anchor=bbox_to_anchor,
              facecolor=facecolor)
    return ax

def _insensitive_header_keys(header_dict):
    """
    

    Parameters
    ----------
    header_dict : change the string value 
        DESCRIPTION.

    Returns
    -------
    None.

    """

def main():
    print('Fucntions to process asc data')

if __name__=='__main__':
    main()
