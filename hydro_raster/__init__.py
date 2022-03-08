#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Tue Mar  8 15:48:04 2022
# @author: Xiaodong Ming

"""
__init__
===========
To do:
    initialize hydro_raster package
    get sample data

"""

def get_sample_data(format_str):
    """ Get sample data for demonstartion
    
    Args:
        format_str: string to specify the data to get
        'tif', 'shp', 

    Returns:
        out_str: string returned as a file name
    """
    import os
    import pkg_resources
    data_path = pkg_resources.resource_filename(__name__, 'sample')
    if format_str == 'tif':
        out_str = os.path.join(data_path, 'CA1_5m.tif')
    elif format_str == 'shp':
        out_str = os.path.join(data_path, 'CA1_overhead_features.shp')
    else:
        raise ValueError('unsupported data format')
    
    return out_str