#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Wed Feb  9 16:47:46 2022
# @author: Xiaodong Ming

"""
setup the package
To do:
    setup hydro-raster package
Created on Wed Apr  1 15:03:58 2020

@author: Xiaodong Ming
"""
from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
      name='hydro_raster',
      version='0.0.6',
      description='To process raster data for hydro-logical/dynamic modelling',
      url='https://github.com/mingxiaodong/hydro-raster',
      author='Xiaodong Ming',
      author_email='x.ming@lboro.ac.uk',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=['Development Status :: 3 - Alpha',
                   'Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3',
                   'Operating System :: OS Independent',],
      keywords='Raster Hydrological Hydrodynamic',
      license='LICENSE.txt',
      packages=find_packages(),
      include_package_data=True,
      package_data={'hydro_raster': ['sample/CA1_5m.tif',
                                     'sample/CA1_overhead_features.shp',]},
      install_requires=['rasterio', 'scipy', 'pyshp', 
                        'matplotlib', 'numpy', 'pandas', 'imageio'],
      python_requires='>=3.6')

"""
Required python package

#Sometimes you’ll want to use packages that are properly arranged with 
#setuptools, but aren’t published to PyPI. In those cases, you can specify a 
#list of one or more dependency_links URLs where the package can be downloaded,
#along with some additional hints, and setuptools will find and install the
#package correctly.
#For example, if a library is published on GitHub, you can specify it like:
setup(
    ...
    dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0']
    ...
)
"""
