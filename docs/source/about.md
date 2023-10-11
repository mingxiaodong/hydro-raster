## About

### About hydro_raster

Python code to process raster data for hydrological or hydrodynamic modelling, 
e.g., [SynxFlow](https://github.com/SynxFlow/SynxFlow) or [HiPIMS-CUDA](https://github.com/HEMLab/hipims). The style of this package follows the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html).

Functions included in this package:

1. merge raster files
2. edit raster cell values based on shapefile
3. convert cross-section lines to river bathymetry raster
4. remove overhead buildings/bridges on raster 
5. read, write, and visualise raster file

**The CRS of both DEM and Shapfiles must be projected crs whose map unit is meter.**