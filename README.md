hydro_raster
--------
Python code to process raster data for hydroligical or hydrodynamic modelling, 
e.g., [HiPIMS flood model](https://github.com/HEMLab/hipims). This code follows
 [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html).

Python version: >=3.6. To use the full function of this package for processing 
raster and shapefile, rasterio, and pyshp are required.

Functions included in this package:

1. merge raster files
2. edit raster cell values based on shapefile
3. convert cross-section lines to river bathymetry raster
4. remove overhead buildings/bridges on raster 
5. read, write, and visualise raster file

To install hydro_raster from command window/terminal:
```
pip install hydro_raster
```
To install using github repo:
```
git clone https://github.com/mingxiaodong/hydro-raster
cd hydro-raster
pip install .
```