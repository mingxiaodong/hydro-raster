hydro_raster
--------
Python code to process raster data for hydroligical or hydrodynamic modelling, 
e.g., [HiPIMS flood model](https://github.com/HEMLab/hipims). This code follows
 [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html).

Python version: >=3.6. To use the full function of this package for processing 
raster and feature files, *rasterio* and *pyshp* are required.

**The CRS of both DEM and Shapfiles must be projected crs whose map unit is meter.**

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

Tutorial

1. Read a raster file
```
from hydro_raster.Raster import Raster
from hydro_raster import get_sample_data
tif_file_name = get_sample_data('tif')
ras_obj = Raster(tif_file_name)
```
2. Visualize a raster file
```
ras_obj.mapshow()
ras_obj.rankshow(breaks=[0, 10, 20, 30, 40, 50])
```
3. Clip raster file
```
clip_extent = (340761, 341528, 554668, 555682) # left, right, bottom, top
ras_obj_cut = ras_obj.rect_clip(clip_extent) # raster can be cutted by a shapfile as well using clip function
ras_obj_cut.mapshow()
```
3. Rasterize polygons on a raster and return an index array with the same dimension of the raster array
```
shp_file_name = get_sample_data('shp')
index_array = ras_obj_cut.rasterize(shp_file_name)
```
4. Change raster cell values within the polygons by adding 20
```
ras_obj_new = ras_obj_cut.duplicate()
ras_obj_new.array[index_array] = ras_obj_cut.array[index_array]+20
```
5. Show the edited raster with the shapefile polygons
```
import matplotlib.pyplot as plt
from hydro_raster.grid_show import plot_shape_file
fig, ax = plt.subplots(1, 2)
ras_obj_cut.mapshow(ax=ax[0])
plot_shape_file(shp_file_name, ax=ax[0], linewidth=1)
ras_obj_new.mapshow(ax=ax[1])
plot_shape_file(shp_file_name, ax=ax[1], linewidth=1)
# values can also be changed based on the attributes of each shapefile features
```