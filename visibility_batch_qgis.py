""""
Run this script in the python console in qgis. Can't run from here because 'processing' uses an old python? One of the wheels is in python2x and can't be installed.
Change input paths as needed.
Change output path as needed.

Potential problems occuring when running script:
When I run it for a small batch first to test, then running it again afterwards makes it not work. Probably problem with temporary files in QGIS.
A solution to this is to open a new project in QGIS and run it from there
TODO: make it so it deletes the tif files non-binary
"""
import numpy as np
import math
import processing
import os
from glob import glob
import geopandas as gpd

# Load the point shapefile into QGIS (make sure the path is correct)
points_layer = QgsVectorLayer(
    'C:/Users/susan/Documents/thesis/Thesis-terminal/output/river/KanaalVanWalcheren/KanaalVanWalcheren_mid.shp', 'points',
    'ogr')
if not points_layer.isValid():
    print("Layer failed to load!")
else:
    print(f"Loaded {points_layer.featureCount()} features.")

# THIS COMPUTES THE CLIPPED DSM!-----------------------------------------------------------------------------------------
# river_layer = QgsVectorLayer('C:/Users/susan/Documents/thesis/Thesis-terminal/input/river/longest_river.shp', 'river', 'ogr')
# if not river_layer.isValid():
#     print("River layer failed to load!")
# else:
#     print("River layer loaded successfully.")
#
# buffered_river_path = 'C:/Users/susan/Documents/thesis/Thesis-terminal/input/river/buffered_river.shp'
# buffer_params = {
#     'INPUT': river_layer,
#     'DISTANCE': 101,
#     'OUTPUT': buffered_river_path
# }
# processing.run("native:buffer", buffer_params)
# print(f"Buffered river saved to {buffered_river_path}")
#
# # Path to the DEM raster for viewshed analysis
# # dem_path = 'C:/Users/susan/Documents/thesis/Thesis-terminal/thesis_output/AHN_tiles_DSM/clipped_total.tif'
# dem_folder = 'C:/Users/susan/Documents/thesis/Thesis-terminal/thesis_output/AHN_tiles_DSM/'
# tif_files = glob(os.path.join(dem_folder, '*.tif'))
# merged_dem_path = 'C:/Users/susan/Documents/thesis/Thesis-terminal/input/AHN/DSM/merged_dem.tif'
# merge_params = {
#     'INPUT': tif_files,
#     'PCT': False,
#     'SEPARATE': False,
#     'NODATA_INPUT': None,
#     'NODATA_OUTPUT': None,
#     'OPTIONS': '',
#     'DATA_TYPE': 0,  # Byte
#     'OUTPUT': merged_dem_path
# }
# processing.run("gdal:merge", merge_params)
# print(f"Merged DSM saved to {merged_dem_path}")
#
# clipped_dem_path = 'C:/Users/susan/Documents/thesis/Thesis-terminal/input/AHN/DSM/clipped_dsm.tif'
# clip_params = {
#     'INPUT': merged_dem_path,
#     'MASK': buffered_river_path,
#     'OUTPUT': clipped_dem_path
# }
# processing.run("gdal:cliprasterbymasklayer", clip_params)
# print(f"Clipped DEM saved to {clipped_dem_path}")
# ----------------------------------------------------------------------------------------------------------------------

clipped_dem_path = 'C:/Users/susan/Documents/thesis/Thesis-terminal/input/AHN/KanaalVanWalcheren/DSM_test/merged_clipped.tif'
dem_path = clipped_dem_path

# Loop through each point in the shapefile
for i, feature in enumerate(points_layer.getFeatures()):
    if i < 1 :
        river_width = feature['width']
        print(f"river width before is {river_width}")
        if river_width == 0.0: #If width is zero, then there is no embankment points (only water)
            river_width = 200 #Default value for river width (cs width)
        print(f"river width after is {river_width}")
        max_distance = 100 + 0.5 * river_width
        point_geom = feature.geometry().asPoint()  # Get the point's geometry as a coordinate (x, y)
        height = feature['height'] or 1.75 #if height value is None/NULL then we assign a default. Height is only None is we don't use the cross-section as it only contains water
        print(f"river height is {height}")
        print(f"Processing feature ID: {feature.id()} with height: {feature['height']}")
        point_id = feature.id()  # Get the ID of the point feature

        # Set up parameters for r.viewshed
        params = {
            'input': dem_path,  # The DEM raster
            'coordinates': f'{point_geom.x()},{point_geom.y()}',  # Observer coordinates (point)
            'max_distance': max_distance,  # Maximum visibility distance
            'observer_elevation': f'{height}',  # Observer height (adjust as needed)
            'target_elevation': 0,  # Target height (adjust if needed)
            'output': f'C:/Users/susan/Documents/thesis/Thesis-terminal/output/visibility/KanaalVanWalcheren/viewsheds/viewshed_{point_id}.tif',
            # Output file path for each point
            '-b': True
        }

        # Run the r.viewshed tool
        processing.run('grass7:r.viewshed', params)
        if os.path.exists(params['output']):
            print(f"Viewshed output saved to: {params['output']}")
        else:
            print(f"Viewshed output failed for point ID: {point_id}")

print("Batch process completed!")
