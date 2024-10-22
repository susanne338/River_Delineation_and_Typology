"""
I try to call qgis functions but I can't seem to make it work
"""
import os
import sys
import json
import pandas as pd
import geopandas as gpd
import numpy as np

import sys
print(sys.path)
# from qgis.core import *
# from qgis.core import QgsApplication, QgsProcessingFeedback, QgsProcessingRegistry
# from qgis.analysis import QgsNativeAlgorithms
# def setup_qgis_environment():
#     # Set up system paths
#     qspath = './qgis_sys_paths.csv'
#     try:
#         paths = pd.read_csv(qspath).paths.tolist()
#         sys.path += paths
#     except FileNotFoundError:
#         print(f"Error: {qspath} not found. Please check the file path.")
#         return False
#
#     # Set up environment variables
#     qepath = './qgis_env.json'
#     try:
#         with open(qepath, 'r') as f:
#             js = json.load(f)
#         for k, v in js.items():
#             os.environ[k] = v
#     except FileNotFoundError:
#         print(f"Error: {qepath} not found. Please check the file path.")
#         return False
#
#     # Set PROJ_LIB for macOS
#     if sys.platform == 'darwin':
#         os.environ['PROJ_LIB'] = '/Applications/Qgis.app/Contents/Resources/proj'
#
#     return js
#
# def initialize_qgis(home_path, qgis=None):
#     # QGIS library imports
#     from qgis.core import QgsApplication, QgsProcessingFeedback, QgsProcessingRegistry
#     from qgis.analysis import QgsNativeAlgorithms
#
#     # Initialize QGIS
#     QgsApplication.setPrefixPath(home_path, True)
#     qgs = QgsApplication([], False)
#     qgs.initQgis()
#
#     # Initialize processing module
#     from processing.core.Processing import Processing
#     Processing.initialize()
#     import processing
#     QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
#
#     return qgs
#
#
#     # Setup QGIS environment
# js = setup_qgis_environment()
# if not js:
#     return
#
#     # Initialize QGIS
# qgs = initialize_qgis(js['HOME'])
#
#     # Get list of available algorithms
# algs = {alg.displayName(): alg.id() for alg in QgsApplication.processingRegistry().algorithms()}
# print(algs)
#
#     # Your SAGA processing code goes here
#
#     # Clean up
# qgs.exitQgis()





# THIS GETS THE RIVER MIDPOINTS FOR VISIBILITY ANALYSIS--------------------------------------------
river_file = '../river_shapefiles/longest_river.shp'

def river_points(interval, river_shp, output_file):
    river_gdf = gpd.read_file(river_shp)
    river_gdf = river_gdf.to_crs(epsg=28992)
    river_line = river_gdf['geometry'].iloc[0]
    river_points = []

    for distance_along_line in np.arange(0, river_line.length, interval):
        point_on_line = river_line.interpolate(distance_along_line)
        river_points.append(point_on_line)

    river_points_gdf = gpd.GeoDataFrame(geometry=river_points)
    # Give each point elevation
    river_points_gdf['elevation'] = 2.0
    river_points_gdf.to_file(output_file, driver='ESRI Shapefile')

    return river_points

river_points(100, river_file, '../river_shapefiles/river_midpoints_elev_2m.shp')
# processing.run("sagang:visibilityanalysis", {'ELEVATION':'C:/Users/susan/AppData/Local/Temp/processing_BQIVNW/253d55b7b2864863812e1db2c6aa6915/OUTPUT.tif','VISIBILITY':'TEMPORARY_OUTPUT','METHOD':1,'NODATA':True,'POINTS':'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/pythonProject/river_shapefiles/river_midpoints_elev_2m.shp','HEIGHT':'elevation','HEIGHT_DEFAULT':10})