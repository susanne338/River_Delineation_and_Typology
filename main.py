"""
what does this script do? it calls LoS
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_gradient_magnitude
import ruptures as rpt
from read_data import read_coordinates
from geopy.distance import great_circle
from LoS import los_analysis

from OSM_data_river_retrieval import fetch_river_overpass
from AHN_data_retrieval import fetch_AHN_data_bbox, extract_elevation
from profile_extraction import profile_extraction
from cross_section_extraction import cross_section_extraction
from ThreeDBAG_data_retrieval import fetch_3DBAG_tiles, combine_geotiles
from shapely.geometry import box
import pandas as pd
import geopandas as gpd


# URL OF WCS SERVICE AHN
wcs_url = 'https://api.ellipsis-drive.com/v3/ogc/wcs/8b60a159-42ed-480c-ba86-7f181dcf4b8a?request=getCapabilities&version=1.0.0&requestedEpsg=28992'

# SET PARAMETERS
river = 'Kanaal door Walcheren'
output_file_river = "KanaalDoorWalcherenTESTTTTTTTTT.shp"
interval = 1000  # Set interval distance (meters)
width = 100      # Set width of the cross-section (meters)
n_points = 50

# GET ELEVATION PROFILES
gdf_river = fetch_river_overpass(river, output_file_river)
for index, row in gdf_river.iterrows():
    riverline = row['geometry']
    print("index, riverline legth: ", index, riverline.length)

# gdf_cross_sections = cross_section_extraction(gdf_river, interval, width)
# combined_gdf, combined_gdf_left, combined_gdf_right = profile_extraction(gdf_cross_sections, n_points, wcs_url)
#
# # Then these files contains the elevation profiles. These are just for visualization
# combined_gdf.to_file(
#     r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/extracted_sections_BIGTESTTEST_shp/')
# combined_gdf_left.to_file(
#     r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/combined_gdf_left/')
# combined_gdf_right.to_file(
#     r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/combined_gdf_righ/')
#
#
# # print("elevation profile ", elevation_profiles[0])
#
# # CONTINUE BUILDING DATA 3DBAG
# # PARAMETERS
# buffer = 100  #Area around river so that there is some extra space for sure taken into account
# tile_index_path = 'tile_index.fgb'
# tiles_folder = '3dbag_tiles'
#
# # EXECUTE SCRIPT TO GET ALL TILES
# fetch_3DBAG_tiles(tile_index_path, buffer, gdf_river, tiles_folder)
# output_file = 'combined_3dbag.gpkg'  # name of the output combined file
# combine_geotiles(tiles_folder, output_file)


# THIS CANT BE DONE BECAUSE THE FETCH AHN FUNCTION WIDTH AND HEIGHT ARE NOT ALLOWED TO EXCEED 2048 SO I WOULD NEED TO SAVE THEM SEPERATLY I GUESS
# MAYBE FETCHING TILES
# # FETCH AHN CMPLETE
# river_geometry = gdf_river.geometry.unary_union
# buffer = 100
# buffered_area = river_geometry.buffer(buffer)
#
# # Get the bounding box for the buffered area
# minx, miny, maxx, maxy = buffered_area.bounds
# bbox = (minx, miny, maxx, maxy)
# fetch_AHN_data_bbox(wcs_url, bbox, 'FULL_AHN_INFO_TEMP.tif', 'FULL_AHN_INFO_MOD.tif' )

# SOME SHIT I WORKED ON BEFORE  ----------------------------------------------------------------------------------------
#
# def calculate_distances(x_coord, y_coord):
#     """
#     Calculate the Euclidean distances between consecutive points (x, y).
#     """
#     diff_x = np.diff(x_coords)
#     # print("difference: ", diff_x)
#     diff_y = np.diff(y_coords)
#     distance = np.sqrt(diff_x ** 2 + diff_y ** 2)
#     # cumulative_distances = np.concatenate([[0], np.cumsum(distance)])
#     # print("cum: ", cumulative_distances)
#     return distance
#
#
# # def distance_meters(x_coord, y_coord):
# #     distances_meter = [0]
# #     for i in range(len(x_coords) - 1):
# #         coord1 = (i, y_coords[i])
# #         coord2 = (i + 1, y_coords[i + 1])
# #         distance_meter = great_circle(coord1, coord2).meters
# #         distances_meter.append(distance_meter)
# #     array_distances = np.array(distances_meter)
# #     return array_distances
#
#
# # Load the CSV file (adjust the path accordingly)
# # file_path = 'test_images_and_data/ahn4_crossection_excel/Analyse_AHN4_24-09-05_1246_steps_almerecentrum.csv'  # replace with your actual file path
# file_path = 'test_images_and_data/ahn4_crossection_excel/Analyse_AHN4_24-09-06_1149_vlissinge_crossection.csv'
# # Call the function to read the CSV data
# x_coords, y_coords, elevation = read_coordinates(file_path)
#
# # Display the data (optional)
# print("X coordinates:", x_coords)
# print("Y coordinates:", y_coords)
# print("Elevation values:", elevation)
#
# # Step 1: Calculate actual distances between consecutive points
# distances_inv = calculate_distances(x_coords, y_coords)
# distances_sep = np.insert(distances_inv, 0, 0.0)
# distances = np.cumsum(distances_sep) #cumulative
# final_distance = distances[-1]
#
# print("distances: ", distances)
# print("lengths x, y, z, distance: ", len(x_coords), len(y_coords), len(elevation), len(distances))
#
# # This is for uniform distance between points
# # x = np.arange(len(elevation))
#
# los_analysis(elevation, final_distance, distances)
#
# # Compute gradient (slope)
# slope_inv = np.gradient(elevation, distances)
# print("slope invalid: ", slope_inv)
#
# slope = np.nan_to_num(slope_inv, nan=0.0)
#
#
# print("slope: ", slope)
#
# # Detect steep changes using Gaussian gradient magnitude
# gradient_magnitude = gaussian_gradient_magnitude(elevation, sigma=1)
#
#
# # Set a threshold for detecting flood walls
# threshold = 0.6  # You can fine-tune this value
# flood_wall_indices = np.where(gradient_magnitude > threshold)[0]
# # -----------------------------------------
# # Change Point Detection using `ruptures`
# # -----------------------------------------
# # Use the elevation data to detect change points (abrupt shifts)
# model = "l2"  # 'l2' model is used for detecting changes in mean or variance
# algo = rpt.Binseg(model=model).fit(elevation)
# change_points = algo.predict(n_bkps=5)  # Adjust `n_bkps` (number of breakpoints) as necessary
#
# # Output detected change points
# print(f"Change points detected at indices: {change_points[:-1]}")  # Exclude last index (end of the profile)
# print(f"Corresponding distances (x-values): {distances[change_points[:-1]]}")
# print(f"Flood wall indices: {flood_wall_indices}")
# print(f"Corresponding distances: {distances[flood_wall_indices]}")
#
# # Plot results
# plt.figure(figsize=(12, 6))
#
# # Plot the elevation profile
# plt.subplot(211)
# plt.plot(distances, elevation, label='Elevation')
# plt.scatter(distances[change_points[:-1]], elevation[change_points[:-1]], color='orange',
#             label='Detected Change Points',
#             zorder=5)
# plt.title('Elevation Profile with Change Points')
# plt.xlabel('Distance along cross-section')
# plt.ylabel('Elevation')
# plt.legend()
#
# # Plot gradient magnitude and detected flood walls
# plt.subplot(212)
# plt.plot(distances, gradient_magnitude, label='Gradient Magnitude', linestyle='--')
# plt.scatter(distances[flood_wall_indices], elevation[flood_wall_indices], color='red', label='Detected Flood Walls',
#             zorder=5)
# plt.title('Gradient Magnitude with Flood Walls')
# plt.xlabel('Distance along cross-section')
# plt.ylabel('Gradient Magnitude')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
