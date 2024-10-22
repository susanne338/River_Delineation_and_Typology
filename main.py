"""
main
"""

import geopandas as gpd
# from river_space_delineation import boundary_line_smooth


# URL OF WCS SERVICE AHN
# wcs_url = 'https://api.ellipsis-drive.com/v3/ogc/wcs/8b60a159-42ed-480c-ba86-7f181dcf4b8a?request=getCapabilities&version=1.0.0&requestedEpsg=28992'
#
# # SET PARAMETERS
# river = 'Kanaal door Walcheren'
# output_file_river = "river_shapefiles/KanaalDoorWalcheren.shp"
# interval = 100  # Set interval distance (meters) between cross-sections
# width = 250      # Set width of the cross-section (meters)
# n_points = 100    # points along the cross-section
# buffer = 100
# # The fgb file describes the tile indices and locations in the 3DBAG
# tile_index_path = 'needed_files/tile_index.fgb'
# # This folder will contain all the collected 3DBAG tiles (whne finding intersectiong with buildings, I only want to search in the tiles that intersect with the cross-section, so I have to keep all the files seperate)
# tiles_folder = '3dbag_tiles'
# # This folder will contain the file that contains all the tiles combined (mainly for visualization)
# output_file_tiles = '3DBAG_combined_tiles/combined_3DBAG.gpkg'
# # These folders contain the elevation profiles in csv format
# output_folder_profiles_left = 'profiles/left'
# output_folder_profiles_right = 'profiles/right'
# # These files contain the cross-sections (left, right, whole)
# output_file_cross_sections_1 = 'cross_sections/cross_sections_1.shp'
# output_file_cross_sections_2 = 'cross_sections/cross_sections_2.shp'
# output_file_cross_sections = 'cross_sections/cross_sections.shp'
# buildings = gpd.read_file('3DBAG_combined_tiles/combined_3DBAG.gpkg')
#
# # These files contain the cross-sections (left, right, whole)
# output_file_cross_sections_1_longest = 'cross_sections/cross_sections_1_longest.shp'
# output_file_cross_sections_2_longest = 'cross_sections/cross_sections_2_longest.shp'
# output_file_cross_sections_longest = 'cross_sections/cross_sections_longest.shp'

def select_longest_river(gdf_river):
    """
    Selects the longest river from a GeoDataFrame and returns it as a new GeoDataFrame.

    :param gdf_river: GeoDataFrame containing river geometries
    :return: GeoDataFrame containing only the longest river
    """
    # Initialize variables to store the longest river information
    max_length = 0
    longest_river = None

    # Iterate over the rows and find the river with the longest geometry
    for index, row in gdf_river.iterrows():
        riverline = row['geometry']
        if riverline.length > max_length:
            max_length = riverline.length
            longest_river = row

    # Create a new GeoDataFrame with the longest river
    gdf_longest_river = gpd.GeoDataFrame([longest_river], columns=gdf_river.columns, crs=gdf_river.crs)

    return gdf_longest_river

# GET RIVER
# gdf_river = fetch_river_overpass(river, output_file_river)
# gdf_river = gpd.read_file(output_file_river)

# TEMP save river buffer
# combined_river = gdf_river.unary_union
# river_buffer = combined_river.buffer(100)
# river_buffer_gdf = gpd.GeoDataFrame(geometry=[river_buffer], crs="EPSG:28992")
# river_buffer_gdf.to_file('river_shapefiles/buffer.shp')

# # Print all subrivers and lengths
# for index, row in gdf_river.iterrows():
#     riverline = row['geometry']
#     print("index, riverline length: ", index, riverline.length)

# gdf_longest_river = select_longest_river(gdf_river)
# gdf_longest_river.to_file('river_shapefiles/longest_river.shp', driver='ESRI Shapefile')
# riverlinee = gdf_longest_river.geometry
# print('longest river ', riverlinee.length)

# # GET CROSS-SECTIONS
# gdf, gdf_1, gdf_2, cross_sections_1, cross_sections_2, index_list = cross_section_extraction(gdf_longest_river, interval, width, output_file_cross_sections_longest,output_file_cross_sections_1_longest, output_file_cross_sections_2_longest)
#
# gdf_1 = gpd.read_file(output_file_cross_sections_1)

# # GET ELEVATION PROFILES
# combined_gdf, combined_gdf_left, combined_gdf_right = profile_extraction(gdf, n_points, wcs_url, output_folder_profiles_left, output_folder_profiles_right, 'AHN_tif/ahn_tif', 'AHN_tif/ahn_tif_mod')
#
#
# # GET ALL 3DBAG TILES
# fetch_3DBAG_tiles(tile_index_path, buffer, gdf_river, tiles_folder)
# combine_geotiles(tiles_folder, output_file_tiles)

# cross_section1_file = gpd.read_file(output_file_cross_sections_1_longest)
# cs_1_geo_list = cross_section1_file['geometry'].to_list()
# cross_section2_file = gpd.read_file(output_file_cross_sections_2_longest)
# cs_2_geo_list = cross_section2_file['geometry'].to_list()
#
# gdf_boundary_points, boundary_points_1 = process_cross_sections(cs_1_geo_list,  gdf_longest_river, buffer ,tile_index_path, tiles_folder)
# gdf_boundary_points.to_file('boundary_points_buildings/boundary_points_1.shp', driver='ESRI Shapefile')
# gdf_boundary_points, boundary_points_2 = process_cross_sections(cs_2_geo_list,  gdf_longest_river, buffer ,tile_index_path, tiles_folder)
# gdf_boundary_points.to_file('boundary_points_buildings/boundary_points_2.shp', driver='ESRI Shapefile')

# bp_1 = gpd.read_file('boundary_points_buildings/boundary_points_1.shp')
# bp_1_list = bp_1['geometry'].to_list()
# boundary_line_smooth(bp_1_list, gdf_longest_river, buildings, '../boundary_line_1.shp')

#THIS IS THE SCRIPT I RUN IN THE QGIS PYTHON CONSOLE IN THE SCRIPT THINGY START--------------
# import processing
#
# # Load the point shapefile into QGIS (make sure the path is correct)
# points_layer = QgsVectorLayer('C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/pythonProject/river_shapefiles/river_midpoints_elev_2m.shp', 'points', 'ogr')
# if not points_layer.isValid():
#     print("Layer failed to load!")
# else:
#     print(f"Loaded {points_layer.featureCount()} features.")
#
# # Path to the DEM raster for viewshed analysis
# dem_path = 'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/pythonProject/AHN_tiles_DSM/clipped_total.tif'
#
# # Max visibility distance for r.viewshed
# max_distance = 100  # Adjust as needed
#
# # Loop through each point in the shapefile
# for i, feature in enumerate(points_layer.getFeatures()):
#     if i < 2:
#         point_geom = feature.geometry().asPoint()  # Get the point's geometry as a coordinate (x, y)
#         point_id = feature.id()  # Get the ID of the point feature
#
#         # Set up parameters for r.viewshed
#         params = {
#             'input': dem_path,  # The DEM raster
#             'coordinates': f'{point_geom.x()},{point_geom.y()}',  # Observer coordinates (point)
#             'max_distance': max_distance,  # Maximum visibility distance
#             'observer_elevation': 2.00,  # Observer height (adjust as needed)
#             'target_elevation': 0,  # Target height (adjust if needed)
#             'output': f'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/pythonProject/visibility/viewshed_{point_id}.tif',  # Output file path for each point
#         }
#
#         # Run the r.viewshed tool
#         processing.run('grass7:r.viewshed', params)
#     else:
#         break
#
# print("Batch process completed!")
#THIS IS THE SCRIPT I RUN IN THE QGIS PYTHON CONSOLE IN THE SCRIPT THINGY END--------------

import rasterio
import matplotlib.pyplot as plt
import numpy as np
# Replace 'your_file.tif' with the path to your TIFF file
tif_file = 'visibility/viewshed_0.tif'

# Open the TIFF file
with rasterio.open(tif_file) as src:
    # Read the data from the first band
    band1 = src.read(1)

# Convert the band data to a numpy array for analysis
data = band1.astype(np.float32)  # Convert to float for calculations

# Calculate statistics
min_value = np.min(data)
max_value = np.max(data)
mean_value = np.mean(data)
std_dev = np.std(data)

# Print statistics
print(f'Minimum Value: {min_value}')
print(f'Maximum Value: {max_value}')
print(f'Mean Value: {mean_value}')
print(f'Standard Deviation: {std_dev}')

# Plot the image using matplotlib
print('show the plot!)')
plt.imshow(band1, cmap='gray')  # You can change the colormap if needed
plt.colorbar()  # Optional: Show color scale
plt.title('TIFF Image')
plt.axis('off')  # Hide axes
plt.show()
