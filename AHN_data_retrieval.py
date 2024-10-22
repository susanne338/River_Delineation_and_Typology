"""
Retrieves WCS AHN data in a defined bounding box by connecting to the wcs server
input: bbox
output: tif file
TO DO:
Here I change the nodata values to -246. The nodata values in the WCS are 0? Or None but have height 0 in the TIF at least that was what I saw in the QGIS but maybe its just None
It would be better to keep them at None! Becase now I changed elevations of 0 to -246 but maybe there are actual eleveations of 0
"""

import datetime
from owslib.wcs import WebCoverageService
import rasterio
from rasterio.io import MemoryFile
import numpy as np
import json
import requests
from shapely.geometry import Polygon, box
import os
import geopandas as gpd
import glob
from rasterio.merge import merge
import sys

# Set PROJ_LIB to the correct PROJ path used by GDAL/rasterio
os.environ['PROJ_LIB'] = r"C:\Program Files\QGIS 3.28.6\share\proj"

os.environ['GISBASE'] = r"C:\Program Files\QGIS 3.28.6\bin\grass82.bat"  # Path to GRASS installation
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['GISBASE'], 'bin')
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['GISBASE'], '')
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['GISBASE'], 'lib')

sys.path.append(os.path.join(os.environ['GISBASE'], 'etc', 'python'))


# import grass.script as gscript
def fetch_AHN_data_bbox(wcs_url, bbox, TIFfilename, TIFfilename_modified):
    """

    :param wcs_url:
    :param bbox:
    :return: elevation numpy.ndarray in BBOX from AHN wth nodata as -246
    """
    # Connect to the WCS service
    wcs = WebCoverageService(wcs_url, version='1.0.0')

    # # Get information about the available layers
    # print(list(wcs.contents))

    # List all available layers
    # print("Available layers:")
    # for layer in wcs.contents:
    #     print(layer)  # Print the layer name

    layer = '54e1e21c-50fe-42a4-b9fb-1f25ccb199af'

    # Get layer metadata
    # layer_info = wcs.contents[layer]
    # print(layer_info)

    # Set the timestamp (e.g., use the most recent date)
    # Use this timestamp when using elipse drive DTM
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d')
    # Use this timestamp when using elipse drive wcs for DSM 1.1.1
    # timestamp = '2023-04-14'
    print(f"Using timestamp: {timestamp}", 'and type ', type(timestamp))


    # Making the bbox slightly larger to avoid issues when extracting elevation for cross-section
    min_x, min_y, max_x, max_y = bbox
    min_x = min_x * 0.99999
    min_y = min_y * 0.99999
    max_x = max_x * 1.00001
    max_y = max_y * 1.00001
    bbox_dem = (min_x, min_y, max_x, max_y)

    # Calculate the width and height of the bounding box
    # bbox_width = bbox[2] - bbox[0]  # xmax - xmin
    # bbox_height = bbox[3] - bbox[1]  # ymax - ymin
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    # print(f"Bounding Box Width (in units): {bbox_width}")
    # print(f"Bounding Box Height (in units): {bbox_height}")

    # Fetch correct resolution
    desired_resolution = 0.5  # meters/pixel
    width_pixels = max(1, int(bbox_width / desired_resolution))
    height_pixels = max(1, int(bbox_height / desired_resolution))
    # print(f"Bounding Box Width (in pixels): {width_pixels}")
    # print(f"Bounding Box Height (in pixels): {height_pixels}")

    # Request the raster data for this area
    response = wcs.getCoverage(
        identifier=layer,
        bbox=bbox,
        crs='EPSG:28992',
        srs='EPSG:28992',
        format='geotiff',
        width=width_pixels,
        height=height_pixels,
        time=timestamp
    )
    print('Response content: ', response)
    # Read the data into a NumPy array
    with MemoryFile(response.read()) as memfile:
        with memfile.open() as dataset:
            elevation_data = dataset.read(1)  # Read the first band into a NumPy array

    # print('elevation data before ', elevation_data)
    # Replace NoData values (0) with -246
    elevation_data[elevation_data == 0] = -246

    # print('elevation data ', elevation_data)

    # Save the response as a .tif file
    with open(TIFfilename, 'wb') as f:
        f.write(response.read())

    # Open the input TIFF file
    with rasterio.open(TIFfilename) as src:
        # Retrieve the metadata
        metadata = src.meta

        # Print the NoData value for each band
        for band in range(1, src.count + 1):
            nodata_value = src.nodata  # Get the NoData value for the dataset
            # print(f"Band {band}: NoData value = {nodata_value}")
        # Read the number of bands
        # band_count = src.count
        # print('band count ', band_count)
        # Read the data
        data = src.read(1)  # Read the first band
        # print('data and type ', data, data.dtype)
        # # Ensure the data type is float32 for proper handling
        # if data.dtype != np.float32:
        #     data = data.astype(np.float32)

        # Create a mask for the NoData values (which are currently None and 0 in qgis)
        no_data_mask = (data == 0)
        # Print mask for debugging
        # print("NoData Mask:\n", no_data_mask)

        # # Change NoData values from 0 to -246
        # data[no_data_mask] = -246.0
        # Use np.where to replace NoData values with -246
        data = np.where(no_data_mask, -246, data)
        # print('data ', data)

        # Update the metadata for the new output
        metadata = src.meta.copy()
        metadata.update({
            'dtype': 'float32',
            'nodata': -246,
        })

    # Write the modified data to a new TIFF file
    with rasterio.open(TIFfilename_modified, 'w', **metadata) as dst:
        dst.write(data, 1)  # Write the modified data back to the first band

    # print("NoData values have been updated successfully!")
    return elevation_data, bbox_dem


# fetch_AHN_data_bbox(wcs_url, bbox)

def extract_elevation(elevation_data, points, bbox):
    """
    Extract elevation values from the elevation data for specified points.

    Parameters:
    - elevation_data (numpy.ndarray): 2D array of elevation data.
    - points (list of tuples): List of (x, y) coordinates for which to extract elevation.
    - bbox (tuple): The bounding box coordinates (min_x, min_y, max_x, max_y).

    Returns:
    - elevations (list): Elevation values at the specified points.
    """

    min_x, min_y, max_x, max_y = bbox
    elevations = []

    # Calculate pixel size
    pixel_size_x = (max_x - min_x) / elevation_data.shape[1]
    pixel_size_y = (max_y - min_y) / elevation_data.shape[0]

    for point in points:
        x, y = point
        # print('point ! and bbox stff ', point, bbox)

        # Check if the point is within the bounding box
        tolerance = 1e-5
        if (min_x - tolerance <= x <= max_x + tolerance) and (min_y - tolerance <= y <= max_y + tolerance):
            # Calculate pixel coordinates
            pixel_x = int((x - min_x) / pixel_size_x)
            pixel_y = int((y - min_y) / pixel_size_y)

            # Check bounds to avoid index errors
            if 0 <= pixel_x < elevation_data.shape[1] and 0 <= pixel_y < elevation_data.shape[0]:
                elevation_value = elevation_data[pixel_y, pixel_x]  # Note: pixel_y comes first in row-major order
                elevations.append(elevation_value)
            else:
                # print('the point is NOT within the bounding box')
                # print('point ! and bbox stff ', point, bbox)
                elevations.append(None)  # Out of bounds
        else:
            # print('the point is NOT within the bounding box')
            # print('point ! and bbox stff ', point, bbox)
            elevations.append(None)  # Outside the bbox

    return elevations


# TIF-------------------------------------------------------------------------------------------------------------------
# import geopandas as gpd
# import requests
# from shapely.geometry import box
#
# # Load the shapefile with the tile index (this can be provided by AHN services)
# tile_index = gpd.read_file('path_to_tile_index.shp')
#
# # Calculate the bounding box of the river
# river_bbox = gpd.GeoSeries([river_layer.total_bounds], crs="EPSG:28992")
#
# # Find the tiles that intersect the river's bounding box
# tiles_to_download = tile_index[tile_index.geometry.intersects(river_bbox.unary_union)]
#
# # Loop over the selected tiles and download the .tif files
# for index, row in tiles_to_download.iterrows():
#     tile_url = row['tile_url']  # Get the download URL from the index
#     tile_name = row['tile_name']
#
#     # Download the tile
#     response = requests.get(tile_url)
#     with open(f'{tile_name}.tif', 'wb') as f:
#         f.write(response.content)
#     print(f'Downloaded: {tile_name}')


# DOWNLOAD AHN TILES-------------------------------------------------------------------------------------------

def load_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


def tile_intersects(tile_coords, bbox):
    tile_polygon = Polygon(tile_coords)
    bbox_polygon = box(*bbox)  # Create a Polygon for the bounding box
    return tile_polygon.intersects(bbox_polygon)


def download_AHN_tile(url, destination_folder, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(destination_folder, filename)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {filename} (Status code: {response.status_code})")


def download_dsm_tiles(json_path, bbox, destination_folder):
    # Load the JSON data
    data = load_json(json_path)

    # Ensure destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through the features (tiles) in the JSON
    for feature in data['features']:
        tile_coords = feature['geometry']['coordinates'][0]  # Polygon coordinates
        tile_url = feature['properties']['url']  # DSM tile URL
        tile_name = feature['properties']['name']  # Filename

        # Check if the tile intersects with the bounding box
        if tile_intersects(tile_coords, bbox):
            print(f"Tile {tile_name} intersects with the bounding box. Downloading...")
            # Download the tile
            download_AHN_tile(tile_url, destination_folder, tile_name)
        else:
            print(f"Tile {tile_name} does not intersect with the bounding box.")


# json_path = 'needed_files/kaartbladindex_AHN_DSM.json'
# gdf = gpd.read_file('cross_sections/cross_sections_longest.shp')
# bbox = gdf.total_bounds
destination_folder = 'AHN_tiles_DSM'


# download_dsm_tiles(json_path, bbox, destination_folder)


# MERGE tif files DOESNT WORK------------------------------------------------------------------------------------------
def merge_tiles(combined_tiles, folder_of_tiles):
    """
    :param combined_tiles: file output
    :param folder_of_tiles: folder that contains the tiles to be merged
    :param nodata_value:
    :return:
    """

    tif_files = glob.glob(os.path.join(folder_of_tiles, '*.tif'))
    print('tif files ', tif_files)

    src_files_to_mosaic = []
    for tif_file in tif_files:
        src = rasterio.open(tif_file)
        src_files_to_mosaic.append(src)
    # Sources (list) is the sequence of dataset objects opened in r mode or path-like objects
    mosaic, out_transform = merge(src_files_to_mosaic)

    out_meta = src_files_to_mosaic[0].meta.copy()
    print('meta ', out_meta)

    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
        # "nodata": nodata_value,
        # "dtype": "float32"
    })

    with rasterio.open(combined_tiles, "w", **out_meta) as dest:
        dest.write(mosaic)
    for src in src_files_to_mosaic:
        print(f"File: {src.name}, Nodata Value: {src.nodata}")
        src.close()
    print(f"Merged TIF saved to {combined_tiles}")

# output_merged = 'AHN_tiles_DSM/merged.tif'
# folder = 'AHN_tiles_DSM'
# merge_tiles(output_merged, folder)
