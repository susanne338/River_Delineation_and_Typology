"""
data retrieval

AHN: via WFS request for bbox, and tie download of .tif files.
TODO: merge function does not work.
TODO: Nodata. values -246 is given now. Later these are removed, but I need to keep the points somehow, and check for water and building. Maybe check for height at DSM?
TODO: I think wcs url for DSM did not work. Alter so it takes tiles.
OSM: via overpy api.
TODO: tags
3DBAG: via WFS request

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
import overpy
from shapely.geometry import LineString, Polygon
import requests
from shapely.geometry import box
import geopandas as gpd
import pandas as pd
import fiona
import os
import gzip
import shutil
from tqdm import tqdm


# OSM RIVER-------------------------------------------------------------------------------------------------------------
def fetch_river_overpass(river_name, output_file):
    """
    :param river_name:
    :param output_file:
    :return: geodataframe of river eihter as geometry=lines or geometry = polygons AND saves as shapefile
    """
    api = overpy.Overpass()
    # These are all possibilities but I would get multiple rivers if I didn't specify that it was a canal. I need to alter the function so that I can specify more? or even just take some id instead of river name
    # Overpass query
    # query = f"""
    # [out:json];
    # (
    #   way["waterway"="river"]["name"="{river_name}"];
    #   way["waterway"="canal"]["name"="{river_name}"];
    #   relation["waterway"="river"]["name"="{river_name}"];
    #   way["water"="river"]["name"="{river_name}"];
    #   relation["water"="river"]["name"="{river_name}"];
    #   way["natural"="water"]["name"="{river_name}"];
    #   relation["natural"="water"]["name"="{river_name}"];
    # );
    # out body;
    # """
    query = f"""
        [out:json];
        (
          way["waterway"="river"]["name"="{river_name}"];
        );
        out body;
        """

    result = api.query(query)

    lines = []
    polygons = []

    for way in result.ways:
        # Resolve missing nodes (fetch missing node data if needed)
        way.get_nodes(resolve_missing=True)
        coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
        lines.append(LineString(coords))

    for rel in result.relations:
        for member in rel.members:
            if member.geometry:
                coords = [(float(geom.lon), float(geom.lat)) for geom in member.geometry]
                polygons.append(Polygon(coords))

    # Save to shapefile
    if lines:
        gdf = gpd.GeoDataFrame(geometry=lines,  crs="EPSG:4326")
        gdf = gdf.to_crs("EPSG:28992")
        gdf.to_file(output_file, driver='ESRI Shapefile')
        print(f"Shapefile saved with line geometries: {output_file}")
        return gdf
    elif polygons:
        gdf = gpd.GeoDataFrame(geometry=polygons,  crs="EPSG:4326")
        gdf = gdf.to_crs("EPSG:28992")
        gdf.to_file(output_file, driver='ESRI Shapefile')
        print(f"Shapefile saved with polygon geometries: {output_file}")
        return gdf
    else:
        print(f"No data found for {river_name}")


# Example usage
fetch_river_overpass("Maas", "input/river/maas.shp")






# AHN DATA RETRIEVAL----------------------------------------------------------------------------------------------------
# VIA WFS REQUEST FOR A BOUNDING BOX: CAN'T BE DONE FOR WHOLE AREA CAUSE AREA IS TOO BIG
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
    - bbox (tuple): The bounding box coordinates (min_x, min_y, max_x, max_y) of the cross-section

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


# VIA TILE DOWNLOAD USING A .json FILE FOR TILE INDICES
# json files need to be dowloaded and are in needed_files folder
def load_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


def tile_intersects(tile_coords, bbox):
    tile_polygon = Polygon(tile_coords)
    print(f"tile polygon {tile_polygon}")
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


def download_AHN_tiles(json_path, river_file, destination_folder):
    # Load the JSON data
    data = load_json(json_path)
    gdf_river = gpd.read_file(river_file)
    river_geometry = gdf_river.geometry.unary_union
    buffered_area = river_geometry.buffer(100)

    # Get the bounding box for the buffered area
    minx, miny, maxx, maxy = buffered_area.bounds

    # Ensure destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through the features (tiles) in the JSON
    for feature in data['features']:
        tile_coords = feature['geometry']['coordinates'][0]  # Polygon coordinates
        tile_url = feature['properties']['url']  # DSM tile URL
        tile_name = feature['properties']['name']  # Filename

        # Check if the tile intersects with the bounding box
        if tile_intersects(tile_coords, buffered_area.bounds):

            file_path = os.path.join(destination_folder, tile_name)
            if os.path.exists(file_path):
                print(f"Already downloaded tile {tile_name}")
                continue

            print(f"Tile {tile_name} intersects with the buffer. Downloading...")
            # Download the tile
            download_AHN_tile(tile_url, destination_folder, tile_name)
        else:
            print(f"Tile {tile_name} does not intersect with the bounding box.")


river='input/river/longest_river.shp'
json_path = 'input/AHN/kaartbladindex_AHN_DSM.json'
destination_folder = 'input/AHN/KanaalVanWalcheren/DSM_test'
download_AHN_tiles(json_path,river, destination_folder)

import geopandas as gpd

# CHECKING MAAS. THERE ARE 66 LINES
# gdf = gpd.read_file("input/river/maas.shp")
# # Check if the geometries are LineString or MultiLineString
# # If you have MultiLineString, we will split them into individual LineStrings
# all_lines = []
# for geom in gdf.geometry:
#     if geom.geom_type == 'MultiLineString':
#         all_lines.extend(list(geom.geoms))  # Split into individual LineStrings
#     elif geom.geom_type == 'LineString':
#         all_lines.append(geom)  # Add LineString directly
# # Calculate and print lengths
# for i, line in enumerate(all_lines):
#     length = line.length
#     print(f"Line {i + 1}: Length = {length}")

# MERGE tif files DOESNT WORK
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

# this checks the nodata value of a file and says 3.4028234663852886e+38 for R_65AZ2.tif and None for clipper_total.tif
# with rasterio.open('thesis_output/AHN_tiles_DSM/clipped_total.tif') as src:
#     nodata_value = src.nodata  # Get the NoData value
#     print(f"NoData value: {nodata_value}")




# 3DBAG-----------------------------------------------------------------------------------------------------------------
def fetch_3DBAG_tiles(fgb_path, buffer, river_gdf, output_folder, target_crs='EPSG:28992'):
    river_geometry = river_gdf.geometry.unary_union
    buffered_area = river_geometry.buffer(buffer)

    # Get the bounding box for the buffered area
    minx, miny, maxx, maxy = buffered_area.bounds
    bounding_box = box(minx, miny, maxx, maxy)
    # To also get the AHN information for QGIS visualization
    # Load the tile index from the .fgb file
    with fiona.open(fgb_path) as src:
        tile_index = gpd.GeoDataFrame.from_features(src)
    # print("Column names in the GeoDataFrame:", tile_index.columns.tolist())
    # print(tile_index.head())
    # # Display the summary of the GeoDataFrame
    # tile_index.info()

    # Check which tiles intersect with the bounding box
    # intersecting_tiles = tile_index[tile_index.geometry.intersects(bounding_box)]
    intersecting_tiles = tile_index[tile_index.geometry.intersects(buffered_area)]
    num_items = len(intersecting_tiles)
    print(f'Number of intersecting tiles: {num_items}')
    # Download each intersecting tile
    iteration = 0
    for index, row in intersecting_tiles.iterrows():

        tile_id = row['tile_id']
        gpkg_download = row['gpkg_download']  # this is the download link
        tile_id = tile_id.replace('/', '-')
        # print("tile id ", tile_id)
        print("gpkg download ", gpkg_download)
        download_url = gpkg_download
        # download_url = f"https://3dbag.nl/en/download?tid={tile_id}"
        # download_url = f"https://data.3dbag.nl/v20240420/tiles/9/284/556/{tile_id}.gpkg"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(download_url, headers=headers)
        # response = requests.get(download_url)

        if response.status_code == 200:
            # the tile is in compressed gz format
            gz_tile_filename = os.path.join(output_folder, f"{tile_id}.gpkg.gz")
            with open(gz_tile_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded tile: {tile_id} and iteration {iteration}")

            # I use iterations so it doesnt mmediatly download 4000 tiles here becuase i did something wrong before
            iteration += 1

            # Decompress the .gz file
            gpkg_filename = os.path.join(output_folder, f"{tile_id}.gpkg")
            with gzip.open(gz_tile_filename, 'rb') as f_in:
                with open(gpkg_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # print(f"Decompressed file: {gpkg_filename}")

            # Remove the .gz file to save space
            os.remove(gz_tile_filename)
            # print(f"Removed compressed file: {gz_tile_filename}")

            # Now try to open with geopandas
            try:
                # List all layers in the GeoPackage
                layers = fiona.listlayers(gpkg_filename)
                print(f"Layers in the GeoPackage: {layers}")

                # Process each layer
                for layer in layers:
                    gdf = gpd.read_file(gpkg_filename, layer=layer)
                    print(f"Processing layer: {layer}")
                    print(f"Original CRS: {gdf.crs}")

                    if gdf.crs != target_crs:
                        gdf = gdf.to_crs(target_crs)
                        print(f"Transformed CRS to: {gdf.crs}")

                    # Save each layer to a separate file or to the same file with layer name
                    transformed_gpkg = os.path.join(output_folder, f"{tile_id}_transformed.gpkg")
                    gdf.to_file(transformed_gpkg, driver="GPKG", layer=layer)
                    print(f"Saved transformed layer {layer} to GeoPackage: {transformed_gpkg}")

                # Optionally, remove the original file to save space
                os.remove(gpkg_filename)
                print(f"Removed original untransformed file: {gpkg_filename}")

            except Exception as e:
                print(f"Failed to process GeoPackage: {e}")

            #     gdf = gpd.read_file(gpkg_filename)
            #     print(f"Successfully opened GeoPackage with GeoPandas: {gpkg_filename}")
            #     print(f"Original CRS: {gdf.crs}")
            #     # print(gdf.head())
            #     if gdf.crs != target_crs:
            #         gdf = gdf.to_crs(target_crs)
            #         print(f"Transformed CRS to: {gdf.crs}")
            #
            #     transformed_gpkg = os.path.join(output_folder, f"{tile_id}_transformed.gpkg")
            #     gdf.to_file(transformed_gpkg, driver="GPKG")
            #     print(f"Saved transformed GeoPackage: {transformed_gpkg}")
            #
            #     # Optionally, remove the original untransformed file
            #     os.remove(gpkg_filename)
            #     print(f"Removed original untransformed file: {gpkg_filename}")
            #
            #
            #
            # except Exception as e:
            #     print(f"Failed to open GeoPackage with GeoPandas: {e}")

            # buildings = gpd.read_file(f"{output_folder}/{tile_id}.gpkg", layer='lod12_2d')
            # print(buildings.head())

            # try:
            #     with fiona.open(tile_filename) as src:
            #         print(f"Successfully opened GeoPackage: {tile_filename}")
            #         print(f"Number of layers: {len(src)}")
            #         for layer in src:
            #             print(layer)
            # except Exception as e:
            #     print(f"Failed to open GeoPackage: {e}")

            # if iteration > 5:
            #     return

        else:
            print(f"Failed to download tile: {tile_id}, Status code: {response.status_code}")


def combine_geotiles(input_folder, output_file):
    """
    Combines the seperate tif files into one file
    Args:
        input_folder: Folder of tif files
        output_file: Path to output tif file. Specify if it is DTM or DSM in the filename!

    Returns:
    TODO: make it so it deletes the seperate tiles, actually just the whole folder
    """
    # Get all .gpkg files in the input folder
    gpkg_files = [f for f in os.listdir(input_folder) if f.endswith('.gpkg')]

    if not gpkg_files:
        print("No GeoPackage files found in the input folder.")
        return

    print(f"Found {len(gpkg_files)} GeoPackage files. Starting to combine...")

    # Dictionary to store GeoDataFrames for each layer
    layer_gdfs = {}

    # Process each GeoPackage file
    for file in tqdm(gpkg_files, desc="Processing files"):
        file_path = os.path.join(input_folder, file)
        try:
            # List all layers in the GeoPackage
            layers = fiona.listlayers(file_path)

            for layer in layers:
                gdf = gpd.read_file(file_path, layer=layer)
                if layer not in layer_gdfs:
                    layer_gdfs[layer] = []
                layer_gdfs[layer].append(gdf)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not layer_gdfs:
        print("No valid GeoDataFrames were created. Check your input files.")
        return

    print("Concatenating GeoDataFrames for each layer...")

    # Combine GeoDataFrames for each layer
    for layer, gdfs in layer_gdfs.items():
        print(f"Processing layer: {layer}")
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

        # Ensure the CRS is set (use the CRS from the first GeoDataFrame)
        combined_gdf.crs = gdfs[0].crs

        print(f"Saving combined layer {layer} to {output_file}...")
        # Save the combined GeoDataFrame to the output GeoPackage file
        combined_gdf.to_file(output_file, driver="GPKG", layer=layer)

        print(f"Layer {layer} saved successfully. Total features: {len(combined_gdf)}")

    print(f"All layers combined and saved to {output_file}")

wfs_url = 'https://data.3dbag.nl/api/BAG3D/wfs?request=getcapabilities'
buffer = 100  # Area around river so that there is some extra space for sure taken into account
tile_index_path = 'thesis_output/needed_files/tile_index.fgb' #fgb file from 3DBAG with dimensions of tiles
tiles_folder = 'input/3DBAG/maas/tiles' #output folder for my retrieved tiles
gdf_river = gpd.read_file('input/river/maas.shp') #shapefile of my river in NL
output_file = 'input/3DBAG/maas/combined_3DBAG_tiles_DTM.gpkg' #File to write all tiles combined to

# EXECUTE SCRIPT TO GET ALL TILES
# fetch_3DBAG_tiles(tile_index_path, buffer, gdf_river, tiles_folder)
# combine_geotiles(tiles_folder, output_file)

