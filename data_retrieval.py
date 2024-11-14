"""
Data retrieval script for river, AHN and 3DBAG data. To run this script, first download the needed JSON and
.fgb file for tile indices for AHN and 3DBAG.


OSM: via overpy api.
AHN: via WFS request via JSON file
3DBAG: via WFS request via .fgb file
"""

import json
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
# fetch_river_overpass("Maas", "input/river/maas/maas.shp")

def select_subrivers(river_shp, subrivers_list, river_output_shp):
    river = gpd.read_file(river_shp)
    selected_rivers = river[river['FID'].isin(subrivers_list)]
    selected_rivers.to_file(river_output_shp)

river = "input/river/maas/maas.shp"

subrivers_maas_roermond = [10, 37, 6]
river_roermond = "input/river/maas/roermond.shp"
subrivers_maas_venlo = [7, 28, 27, 34]
river_venlo = "input/river/maas/venlo.shp"
subrivers_maas_cuijk = [64, 9, 18]
river_cuijk = "input/river/maas/cuijk.shp"

# select_subrivers(river, subrivers_maas_cuijk,river_cuijk)
# AHN DATA RETRIEVAL----------------------------------------------------------------------------------------------------

def load_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


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

        tile_polygon = Polygon(tile_coords)
                # Check if the tile intersects with the bounding box
        if tile_polygon.intersects(buffered_area):
            # if tile_intersects(tile_coords, buffered_area.bounds):

            file_path = os.path.join(destination_folder, tile_name)
            if os.path.exists(file_path):
                print(f"Already downloaded tile {tile_name}")
                continue

            print(f"Tile {tile_name} intersects with the buffer. Downloading...")
            # Download the tile
            download_AHN_tile(tile_url, destination_folder, tile_name)
        else:
            print(f"Tile {tile_name} does not intersect with the buffer.")

        # file_path_delete = os.path.join(destination_folder, tile_name)
        # if os.path.exists(file_path_delete):
        #     if not tile_polygon.intersects(buffered_area):
        #         # If it does not intersect with the buffered area, delete the file
        #         os.remove(file_path_delete)
        #         print(f"Deleted tile {tile_name} as it does not intersect with the buffer.")
        #     else:
        #         print(f"Tile {tile_name} intersects with the buffer, keeping it.")


"""
download_ahn_tiles fetches the tiles that intersect with the buffered river using the tile indices
and the url in the JSON files that need to be downloaded beforehand. 
"""


# JSON paths
json_path_dsm = 'input/AHN/kaartbladindex_AHN_DSM.json'
json_path_dtm = 'input/AHN/kaartbladindex_AHN_DTM.json'

# Change these paths to correct river and destination folder
river='input/river/maas/maas.shp'
destination_folder= 'input/AHN/Maas/DTM'
# download_AHN_tiles(json_path_dtm,river, destination_folder)



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
        # print("gpkg download ", gpkg_download)
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


"""
3DBAG data retrieval is done via the .fgb file specifying the tile indices and url
"""


# buffer and path
buffer = 101  # Area around river so that there is some extra space for sure taken into account
tile_index_path = 'thesis_output/needed_files/tile_index.fgb' #fgb file from 3DBAG with dimensions of tiles

# Change these paths to correct river and destination folder
tiles_folder = 'input/3DBAG/maas/tiles' #output folder for my retrieved tiles
gdf_river = gpd.read_file('input/river/maas/maas.shp') #shapefile of my river in NL
output_file = 'input/3DBAG/maas/combined_tiles/combined.gpkg' #File to write all tiles combined to

# EXECUTE SCRIPT TO GET ALL TILES
# fetch_3DBAG_tiles(tile_index_path, buffer, gdf_river, tiles_folder)
# combine_geotiles(tiles_folder, output_file)

