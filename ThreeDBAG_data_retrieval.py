import requests
import geopandas as gpd
from shapely.geometry import box
import geopandas as gpd
import pandas as pd
import fiona
from OSM_data_river_retrieval import fetch_river_overpass
import matplotlib.pyplot as plt
import os
import subprocess
import gzip
import shutil
from tqdm import tqdm
from AHN_data_retrieval import fetch_AHN_data_bbox

river = 'Kanaal door Walcheren'
river = fetch_river_overpass(river, 'candelete.shp')
buffer = 100
# WFS URL for 3DBAG Note that only the 2D projection of the models is served via WMS/WFS.
wfs_url = 'https://data.3dbag.nl/api/BAG3D/wfs?request=getcapabilities'
tile_index_path = 'tile_index.fgb'
tiles_folder = '3dbag_tiles'


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
    print("Column names in the GeoDataFrame:", tile_index.columns.tolist())
    print(tile_index.head())
    # Display the summary of the GeoDataFrame
    tile_index.info()

    # Check which tiles intersect with the bounding box
    intersecting_tiles = tile_index[tile_index.geometry.intersects(bounding_box)]
    num_items = len(intersecting_tiles)
    print(f'Number of intersecting tiles: {num_items}')
    # Download each intersecting tile
    iteration = 0
    for index, row in intersecting_tiles.iterrows():
        tile_id = row['tile_id']
        gpkg_download = row['gpkg_download'] #this is the download link
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
                gdf = gpd.read_file(gpkg_filename)
                print(f"Successfully opened GeoPackage with GeoPandas: {gpkg_filename}")
                print(f"Original CRS: {gdf.crs}")
                # print(gdf.head())
                if gdf.crs != target_crs:
                    gdf = gdf.to_crs(target_crs)
                    print(f"Transformed CRS to: {gdf.crs}")

                transformed_gpkg = os.path.join(output_folder, f"{tile_id}_transformed.gpkg")
                gdf.to_file(transformed_gpkg, driver="GPKG")
                print(f"Saved transformed GeoPackage: {transformed_gpkg}")

                # Optionally, remove the original untransformed file
                os.remove(gpkg_filename)
                print(f"Removed original untransformed file: {gpkg_filename}")



            except Exception as e:
                print(f"Failed to open GeoPackage with GeoPandas: {e}")

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
    # Get all .gpkg files in the input folder
    gpkg_files = [f for f in os.listdir(input_folder) if f.endswith('.gpkg')]

    if not gpkg_files:
        print("No GeoPackage files found in the input folder.")
        return

    print(f"Found {len(gpkg_files)} GeoPackage files. Starting to combine...")

    # Initialize an empty list to store GeoDataFrames
    gdfs = []

    # Read each GeoPackage file and append to the list
    for file in tqdm(gpkg_files, desc="Processing files"):
        file_path = os.path.join(input_folder, file)
        try:
            gdf = gpd.read_file(file_path)
            gdfs.append(gdf)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not gdfs:
        print("No valid GeoDataFrames were created. Check your input files.")
        return

    print("Concatenating all GeoDataFrames...")
    # Combine all GeoDataFrames using pandas concat
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    # Ensure the CRS is set (use the CRS from the first GeoDataFrame)
    combined_gdf.crs = gdfs[0].crs

    print(f"Saving combined GeoPackage to {output_file}...")
    # Save the combined GeoDataFrame to a new GeoPackage file
    combined_gdf.to_file(output_file, driver="GPKG")

    print(f"Combined GeoPackage saved successfully. Total features: {len(combined_gdf)}")



# Buildings have Z polygons now:
# buildings = gpd.read_file('3dbag_tiles/10-74-338_transformed.gpkg')
# print(buildings.head())
# print(buildings.geom_type.unique())  # It's polygon
# buildings.plot(figsize=(10, 10), color='grey', edgecolor='black')
# plt.show()
# def fetch_3dbag(river_gdf, buffer, wfs_url):
#     river_geometry = river_gdf.geometry.unary_union
#     buffered_area = river_geometry.buffer(buffer)
#
#     # Get the bounding box for the buffered area
#     minx, miny, maxx, maxy = buffered_area.bounds
#     capabilities_params = {
#         'service': 'WFS',
#         'version': '2.0.0',
#         'request': 'GetCapabilities'
#     }
#     capabilities_response = requests.get(wfs_url, params=capabilities_params)
#     print(capabilities_response.text)  # This will show you what layers and operations are available
#
#     # WFS request to get building data
#     params = {
#         'service': 'WFS',
#         'version': '2.0.0',
#         'request': 'GetFeature',
#         'typeName': 'lod12',  # Replace with the actual layer name
#         'bbox': f'{minx},{miny},{maxx},{maxy},EPSG:28992',  # Adjust EPSG as needed
#         'outputFormat': 'application/json'
#     }
#
#     response = requests.get(wfs_url, params=params)
#
#     # Load the data into a GeoDataFrame
#     if response.status_code == 200:
#         print(f"Response status code: {response.status_code}, Response text : {response.text}")
#         buildings_data = response.json()
#         gdf_buildings = gpd.GeoDataFrame.from_features(buildings_data['features'], crs='EPSG:28992')
#         print(gdf_buildings)
#     else:
#         print(f"Error: {response.status_code}, Response: {response.text}")




# fetch_3dbag(river_gdf, buffer, wfs_url)
