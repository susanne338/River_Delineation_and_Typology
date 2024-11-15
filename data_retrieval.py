"""
Data retrieval script for river, AHN and 3DBAG data. To run this script, first download the needed JSON and
.fgb file for tile indices for AHN and 3DBAG.


OSM: via overpy api.
AHN: via WFS request via JSON file
3DBAG: via WFS request via .fgb file
"""


import overpy
from shapely.geometry import LineString, Polygon
from shapely.geometry import box
import pandas as pd
import fiona
import os
import gzip
import shutil
from tqdm import tqdm
import geopandas as gpd
import requests
import json
import time
import os
from urllib.parse import urlparse
import zipfile



# OSM RIVER-------------------------------------------------------------------------------------------------------------
# to fetch, change tags in function
def fetch_river_overpass(river_name ,type, relation,  output_file):
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
          way["waterway"={type}]["name"="{river_name}"];
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
# fetch_river_overpass('Waal', 'river', '2675569','input/river/waal/waal.shp')
def select_subrivers(river_shp, subrivers_list, river_output_shp):
    river = gpd.read_file(river_shp)
    selected_rivers = river[river['FID'].isin(subrivers_list)]
    selected_rivers.to_file(river_output_shp)



subrivers_maas_roermond = [10, 37, 6]
subrivers_maas_venlo = [7, 28, 27, 34]
subrivers_maas_cuijk = [64, 9, 18]
subrivers_maas_maastricht = [2, 5]
subrivers_ar_amsterdam = [0]
subrivers_ar_utrecht = [27, 25]
subrivers_ijssel_zwolle_deventer = [0]
subrivers_lek = [2]
subrivers_vliet = [0, 5, 4, 11, 10, 6, 3]
subrivers_schie = [0]
subrivers_winschoterdiep = [0, 1, 2]
subrivers_harinx = [1, 6, 4, 11, 12, 2]
subrivers_nijmegen = [12,13,23]
subrivers_zaltbommel = [14]

river = 'input/river/maas/maas.shp'
river_maastricht = 'input/river/maas/maastricht/maastricht.shp'
# select_subrivers(river, subrivers_maas_maastricht, river_maastricht)
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
main_rivers = ['ijssel', 'waal']
cities = [['deventer'], ['nijmegen','zaltbommel']]



lonely_rivers = ['BovenMark', 'Harinxmakanaal', 'lek', 'vliet', 'Winschoterdiep']
lonelier_rivers = ['RijnSchieKanaal', 'Schie', ]
def ahn_for_loneliness(rivers):
    for river in rivers:
        print(river)
        file = f'input/river/{river}/{river}.shp'
        dsm_folder = f"input/AHN/{river}/DSM"
        os.makedirs(dsm_folder, exist_ok=True)
        dtm_folder = f"input/AHN/{river}/DTM"
        os.makedirs(dtm_folder, exist_ok=True)
        download_AHN_tiles(json_path_dsm, file, dsm_folder)
        download_AHN_tiles(json_path_dtm, file, dtm_folder)
def ahn_data(cities, main_river):
    for city in cities:
        river=f'input/river/{main_river}/{city}/{city}.shp'
        destination_folder_dsm= f'input/AHN/{main_river}/{city}/DSM'
        os.makedirs(destination_folder_dsm, exist_ok=True)
        destination_folder_dtm = f'input/AHN/{main_river}/{city}/DTM'
        os.makedirs(destination_folder_dtm, exist_ok=True)

        download_AHN_tiles(json_path_dsm,river, destination_folder_dsm)
        download_AHN_tiles(json_path_dtm, river , destination_folder_dtm)

# ahn_data(cities, main_river)
for idx, main_river in enumerate(main_rivers):
    ahn_data(cities[idx], main_river)

ahn_for_loneliness(lonelier_rivers)

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

# BGT--------------------------------------------------------------------------------------------------
def download_bgt_data(river, shapefile_path, output_dir, buffer_distance=100):
    try:
        # Read and buffer the shapefile
        gdf = gpd.read_file(shapefile_path)
        buffered_gdf = gdf.buffer(buffer_distance)
        buffered_union = buffered_gdf.unary_union
        wkt_polygon = buffered_union.wkt

        # API configuration
        api_url = "https://api.pdok.nl/lv/bgt/download/v1_0"
        api_url_p = urlparse(api_url)
        base_url = f"{api_url_p.scheme}://{api_url_p.netloc}"
        full_custom_url = f"{api_url}/full/custom"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Feature types from your original function
        featuretypes = [
            "bak", "begroeidterreindeel", "bord", "buurt", "functioneelgebied",
            "gebouwinstallatie", "installatie", "kast", "kunstwerkdeel", "mast",
            "onbegroeidterreindeel", "ondersteunendwaterdeel", "ondersteunendwegdeel",
            "ongeclassificeerdobject", "openbareruimte", "openbareruimtelabel",
            "overbruggingsdeel", "overigbouwwerk", "overigescheiding", "paal",
            "pand", "plaatsbepalingspunt", "put", "scheiding", "sensor", "spoor",
            "stadsdeel", "straatmeubilair", "tunneldeel", "vegetatieobject",
            "waterdeel", "waterinrichtingselement", "waterschap", "wegdeel",
            "weginrichtingselement", "wijk"
        ]

        # Prepare request data - using 'gmllight' format
        data = {
            "featuretypes": featuretypes,
            "format": "gmllight",
            "geofilter": wkt_polygon
        }

        # Submit initial request
        print("Submitting download request...")
        response = requests.post(
            full_custom_url,
            headers=headers,
            data=json.dumps(data)
        )

        if response.status_code != 202:
            print(f"Error creating custom download: {response.status_code} - {response.text}")
            print(f"Response content: {response.text}")
            return None

        # Get request ID and status URL
        response_object = response.json()
        status_path = response_object["_links"]["status"]["href"]
        download_request_id = response_object["downloadRequestId"]
        status_url = f"{base_url}{status_path}"

        # Poll for completion
        while True:
            response = requests.get(status_url)
            status_object = response.json()
            status = status_object["status"]
            print(f"Status: {status}")

            if status == "COMPLETED":
                download_path = status_object["_links"]["download"]["href"]
                download_url = f"{base_url}{download_path}"
                break
            elif status == "PENDING":
                print(f"Progress: {status_object.get('progress', 'unknown')}%")
            elif status == "FAILED":
                print("Download request failed")
                print(f"Status response: {status_object}")
                return None

            time.sleep(10)

        # Download and extract the zip file
        if download_url:
            # Create output directory
            # output_dir = os.path.dirname(output_bgt)
            # os.mkdir(output_bgt, exist_ok=True)
            zip_path = os.path.join(output_dir, f"{river}.zip")
            extract_dir = os.path.join(output_dir, f"{river}")

            # Download zip file
            print(f"Downloading file to {zip_path}")
            response = requests.get(download_url)
            with open(zip_path, "wb") as f:
                f.write(response.content)

            # Extract zip file
            print(f"Extracting files to {extract_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Remove zip file after extraction
            os.remove(zip_path)

            print("Download and extraction completed successfully")
            return extract_dir

        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Run by looping over all rivers
lonely_rivers = ['lek', 'RijnSchieKanaal', 'schie', 'vliet']
main_rivers = ['waal', 'ar']
their_cities = [['nijmegen'], ['amsterdam']]
# for idx, main in enumerate(main_rivers):
#     for city in their_cities[idx]:
#         river_shp = f"input/river/{main}/{city}/{city}.shp"
#         output_bgt = f"input/BGT"

# for river in lonely_rivers:
#     river_shp = f"input/river/{river}/{river}.shp"
#     output_bgt = f"input/BGT"
    # os.mkdir(output_bgt, exist_ok=True)
    #     result = download_bgt_data(city, river_shp, output_bgt)
    #     if result:
    #         print(f"Data extracted to: {result}")
    #         # Print list of downloaded files
    #         for file in os.listdir(result):
    #             if file.endswith('.gml'):
    #                 print(f"- {file}")
    #     else:
    #         print("Download failed")