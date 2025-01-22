"""
DSM/DTM
--> split segments into 10 pieces
--> process elevation per piece
--> can write script to do it overnight? but not for visiblity analysis

Use spatial indexing?
Imperviousness
Flood
Dijken
BGT
"""
# def load_progress(output_dir):
#     """
#     Safely load progress from a JSON file, creating it if it doesn't exist.
#
#     Args:
#         output_dir: Directory where the progress file is stored
#
#     Returns:
#         dict: Progress data or empty dict if no valid progress exists
#     """
#     progress_file = os.path.join(output_dir, "download_progress.json")
#     try:
#         if os.path.exists(progress_file):
#             with open(progress_file, 'r') as f:
#                 return json.load(f)
#         return {}
#     except json.JSONDecodeError:
#         print("Warning: Progress file exists but contains invalid JSON. Starting fresh.")
#         return {}
#     except Exception as e:
#         print(f"Error loading progress file: {e}. Starting fresh.")
#         return {}


# Import BGT data for all urban areas. Saved per featuretype.

import os
import json

import pyproj
import rasterio
import requests
import zipfile
import geopandas as gpd
from urllib.parse import urlparse
import time
from datetime import datetime
import pandas as pd
from rasterio.transform import rowcol
import os
import glob
import geopandas as gpd
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Downloading BGT data--------------------------------------------------------------------------------------------------
def download_bgt_data(name, polygon_gdf, output_dir, buffer_distance=100):
    try:
        # Read and buffer the shapefile
        # gdf = gpd.read_file(shapefile_path)
        # buffered_gdf = gdf.buffer(buffer_distance)
        # buffered_union = buffered_gdf.unary_union
        # wkt_polygon = buffered_union.wkt
        wkt_polygon = polygon_gdf.wkt

        # API
        api_url = "https://api.pdok.nl/lv/bgt/download/v1_0"
        api_url_p = urlparse(api_url)
        base_url = f"{api_url_p.scheme}://{api_url_p.netloc}"
        full_custom_url = f"{api_url}/full/custom"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }


        featuretypes = [
            "begroeidterreindeel", "kunstwerkdeel",
            "onbegroeidterreindeel", "ondersteunendwaterdeel", "ondersteunendwegdeel",
            "overbruggingsdeel", "pand", "scheiding", "spoor", "vegetatieobject",
            "waterdeel", "waterinrichtingselement", "wegdeel"
        ]
        # remove "plaatsbepalingspunt"

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
            zip_path = os.path.join(output_dir, f"{name}.zip")
            # extract_dir = os.path.join(output_dir, f"{city_name}")

            # Download zip file
            print(f"Downloading file to {zip_path}")
            response = requests.get(download_url)
            with open(zip_path, "wb") as f:
                f.write(response.content)

            # Extract zip file
            print(f"Extracting files to {output_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

            # Remove zip file after extraction
            os.remove(zip_path)

            print("Download and extraction completed successfully")
            return output_dir

        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def merge_and_clean_featuretype(feature_type, output_folder):
    """
    Merges all files of a specific feature type and removes temporal data.

    Args:
        feature_type: Name of the feature type (e.g., 'wegdeel')
        output_folder: Root folder containing the data
    """
    all_files = glob.glob(os.path.join(output_folder, "*", f"{feature_type}.gml"))

    if not all_files:
        return

    merged_gdf = gpd.GeoDataFrame()
    for file in all_files:
        try:
            gdf = gpd.read_file(file)
            merged_gdf = pd.concat([merged_gdf, gdf], ignore_index=True)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    if not merged_gdf.empty:
        if 'objectEindTijd' in merged_gdf.columns:
            print(f"Filtering temporal data for {feature_type}")
            print(f"Original size: {len(merged_gdf)}")
            merged_gdf = merged_gdf[merged_gdf['objectEindTijd'].isna()]
            print(f"Filtered size: {len(merged_gdf)}")

        merged_output = os.path.join(output_folder, f"{feature_type}_merged.gml")
        merged_gdf.to_file(merged_output, driver="GML")

        # for file in all_files:
        #     try:
        #         os.remove(file)
        #     except Exception as e:
        #         print(f"Error removing {file}: {e}")


def merge_featuretypes(output_folder, max_workers=4):
    """
    Merges and cleans all feature types in parallel.

    Args:
        output_folder: Root folder containing the data
        max_workers: Number of parallel workers
    """
    first_subfolder = next(os.walk(output_folder))[1][0]
    feature_types = [os.path.splitext(os.path.basename(f))[0]
                    for f in glob.glob(os.path.join(output_folder, first_subfolder, "*.gml"))]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(merge_and_clean_featuretype, feature_type, output_folder)
            for feature_type in feature_types
        ]

        for future in tqdm(futures, desc="Processing feature types"):
            future.result()


def run_bgt_retrieval(buffered_rivers_path, output_folder):
    """
    Downloads BGT data for individual urban areas.

    Args:
        urban_areas_path: Path to the urban areas shapefile
        output_folder: Root folder for storing downloaded data
    """
    print(f"Reading urban areas from {buffered_rivers_path}")
    urban_gdf = gpd.read_file(buffered_rivers_path)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing {len(urban_gdf)} individual polygons...")
    for idx, row in tqdm(urban_gdf.iterrows(), total=len(urban_gdf), desc="Processing polygons"):
        # Create a unique identifier for each polygon
        polygon_id = f"polygon_{idx}"
        polygon_folder = os.path.join(output_folder, polygon_id)

        # Skip if already processed
        if os.path.exists(polygon_folder) and len(os.listdir(polygon_folder)) > 0:
            print(f"Data for {polygon_id} already exists in {polygon_folder}, skipping...")
            continue

        try:
            # Create folder for this polygon
            os.makedirs(polygon_folder, exist_ok=True)

            # Download data for individual polygon
            success = download_bgt_data(polygon_id, row.geometry, polygon_folder)

            if not success:
                print(f"Failed to download data for {polygon_id}")

        except Exception as e:
            print(f"Error processing {polygon_id}: {e}")
            continue

    # After all downloads are complete, merge and clean feature types
    # merge_featuretypes(output_folder)


# Extracting data from a raster .tif file--------------------------------------------------------------------------------
def normalize_crs(crs):
    """
    Normalize CRS to EPSG:28992 if it's any variant of Amersfoort RD New
    (I was having problems with some 'local' crs)
    """
    if crs:
        # Check for various forms of Amersfoort RD New
        if any(marker in str(crs).upper() for marker in ['AMERSFOORT', 'RD NEW', '28992']):
            return pyproj.CRS.from_epsg(28992)
    return crs

def load_raster_data(raster_path):
    """
    Load raster data and return necessary components for processing.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # Read the first band
        transform = src.transform
        bounds = src.bounds
        crs = normalize_crs(src.crs)
        print(f"Raster transform: {transform}")
        print(f"Raster shape: {data.shape}")
    return data, transform, bounds, crs

def check_raster_value(location, raster_data, transform, bounds):
    """
    Check raster value at a given point location.
    """
    if location is None or location.is_empty or location.geom_type != 'Point':
        print(f"Invalid location: {location}")
        return np.nan

    x, y = location.x, location.y

    # Check if point is within raster bounds with a small buffer
    buffer = 1.0  # 1 meter buffer
    if not (bounds.left - buffer <= x <= bounds.right + buffer and
            bounds.bottom - buffer <= y <= bounds.top + buffer):
        print(f"Point ({x}, {y}) is outside raster bounds: {bounds}")
        return np.nan

    try:
        # Convert coordinates to pixel indices using rasterio's rowcol function
        row, col = rowcol(transform, x, y)

        # Convert to integers
        row, col = int(row), int(col)

        # Debug information
        # print(f"Point coordinates: ({x}, {y})")
        # print(f"Pixel coordinates: (row={row}, col={col})")
        # print(f"Raster shape: {raster_data.shape}")

        # Ensure indices are within array bounds
        if 0 <= row < raster_data.shape[0] and 0 <= col < raster_data.shape[1]:
            value = raster_data[row, col]
            # print(f"Sampled value: {value}")
            return value
        else:
            print(f"Computed pixel coordinates ({row}, {col}) are outside raster dimensions {raster_data.shape}")
            return np.nan

    except Exception as e:
        print(f"Error processing point ({x}, {y}): {str(e)}")
        return np.nan

def compute_raster_value(row, raster_data, transform, bounds):
    """
    Compute raster value for a GeoDataFrame row.
    """
    location = row['geometry']
    return check_raster_value(location, raster_data, transform, bounds)

def add_raster_column(shapefile_path, raster_path, column_name, overwrite=True):
    """

    Args:
        shapefile_path (str): Path to points shapefile
        raster_path (str): Path to raster .tif file
        column_name (str): Name of column to be added
        overwrite:

    Returns:

    """

    raster_data, transform, bounds, raster_crs = load_raster_data(raster_path)
    gdf = gpd.read_file(shapefile_path)

    # Normalize the shapefile CRS
    gdf.crs = normalize_crs(gdf.crs)

    # Verify points overlap with raster
    points_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    if not (bounds.left <= points_bounds[2] and points_bounds[0] <= bounds.right and
            bounds.bottom <= points_bounds[3] and points_bounds[1] <= bounds.top):
        print("WARNING: Points extent does not overlap with raster extent!")

    # Check if column exists
    if column_name in gdf.columns and not overwrite:
        raise ValueError(f"Column {column_name} already exists and overwrite=False")

    # Process points in smaller chunks
    chunk_size = 1000
    num_chunks = len(gdf) // chunk_size + (1 if len(gdf) % chunk_size else 0)

    results = []
    for i in tqdm(range(num_chunks), desc=f"Processing chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(gdf))
        chunk = gdf.iloc[start_idx:end_idx]

        chunk_results = chunk.apply(
            compute_raster_value,
            axis=1,
            raster_data=raster_data,
            transform=transform,
            bounds=bounds
        )
        results.extend(chunk_results)

    gdf[column_name] = results
    gdf.to_file(shapefile_path)
    print(f"\nUpdated shapefile saved to: {shapefile_path}")

    # Print summary statistics
    # valid_values = gdf[column_name].dropna()
    # print("\nSummary statistics for sampled values:")
    # print(f"Total points: {len(gdf)}")
    # print(f"Valid values: {len(valid_values)}")
    # print(f"Invalid/out of bounds: {len(gdf) - len(valid_values)}")
    # if len(valid_values) > 0:
    #     print(f"Min value: {valid_values.min()}")
    #     print(f"Max value: {valid_values.max()}")
    #     print(f"Mean value: {valid_values.mean():.2f}")


# Add elevation from tif tiles
def add_elevation_from_tiles(shapefile_path, tiles_folder, elevation_column_name):
    """
    Adds elevation data from tiles to existing shapefile and updates it.

    Parameters:
    shapefile_path: Path to the existing shapefile with points
    tiles_folder: Folder containing elevation tiles (.tif files)
    elevation_column_name: Name of the column to store elevation values

    Returns:
    GeoDataFrame with added elevation column and count of missing points
    """
    # Read existing shapefile
    points_gdf = gpd.read_file(shapefile_path)
    print(f"Loaded {len(points_gdf)} points from {shapefile_path}")

    # Initialize elevation column with NaN
    points_gdf[elevation_column_name] = np.nan

    # Get list of tif files
    tif_files = [f for f in os.listdir(tiles_folder) if f.endswith(".tif")]
    print(f"List of tif file {tif_files}")
    missing_points = 0
    nodata_points = 0

    for idy, tif_file in tqdm(enumerate(tif_files), total=len(tif_files), desc="Processing points"):
        print(f"name of tif file {tif_file}")
        tif_path = os.path.join(tiles_folder, tif_file)
        elevation_found = False
        with rasterio.open(tif_path) as src:
            tif_bounds = src.bounds
            minx, miny, maxx, maxy = tif_bounds
            for idx, row in points_gdf.iterrows():
                if row.geometry is not None:
                    point = row.geometry

                    if minx <= point.x <= maxx and miny <= point.y <= maxy:
                        try:
                            py, px = src.index(point.x, point.y)
                            window = rasterio.windows.Window(px - 1, py - 1, 2, 2)
                            data = src.read(1, window=window)
                            center_value = data[1, 1]
                            if center_value == src.nodata:
                                is_nodata = True
                            elif data.size > 0:
                                points_gdf.at[idx, elevation_column_name] = float(center_value)
                                elevation_found = True
                                continue

                        except (IndexError, ValueError):
                            print(f"there was an error processing this point in the tile")
                            continue

                        if not elevation_found:
                            missing_points += 1
                            if is_nodata:
                                nodata_points += 1
                                print(f"NoData value found for point {idx} at ({point.x}, {point.y})")
                            else:
                                print(
                                    f"No elevation data found for point {idx} at ({point.x}, {point.y}) - point may be between tiles")

    # Save the updated GeoDataFrame
    points_gdf.to_file(shapefile_path, driver='ESRI Shapefile')
    print(f"\nUpdated shapefile with {elevation_column_name} at: {shapefile_path}")
    print(f"Total missing points: {missing_points}")

    return points_gdf


if __name__ == '__main__':
    flood_raster = "input/flood/middelgrote_kans/MaximaleWaterdiepteNederland_Kaart2.tif"
    imperv_raster = f"input/imperviousness/MERGED_reproj_28992.tif"
    points_shp = "?"
    dsm_folder = "?"
    dtm_folder = "?"
    bgt_folder = "D:/geomatics_thesis/data/bgt"

    run_bgt_retrieval("input/DATA/BGT_rivers/extend_100mbuffer_around_waterdelen_for_bgt_download_NOHOLES.shp", bgt_folder)
    #Figure out how to add bgt data to points

    # add_raster_column(shapefile_path=points_shp, raster_path=imperv_raster, column_name='imperv')
    # add_raster_column(shapefile_path=points_shp, raster_path=flood_raster, column_name='flood')

    #add elevation in QGIS using 'Sample raster values'? or loop over tile names
    # add_elevation_from_tiles(points_shp, dsm_folder, 'dsm')
    # add_elevation_from_tiles(points_shp, dtm_folder, 'dtm')

    #visibility