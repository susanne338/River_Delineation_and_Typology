"""
Adds elev_dsm, imperviousness, landuse, visibility, and flood to points shapefile
"""
import os
import fiona
import pyproj
import rasterio
import pandas as pd
from rasterio.transform import rowcol
from tqdm import tqdm
from rasterio.windows import from_bounds
import geopandas as gpd
import numpy as np
from data_retrieval import run_data_retrieval
from visibility import combine_viewsheds

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

def process_landuse(points_gdf, gml_folder):
    """
    Add landuse information to points GeoDataFrame from gpkg files from BGT data.

    Args:
        points: geodataframe of points
        gpkg_file: Folder containing .gpkg files

    Returns:
        GeoDataFrame: Updated points with landuse information
    """

    # Initialize landuse column
    points_gdf = points_gdf.copy(deep = True)
    points_gdf = points_gdf.to_crs(epsg=28992)
    points_gdf['landuse'] = [[] for _ in range(len(points_gdf))]

    # counter for points that get a landuse
    points_with_landuse = set()

    bbox = points_gdf.total_bounds
    # print('I have found the bbox of points: ', bbox)
    # print('total number of points: ', len(points_gdf))

    # Loop through gmlfiles
    for i, gml_file in enumerate(os.listdir(gml_folder)):
        if not gml_file.endswith('.gml'):
            continue

        gml_path = os.path.join(gml_folder, gml_file)
        print(f'\nProcessing file: {gml_file}')
        original_crs = 'epsg:28992'
        # with fiona.open(gml_path) as src:
        #     print(f"Fiona driver: {src.driver}")
        #     print(f"Fiona CRS: {src.crs}") # No crs for some reason
        #     print(f"Fiona schema: {src.schema}")
        #     print(f"Fiona bounds: {src.bounds}")

        for layer_name in fiona.listlayers(gml_path):
            try:
                print(f'Processing layer: {layer_name} in file: {gml_file}')
                landuse_gdf = gpd.read_file(gml_path, layer=layer_name)

                if len(landuse_gdf) == 0:
                    print(f'Layer {layer_name} is empty')
                    continue

                landuse_filtered = landuse_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                # print('I filtered the landuse to bbox')

                if landuse_filtered.crs is None:
                    landuse_filtered.set_crs(original_crs, inplace=True)

                if len(landuse_filtered) == 0:
                    # print(f'No features in bbox for layer {layer_name}')
                    continue

                # Ensure CRS matches
                if landuse_filtered.crs != points_gdf.crs:
                    landuse_filtered = landuse_filtered.to_crs(points_gdf.crs)
                    # points_gdf = points_gdf.to_crs(landuse_filtered.crs)
                    # print('crs landuse ', landuse_filtered.crs)
                # print('points bbox now is ', points_gdf.total_bounds)
                # print('landuse filtered columns are ', landuse_filtered.columns)
                # print("Filtered landuse count:", len(landuse_filtered))
                    # Check for valid geometries
                invalid_geoms = ~landuse_filtered.geometry.is_valid
                if invalid_geoms.any():
                    print(f'Found {invalid_geoms.sum()} invalid geometries. Attempting to fix...')
                    landuse_filtered.geometry = landuse_filtered.geometry.buffer(0)
                # Process each point
                # I get a warning here that says I should work with a copy but I don't want to I want to write to this specific one no?
                joined_gdf = gpd.sjoin(points_gdf, landuse_filtered, how="left", predicate="intersects")
                matches_in_layer = len(joined_gdf[~joined_gdf.index_right.isna()])
                print(f'Points with matches in this layer: {matches_in_layer}')
                # It's this line: that points_gdf is being altered instead of a copy of it
                # points_gdf.loc[joined_gdf.index, 'landuse'] = layer_name
                # For each point that intersects with a landuse polygon, append the layer_name
                for idx in joined_gdf.index[~joined_gdf.index_right.isna()]:
                    if layer_name not in points_gdf.at[idx, 'landuse']:
                        points_gdf.at[idx, 'landuse'].append(layer_name)
                        points_with_landuse.add(idx)

            except Exception as e:
                print(f'Error processing layer {layer_name}: {str(e)}')
                continue
                # Process each point for CBS data
                # joined_gdf = gpd.sjoin(points_gdf, landuse_filtered, how="left", predicate="intersects")
                # points_gdf['landuse'] = joined_gdf['description']

    # Convert landuse lists to strings for easier handling
    points_gdf['landuse'] = points_gdf['landuse'].apply(lambda x: '; '.join(x) if x else None)
    # Final statistics
    total_points = len(points_gdf)
    points_with_values = len(points_with_landuse)
    points_without_values = total_points - points_with_values
    # Create a summary of found landuse types
    landuse_summary = points_gdf['landuse'].value_counts()
    print("\nLanduse distribution:")
    print(landuse_summary)

    if points_without_values > 0:
        missing_points = points_gdf[points_gdf['landuse'].isna()].head()
        print('\nSample of points without landuse:')
        print('Coordinates:')
        print(missing_points.geometry.to_string())

    return points_gdf


def add_landuse_to_shapefile(shapefile_path, gml_folder):
    """
    Reads a shapefile, adds landuse information from gpkg files, and saves the updated shapefile.

    Parameters:
        shapefile_path: Path to the shapefile containing all cross-section points
        gml_folder:
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    print('The shapefile has been read! its columns are: ', gdf.columns)

    # Add landuse information
    gdf = process_landuse(gdf, gml_folder)
    print("landuse is added to the geodataframe!")

    # Save updated shapefile
    gdf.to_file(shapefile_path)
    print(f"Updated shapefile saved to: {shapefile_path}")


def extract_unique_landuses(shapefile_path, output_file):
    """
    Extract unique landuse types from a shapefile, properly splitting combined values
    and assigning unique indices to individual landuse types.
    """
    # Read the shapefile
    points_gdf = gpd.read_file(shapefile_path)

    # Create a list to store all individual landuse types
    individual_landuses = []

    # Process each landuse entry
    for landuse in points_gdf['landuse'].dropna():
        # Split by semicolon and strip whitespace
        parts = [part.strip() for part in str(landuse).split(';')]
        # Add each individual part to our list
        individual_landuses.extend(parts)

    # Get unique values and sort them
    unique_landuses = sorted(list(set(individual_landuses)))

    # Create DataFrame with index
    landuses_df = pd.DataFrame({
        'landuse': unique_landuses,
        'index': range(len(unique_landuses))
    })

    # Print some debug info
    print("\nFinal DataFrame:")
    print(landuses_df)
    print(f"\nTotal unique landuse types: {len(landuses_df)}")

    # Save to CSV
    landuses_df.to_csv(output_file, index=False)
    print(f"Unique landuses saved to: {output_file}")

    return landuses_df


# IMPERVIOUSNESS AND VISIBILITY
def normalize_crs(crs):
    """
    Normalize CRS to EPSG:28992 if it's any variant of Amersfoort RD New
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
    Add raster values as a new column to a shapefile.
    """
    # Load raster data
    raster_data, transform, bounds, raster_crs = load_raster_data(raster_path)

    # Load shapefile data
    gdf = gpd.read_file(shapefile_path)

    # Normalize the shapefile CRS
    gdf.crs = normalize_crs(gdf.crs)

    # Print CRS information
    # print(f"Normalized Shapefile CRS: {gdf.crs}")
    # print(f"Normalized Raster CRS: {raster_crs}")

    # Print bounds information
    # print(f"\nRaster bounds: {bounds}")
    # print(f"Points extent: {gdf.total_bounds}")

    # Verify points overlap with raster
    points_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    if not (bounds.left <= points_bounds[2] and points_bounds[0] <= bounds.right and
            bounds.bottom <= points_bounds[3] and points_bounds[1] <= bounds.top):
        print("WARNING: Points extent does not overlap with raster extent!")

    # Check if column exists
    if column_name in gdf.columns and not overwrite:
        raise ValueError(f"Column {column_name} already exists and overwrite=False")

    # Process points in smaller chunks to avoid memory issues
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

    # Add results to the GeoDataFrame
    gdf[column_name] = results

    # Save updated shapefile
    gdf.to_file(shapefile_path)
    print(f"\nUpdated shapefile saved to: {shapefile_path}")

    # Print summary statistics
    valid_values = gdf[column_name].dropna()
    print("\nSummary statistics for sampled values:")
    print(f"Total points: {len(gdf)}")
    print(f"Valid values: {len(valid_values)}")
    print(f"Invalid/out of bounds: {len(gdf) - len(valid_values)}")
    if len(valid_values) > 0:
        print(f"Min value: {valid_values.min()}")
        print(f"Max value: {valid_values.max()}")
        print(f"Mean value: {valid_values.mean():.2f}")


def run_parameters(river, city):
    # run_data_retrieval(river, city)

    points_shp = f"output/cross_sections/{river}/{city}/final/points.shp"
    viewshed_file = f'output/visibility/{river}/{city}/combined_viewshed.tif'
    tiles_folder_dsm = f"input/AHN/{river}/{city}/DSM"
    bgt_folder = f"input/BGT/{river}/{city}"
    unique_landuses_output = f"output/unique_landuses/{city}.csv"
    os.makedirs('output/unique_landuses', exist_ok=True)
    flood_folder = "input/flood/middelgrote_kans"
    imperv_raster = f"input/imperviousness/MERGED_reproj_28992.tif"

    add_landuse_to_shapefile(points_shp, bgt_folder)
    extract_unique_landuses(points_shp, unique_landuses_output)

    add_elevation_from_tiles(points_shp,flood_folder , 'flood_dept')
    add_elevation_from_tiles(points_shp, tiles_folder_dsm, 'elev_dsm')

    add_raster_column(shapefile_path=points_shp, raster_path=imperv_raster, column_name='imperv')

    viewshed_dir= f'output/visibility/{river}/{city}/viewsheds'
    visibility_dir = f'output/visibility/{river}/{city}'
    output_file = os.path.join(visibility_dir, 'combined_viewshed.tif')
    combine_viewsheds(viewshed_dir, output_file)
    add_raster_column(shapefile_path=points_shp, raster_path=viewshed_file, column_name='visible')


if __name__ == "__main__":
    run_parameters('dommel', 'gestel')