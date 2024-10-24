"""
Computers parameter table
--> creates points at cross-section and initializes shapefile with all points. Each point has a cross-section id
--> add elevation values to the points in the shapefile
TODO: landuse
"""

import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from rasterio.windows import from_bounds
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import matplotlib.pyplot as plt

# CROSS-SECTION POINTS AND ELEVATION------------------------------------------------------------------------------------
def create_cross_section_points(cross_sections_shapefile, n_points, output_shapefile):
    """
    Creates points along cross-sections and calculates horizontal distances.

    Parameters:
    cross_sections_shapefile: Path to shapefile containing cross-section lines
    n_points: Number of points to create along each cross-section
    output_shapefile: Path where the resulting points shapefile will be saved

    Returns:
    GeoDataFrame containing points with horizontal distances
    """
    # Read cross-sections from shapefile
    cross_sections = gpd.read_file(cross_sections_shapefile)
    print(f"Loaded {len(cross_sections)} cross-sections from {cross_sections_shapefile}")

    all_points = []

    for ind, row in cross_sections.iterrows():
        print('Processing cross-section', ind)

        # Extract start and end coordinates from the LineString geometry
        start_coords = list(row.geometry.coords)[0]
        end_coords = list(row.geometry.coords)[1]

        # Create points along the cross-section
        lon = [start_coords[0]]
        lat = [start_coords[1]]

        for i in np.arange(1, n_points + 1):
            x_dist = end_coords[0] - start_coords[0]
            y_dist = end_coords[1] - start_coords[1]
            point = [(start_coords[0] + (x_dist / (n_points + 1)) * i),
                     (start_coords[1] + (y_dist / (n_points + 1)) * i)]
            lon.append(point[0])
            lat.append(point[1])

        lon.append(end_coords[0])
        lat.append(end_coords[1])

        # Create Point objects and calculate distances
        for i, (x, y) in enumerate(zip(lon, lat)):
            point_geom = Point(x, y)
            h_distance = Point(start_coords).distance(point_geom)
            all_points.append({
                'geometry': point_geom,
                'id': ind,
                'h_distance': h_distance
            })

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(all_points)
    gdf.set_crs(epsg=28992, inplace=True)

    # Save to shapefile
    gdf.to_file(output_shapefile, driver='ESRI Shapefile')
    print(f"Base cross-section points saved to: {output_shapefile}")

    return gdf


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
    missing_points = 0
    nodata_points = 0
    outside_coverage = 0
    points_gdf = gpd.read_file(shapefile_path)
    print(f"Loaded {len(points_gdf)} points from {shapefile_path}")

    # Initialize elevation column with NaN
    points_gdf[elevation_column_name] = np.nan

    # Get list of tif files
    tif_files = [f for f in os.listdir(tiles_folder) if f.endswith(".tif")]

    # Process each cross-section separately
    cross_sections = points_gdf.groupby('id')

    # Loop through each cross-section group
    for cross_section_id, section_gdf in tqdm(cross_sections, desc="Processing cross sections"):
        # Create bounding box for the cross-section
        bounds = section_gdf.total_bounds
        section_bbox = box(*bounds)

        # Find relevant TIF files that intersect with this cross-section
        relevant_tifs = []
        for tif_file in tif_files:
            tif_path = os.path.join(tiles_folder, tif_file)
            with rasterio.open(tif_path) as src:
                tile_bbox = box(*src.bounds)
                if section_bbox.intersects(tile_bbox):
                    relevant_tifs.append(tif_file)


        if not relevant_tifs:
            print(f"Warning: No TIF files found covering cross section {cross_section_id}")
            outside_coverage += len(section_gdf)
            continue

        # Process each point in the cross-section
        for idx, point in section_gdf.iterrows():
            point_x, point_y = point.geometry.x, point.geometry.y
            elevation_found = False
            is_nodata = False

            # Try each relevant TIF file until we find an elevation
            for tif_file in relevant_tifs:
                tif_path = os.path.join(tiles_folder, tif_file)
                with rasterio.open(tif_path) as src:
                    # Check if point is within this tile's bounds
                    if (src.bounds.left <= point_x <= src.bounds.right and
                            src.bounds.bottom <= point_y <= src.bounds.top):

                        try:
                            # Convert point coordinates to pixel coordinates
                            py, px = src.index(point_x, point_y)

                            # Read the elevation value
                            window = rasterio.windows.Window(px - 1, py - 1, 3, 3)
                            data = src.read(1, window=window)

                            # Check center pixel
                            center_value = data[1, 1]
                            if center_value == src.nodata:
                                is_nodata = True
                            elif data.size > 0:
                                points_gdf.at[idx, elevation_column_name] = float(center_value)
                                elevation_found = True
                                break

                        except (IndexError, ValueError):
                            continue

            if not elevation_found:
                missing_points += 1
                if is_nodata:
                    nodata_points += 1
                    print(f"NoData value found for point {idx} at ({point_x}, {point_y})")
                else:
                    print(
                        f"No elevation data found for point {idx} at ({point_x}, {point_y}) - point may be between tiles")

    # Save the updated GeoDataFrame
    points_gdf.to_file(shapefile_path, driver='ESRI Shapefile')
    print(f"\nUpdated shapefile with {elevation_column_name} at: {shapefile_path}")
    print(f"Total missing points: {missing_points}")

    return points_gdf

# RUN SCRIPTS
# create the corss-section points and initialize the shapefile
# create_cross_section_points(
#     cross_sections_shapefile='thesis_output/cross_sections/cross_sections_longest.shp',
#     n_points=100,
#     output_shapefile='output/parameters/parameters_longest.shp'
# )

# Add DTM elevation to the shapefile
# add_elevation_from_tiles(
#     shapefile_path='output/parameters/parameters_longest.shp',
#     tiles_folder='thesis_output/AHN_tiles_DTM',
#     elevation_column_name='elev_dtm'
# )

# Add DSM elevation to the same shapefile
# add_elevation_from_tiles(
#     shapefile_path='output/parameters/parameters_longest.shp',
#     tiles_folder='thesis_output/AHN_tiles_DSM',
#     elevation_column_name='elev_dsm'
# )

# CHECK DATA IN PARAMETER TABLE----------------------------------------------------------------------------------------
def export_cross_sections_to_csv(shapefile_path, output_folder):
    """
    Reads a shapefile containing all cross-section points and exports separate CSV files
    for each cross-section.

    Parameters:
    shapefile_path: Path to the shapefile containing all cross-section points
    output_folder: Folder where individual CSV files will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Convert geometry to x,y coordinates
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y

    # Group by cross_section_id and save separate files
    for cross_section_id, group in gdf.groupby('id'):
        # Select relevant columns and convert to DataFrame
        csv_df = pd.DataFrame({
            'x': group.x,
            'y': group.y,
            'h_distance': group.h_distance,
            'elev_dtm': group.elev_dtm if 'elev_dtm' in group.columns else None,
            'elev_dsm': group.elev_dsm if 'elev_dsm' in group.columns else None
        #     There is a problem here: nodata is either None or '3.402823466385289e+23
        })

        # Save to CSV
        output_path = os.path.join(output_folder, f'cross_section_{cross_section_id}.csv')
        csv_df.to_csv(output_path, index=False)
        print(f"Saved cross-section {cross_section_id} to {output_path}")


def analyze_cross_section_data(shapefile_path):
    """
    Provides a summary of the data in the cross-sections shapefile.

    Parameters:
    shapefile_path: Path to the shapefile containing all cross-section points
    """
    gdf = gpd.read_file(shapefile_path)

    print("\nCross-sections summary:")
    print(f"columns: ", gdf.columns)
    print(f"Total number of cross-sections: {len(gdf.id.unique())}")
    print(f"Total number of points: {len(gdf)}")
    print("\nAvailable columns:", list(gdf.columns))

    # Group by cross-section and show points per cross-section
    points_per_cs = gdf.groupby('id').size()
    print("\nPoints per cross-section:")
    print(points_per_cs)

    # Show a sample of the data
    print("\nSample of the data (first 5 points):")
    sample_df = gdf.copy()
    sample_df['x'] = sample_df.geometry.x
    sample_df['y'] = sample_df.geometry.y
    print(sample_df.drop('geometry', axis=1).head())


def read_single_cross_section(shapefile_path, cross_section_id):
    """
    Extracts data for a single cross-section from the shapefile.

    Parameters:
    shapefile_path: Path to the shapefile containing all cross-section points
    cross_section_id: ID of the cross-section to extract

    Returns:
    DataFrame containing the cross-section data
    """
    gdf = gpd.read_file(shapefile_path)

    # Filter for specific cross-section
    cs_data = gdf[gdf.id == cross_section_id].copy()

    # Add x,y coordinates as columns
    cs_data['x'] = cs_data.geometry.x
    cs_data['y'] = cs_data.geometry.y

    return cs_data.drop('geometry', axis=1)

# Look at parameter shapefile
# analyze_cross_section_data('output/parameters/parameters_longest.shp')

# Export all cross-sections to individual CSV files
# export_cross_sections_to_csv('output/parameters/parameters_longest.shp', 'output/parameters/csv')
#
# gdf = gpd.read_file('output/parameters/parameters_longest.shp')
# missing_dsm = gdf['elev_dsm'].isnull().sum()
# print(f"Number of missing values in 'elev_dsm': {missing_dsm}")

# Check for missing values in the 'elev_dtm' column
# missing_dtm = gdf['elev_dtm'].isnull().sum()
# print(f"Number of missing values in 'elev_dtm': {missing_dtm}")

# PLOT
def plot_csv_parameters(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)  # Replace 'your_file.csv' with your actual file path

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Scatter plot for elev_dtm
    plt.scatter(df['h_distance'], df['elev_dtm'], color='blue', label='elev_dtm', alpha=0.6)

    # Scatter plot for elev_dsm
    plt.scatter(df['h_distance'], df['elev_dsm'], color='red', label='elev_dsm', alpha=0.6)

    # Add vertical lines for each elev_dsm point
    for index, row in df.iterrows():
        plt.axvline(x=row['h_distance'], color='red', linestyle='-', alpha=0.1)

    # Adding labels and title
    plt.xlabel('Horizontal Distance (h_distance)')
    plt.ylabel('Elevation')
    plt.title('Elevation Comparison')
    plt.legend()

    # Show the plot
    plt.grid()
    plt.tight_layout()
    plt.show()

# plot_csv_parameters('output/parameters/csv/cross_section_0.csv')

# LANDUSE---------------------------------------------------------------------------------------------------------------