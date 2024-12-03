"""
This script computes the cross-sectional lines, begin point and end point, along a shapefile line, in this case a river.
First it computes short crosssectional lines crossing the riverline with a smaller width. This is to find the embankment points.
Then cross-sections are computed again, but now for wider sections, so adding up the river width.
1. Compute cross-sections for pre-processing. Get midpoints shapefile of the river. Get preprocess cs shapefile.
2. Compute points along these cross-sections with given interval. Get preprocess points shapefile
3. Get elevation from DTM tiles to points. Find embankment points and river width. Add to midpoints shapefile
4. Compute cross-sections again but now with riverwidth added to the width (from midpoints file)
5. Compute points along these cross-sections with computed interval (0.5)
5. Get elevation values from DSM and DTM
6. Get parameters 100 year flood depth, landuse, imperviousness, and visibility and add to points shapefile
7. Get parameter building intersection and add to midpoints shapefile

Using these embankment points, we compute lateral sections, only extending from this point, for the left and right side of the river. These are the sections we use in our further analysis.
The preprocess cross-sections and points can be removed then.


Adds columns to midpoints.shp
Index(['FID', 'left', 'right', 'max', 'height', 'width', 'geometry'], dtype='object')
Adds columns to points_shp
columns: Index(['id', 'h_distance', 'elev_dtm', 'elev_dsm', 'flood_dept', 'landuse', 'imperv', 'visible', 'geometry'],
dtype='object')

TODO: clean up
TODO: build in removal of preprocess files
"""
import os
from itertools import islice

import fiona
import pyproj
import rasterio
import pandas as pd
import statistics
from rasterio.transform import rowcol
from shapely import MultiLineString, Polygon, MultiPolygon
from tqdm import tqdm
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
from shapely.geometry import Point, box
from concurrent.futures import ProcessPoolExecutor, as_completed
# TODO: i changed cross_section_extraction to try

# Get cross_sections intersecting with river---------------------------------------------------------------------------
def cross_section_extraction(river_file, interval, width, output_file_cs, output_river_mid):
    """
    Get cross-sections
    :param output_file_cs: output shapefile for all cross-sections
    :param width: width of total cross-section
    :param interval: width between cross-sections
    :param river: River geodataframe
    :param output_river_mid: Output file for river midpoints
    :return: gdf
    """
    river = gpd.read_file(river_file)
    projected_gdf = river.to_crs(epsg=28992)
    cross_sections = []
    river_points = []
    total_length = 0

    # starting_point = 0

    for index, row in projected_gdf.iterrows():
        riverline = row['geometry']

        print("index, riverline length: ", index, riverline.length)
        # distance = 0
        total_length += riverline.length

        for distance_along_line in np.arange(0, riverline.length, interval):
            cross_section, point_on_line = get_perpendicular_cross_section(riverline, distance_along_line, width)
            cross_sections.append(cross_section)
            river_points.append(point_on_line)
            # distance = distance_along_line

        # starting_point = interval - (riverline.length - distance)

    # Save cross-sections to a Shapefile (.shp)
    gdf = gpd.GeoDataFrame(geometry=cross_sections)
    gdf.set_crs(epsg=28992, inplace=True)
    gdf.to_file(output_file_cs, driver='ESRI Shapefile')

    # Save river midpoints to shapefile (.shp)
    river_points_gdf = gpd.GeoDataFrame(geometry=river_points)
    river_points_gdf.set_crs(epsg=28992, inplace=True)
    river_points_gdf.to_file(output_river_mid, driver='ESRI Shapefile')
    print(f"total length river {total_length}")
    return


def get_perpendicular_cross_section(line, distance_along_line, width):
    """
    Generates a cross-Section perpendicular to a riverline at a given distance along the line.
    """
    # Get the first point (at distance_along_line)
    point_on_line = line.interpolate(distance_along_line)  # Interpolated point on the riverline

    # Get another point slightly ahead for tangent calculation
    small_distance = 0.001  # A very small distance ahead (you can adjust this)
    if distance_along_line + small_distance <= line.length:
        point_ahead = line.interpolate(distance_along_line + small_distance)
    else:
        point_ahead = line.interpolate(line.length)  # If near the end, use the last point on the line

    # Extract coordinates from the points
    x0, y0 = point_on_line.x, point_on_line.y
    x1, y1 = point_ahead.x, point_ahead.y

    # Calculate the angle of the tangent using arctan2
    angle = np.arctan2(y1 - y0, x1 - x0)

    # create a perpendicular cross-section with the given width
    dx = width / 2 * np.sin(angle)  # X displacement (perpendicular)
    dy = width / 2 * np.cos(angle)  # Y displacement (perpendicular)

    # Two points to define the cross-section
    p1 = Point(x0 + dx, y0 - dy)  # One side of the cross-section
    p2 = Point(x0 - dx, y0 + dy)  # Other side of the cross-section

    return LineString([p1, p2]), point_on_line


# Get points along cross-section, and h_distance. Add elevation to each point from DTM----------------------------------
def create_cross_section_points(cross_sections_shapefile, width, output_shapefile):
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
    n_points= 2 * width

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


# Get max elevation of the embankment for viewshed analysis. Get embankment points.-------------------------------------
def extract_max_elevation_values(points, midpoints_file, embankments_file_left, embankments_file_right,
                                 user_defined_height, embankments_shp):
    """
    Extracts the elevation values of the embankments, and selects the max value for the viewshed analysis
    This max value is then added up with user_defined_height.
    Takes embankment h_dist and selects max, multiplies this my 2, then add this as 'width' to add to radius in viewshed analysis
    Args:
        shapefile: points shapefile
        midpoints_file: cross-section midpoint file that is altered
        user_defined_height: viewpoint height to peform viewshed analysis from

    Returns: saves values left, right (embankment), max, and (viewpoint) height to midpoints_file
    TODO: now when the river extend my buffer, no boundary point gets added. Fix to edge of buffer, or last point
    """
    points = gpd.read_file(points)
    midpoints = gpd.read_file(midpoints_file)

    # Initialize new columns in midpoints for output
    midpoints['left'] = np.nan
    midpoints['right'] = np.nan
    midpoints['max'] = np.nan
    midpoints['height'] = np.nan
    midpoints['width'] = np.nan  # Extra width for viewshed

    # Initialize embankments points
    embankment_pts = []
    distance_list = []
    # embankment_points_left = gpd.GeoDataFrame(columns=['geometry', 'h_distance'])
    # embankment_points_left['geometry'] = np.nan
    # embankment_points_left['h_distance'] = np.nan
    # embankment_points_left.set_crs("EPSG:28992", inplace=True)
    # embankment_points_right = gpd.GeoDataFrame(columns=['geometry', 'h_distance'])
    # embankment_points_right['geometry'] = np.nan
    # embankment_points_right['h_distance'] = np.nan
    # embankment_points_right.set_crs("EPSG:28992", inplace=True)

    # Group by cross-section ID
    grouped = points.groupby('id')
    for idx, row in tqdm(midpoints.iterrows(), total=midpoints.shape[0], desc="Processing midpoints"):
        midpoint_geom = row.geometry
        cross_section_id = row['FID']

        # Select corresponding cross-section points from 'parameters' shapefile
        cross_section_points = grouped.get_group(cross_section_id)

        # Ensure points are sorted by h_distance
        cross_section_points = cross_section_points.sort_values('h_distance').reset_index(drop=True)

        # Calculate approximate midpoint in h_distance
        max_h_distance = cross_section_points['h_distance'].max()
        target_midpoint = max_h_distance / 2 #distance

        # Find the closest point to this target midpoint
        midpoint_idx = (cross_section_points['h_distance'] - target_midpoint).abs().idxmin()

        # Initialize left and right values
        left_value = None
        right_value = None
        right_geom = None
        left_geom = None
        right_hdist = None
        left_hdist = None

        # Search left of the midpoint for the first valid elevation value
        for i in range(midpoint_idx - 1, -1, -1):
            if not pd.isna(cross_section_points.loc[i, 'elev_dtm']):
                left_value = cross_section_points.loc[i, 'elev_dtm']
                left_geom = cross_section_points.loc[i, 'geometry']
                left_hdist = cross_section_points.loc[i, 'h_distance']
                break

        # Search right of the midpoint for the first valid elevation value
        for i in range(midpoint_idx + 1, len(cross_section_points)):
            if not pd.isna(cross_section_points.loc[i, 'elev_dtm']):
                right_value = cross_section_points.loc[i, 'elev_dtm']
                right_geom = cross_section_points.loc[i, 'geometry']
                right_hdist = cross_section_points.loc[i, 'h_distance']
                break

        # Assign left, right, max, and height values to the midpoints DataFrame
        midpoints.at[idx, 'left'] = left_value
        midpoints.at[idx, 'right'] = right_value
        h_dist_midpoint = cross_section_points.loc[midpoint_idx, 'h_distance']
        l_split_h_di = abs(h_dist_midpoint - left_hdist) if left_hdist is not None else None
        r_split_h_di = abs(h_dist_midpoint - right_hdist) if right_hdist is not None else None
        embankment_pts.append([idx, 0, left_geom, left_value, left_hdist, l_split_h_di])
        embankment_pts.append([idx, 1, right_geom, right_value, right_hdist, r_split_h_di])
        # embankment_points_left.at[idx, 'geometry'] = left_geom
        # embankment_points_right.at[idx, 'geometry'] = right_geom
        # embankment_points_left.at[idx, 'h_distance'] = left_hdist
        # embankment_points_right.at[idx, 'h_distance'] = right_hdist
        # embankment_points.at[idx, 'right_geom'] = right_geom

        # Calculate max and height and add to file
        max_value = max(
            filter(None, [left_value, right_value])) if left_value is not None or right_value is not None else None
        midpoints.at[idx, 'max'] = max_value
        midpoints.at[idx, 'height'] = max_value + user_defined_height if max_value is not None else None
        # Add this width to radius of viewshed

        if right_hdist is None or left_hdist is None:
            print("something is wrong. One of the h_dist is None")
            midpoints.at[idx, 'width'] = 0
            # raise ValueError(
            #     f"Invalid distance values at index {i}: right_hdist={right_hdist}, left_hdist={left_hdist}")
        else:
            midpoints.at[idx, 'width'] = abs(right_hdist - left_hdist)
            distance_list.append(abs(right_hdist - left_hdist))
            # print(f"distance {abs(right_hdist - left_hdist)}")

        # Save the result to a new shapefile
    midpoints.to_file(midpoints_file, driver='ESRI Shapefile')
    # embankment_points_left.to_file(embankments_file_left, driver="ESRI Shapefile")
    # print(f"Results saved to {midpoints_file} and {embankments_file_left}")
    # embankment_points_right.to_file(embankments_file_right, driver="ESRI Shapefile")
    # print(f"Results saved to {midpoints_file} and {embankments_file_right}")
    print(f"average width river is {statistics.mean(distance_list)} ")

    gdf_clos = gpd.GeoDataFrame(embankment_pts, columns=['id', 'side', 'geometry', 'height', 'h_dist', 'split_h_di'])
    gdf_clos.set_crs("EPSG:28992", inplace=True)
    gdf_clos.to_file(embankments_shp, driver="ESRI Shapefile")


# RUN PREPROCESS--------------------------------------------------------------------------------------------------------
"""
Get preprocess cross-sections, create the points on these, and add elevation from the DTM. 
Then extract the embankment data and viewpoint height for viewshed analysis
"""


# DATA

# for city in lonely_rivers:
#     river = f"input/river/{city}/{city}.shp"
#     tiles_folder_dtm = f"input/AHN/{city}/DTM"
#     tiles_folder_dsm = f"input/AHN/{city}/DSM"
#     # FILES
#
#     output_cs_preproces = f"output/cross_sections/{city}/preprocess/cross_sections.shp"
#     points_cs_preprocess = f"output/cross_sections/{city}/preprocess/points.shp"
#     river_midpoints = f"output/river/{city}/{city}_mid.shp"
#     embankments_file_left = f"output/embankment/{city}/left/left.shp"
#     embankments_file_right = f"output/embankment/{city}/right/right.shp"
#     embankments_file = f"output/embankment/{city}/embankments.shp"
#     # PARAMETERS
#     interval = 100
#     width_preprocess = 300
#     user_defined_height = 1.75
#     # RUN
#     # cross_section_extraction(river, interval, width_preprocess, output_cs_preproces, river_midpoints)
#     # create_cross_section_points(output_cs_preproces, width_preprocess, points_cs_preprocess)
#     # add_elevation_from_tiles(points_cs_preprocess, tiles_folder_dtm, 'elev_dtm')
#     # extract_max_elevation_values(points_cs_preprocess, river_midpoints, embankments_file_left, embankments_file_right,
#     #                              user_defined_height, embankments_file)
#
# for idx, rv in enumerate(main_rivers):
#     for city in their_cities[idx]:
#
#         river = f"input/river/{rv}/{city}/{city}.shp"
#         tiles_folder_dtm = f"input/AHN/{rv}/{city}/DTM"
#         tiles_folder_dsm = f"input/AHN/{rv}/{city}/DSM"
#         # FILES
#
#         output_cs_preproces = f"output/cross_sections/{rv}/{city}/preprocess/cross_sections.shp"
#         points_cs_preprocess = f"output/cross_sections/{rv}/{city}/preprocess/points.shp"
#         river_midpoints = f"output/river/{rv}/{city}/{city}_mid.shp"
#         embankments_file_left = f"output/embankment/{rv}/{city}/left/left.shp"
#         embankments_file_right = f"output/embankment/{rv}/{city}/right/right.shp"
#         embankments_file = f"output/embankment/{rv}/{city}/embankments.shp"
#         # PARAMETERS
#         interval = 100
#         width_preprocess = 500
#         user_defined_height = 1.75
#         # RUN
#         # cross_section_extraction(river, interval, width_preprocess, output_cs_preproces, river_midpoints)
#         # create_cross_section_points(output_cs_preproces,width_preprocess, points_cs_preprocess)
#         # add_elevation_from_tiles(points_cs_preprocess, tiles_folder_dtm, 'elev_dtm')
#         # extract_max_elevation_values(points_cs_preprocess, river_midpoints, embankments_file_left, embankments_file_right, user_defined_height, embankments_file)

# VISIBILITY ANALYSIS----------------------------------------------------------------------------------------------------
"""
Inbetween the pre-process and the parameter extraction, the viewshed analysis can be done. I do this using a script for
batch process in the QGIS python console visibility_batch_qgis.py
"""


# EXTRACT CROSS-SECTIONS WITH RIVER WIDTH-------------------------------------------------------------------------------
def cross_section_extraction_added_riverwidth(river_file, interval, width, output_file_cs, inputriver_mid):
    """
    Get cross-sections
    :param output_file_cs: output shapefile for all cross-sections
    :param width: width of total cross-section
    :param interval: width between cross-sections
    :param river: River geodataframe
    :param output_river_mid: Input file for river midpoints which contains the river width in column 'width'
    :return: gdf
    """
    river = gpd.read_file(river_file)
    midpoint_gdf = gpd.read_file(inputriver_mid)
    projected_gdf = river.to_crs(epsg=28992)
    cross_sections = []
    river_points = []

    for idx, row in projected_gdf.iterrows():
        riverline = row['geometry']
    #     for index, row in midpoint_gdf.iterrows():
    #         river_width = row['width']
    #         midpoint = row['geometry']
    #         # river_width = midpoint_gdf.iloc[index]['width']
    #         print(f"index: {index}, river width is {river_width}")
    #         if river_width is not None:
    #             total_width = width + river_width
    #             # print("index, riverline length: ", index, riverline.length)

        for indx, distance_along_line in enumerate(np.arange(0, riverline.length, interval)):
            river_width = midpoint_gdf.iloc[indx]['width']
            # print(f"index {indx} and river width {river_width}")
            if river_width != 0.0:
                total_width = width + river_width
                cross_section, point_on_line = get_perpendicular_cross_section(riverline, distance_along_line, total_width)
                cross_sections.append(cross_section)
                river_points.append(point_on_line)
            else:
                cross_sections.append(None)
                continue

    # Save cross-sections to a Shapefile (.shp)
    gdf = gpd.GeoDataFrame(geometry=cross_sections)
    gdf.set_crs(epsg=28992, inplace=True)
    gdf.to_file(output_file_cs, driver='ESRI Shapefile')

    # Save river midpoints to shapefile (.shp)
    # river_points_gdf = gpd.GeoDataFrame(geometry=river_points)
    # river_points_gdf.set_crs(epsg=28992, inplace=True)
    # river_points_gdf.to_file(output_river_mid, driver='ESRI Shapefile')

    return

def create_cross_section_points_added_riverwidth(cross_sections_shapefile, width, inputriver_mid, output_shapefile):
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
    midpoint_gdf = gpd.read_file(inputriver_mid)
    all_points = []

    for ind, row in cross_sections.iterrows():
        # print('Processing cross-section', ind)
        if row.geometry is not None:
            # print('Valid cs', ind)
            river_width = midpoint_gdf.iloc[ind]['width']
            # print(f"river width is {river_width}")
            total_width = river_width + width
            n_points = int(total_width / 0.5)

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

        else:
            print('Invalid cs', ind)
            all_points.append({
                'geometry': None,
                'id': ind,
                'h_distance': None
            })


    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(all_points)
    gdf.set_crs(epsg=28992, inplace=True)

    # Save to shapefile
    gdf.to_file(output_shapefile, driver='ESRI Shapefile')
    print(f"Base cross-section points saved to: {output_shapefile}")

    return gdf

# RUN CROSS-SECTION EXTRACTION------------------------------------------------------------------------------------------
"""
Cross-section that extend a width value beyond the river width are computed and points on these are added
"""


# PARAMETERS----------------------------------------------------------------------------------------------------------
"""
Parameter values for DTM elevation, DSM elevation, visibility and imperviousness are added to each point,
Each point gets a landuse description, sometimes having multiple. 
DTM, DSM, and flood data are added via the previous add_elevation_from_tiles function.
Imperviousness, visibility data are added via add_raster_column.
"""


# LANDUSE
def process_landuse(points_gdf, gpkg_folder):
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
    for i, gml_file in enumerate(os.listdir(gpkg_folder)):
        if not gml_file.endswith('.gml'):
            continue

        gml_path = os.path.join(gpkg_folder, gml_file)
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


def add_landuse_to_shapefile(shapefile_path, gpkg_folder):
    """
    Reads a shapefile, adds landuse information from gpkg files, and saves the updated shapefile.

    Parameters:
        shapefile_path: Path to the shapefile containing all cross-section points
        gpkg_folder: Folder containing landuse .gpkg files
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    print('The shapefile has been read! its columns are: ', gdf.columns)

    # Add landuse information
    gdf = process_landuse(gdf, gpkg_folder)
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


# FILES
# points_file = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_points_test.shp"
# visible_raster = f"output/visibility/{rv}{city}{folder}/combined_viewshed.tif"



# unique_landuses_dir = f"output/unique_landuses"
# os.makedirs(unique_landuses_dir, exist_ok=True)
# unique_landuses_output = f"output/unique_landuses/{city}.csv"
# bgt_folder = f"input/BGT/{city}"


# for city in lonely_rivers:
#     river = f"input/river/{city}/{city}.shp"
#     # tiles_folder_dtm = f"input/AHN/{city}/DTM"
#     # tiles_folder_dsm = f"input/AHN/{city}/DSM"
#     # FILES
#
#     # output_cs_preproces = f"output/cross_sections/{city}/preprocess/cross_sections.shp"
#     # points_cs_preprocess = f"output/cross_sections/{city}/preprocess/points.shp"
#     # river_midpoints = f"output/river/{city}/{city}_mid.shp"
#     # embankments_file_left = f"output/embankment/{city}/left/left.shp"
#     # embankments_file_right = f"output/embankment/{city}/right/right.shp"
#     # embankments_file = f"output/embankment/{city}/embankments.shp"
#     # PARAMETERS
#     interval = 100
#     width_preprocess = 300
#     user_defined_height = 1.75
#     width = 200
#     interval = 100
#     # FILES
#     # output_pts = f"output/cross_sections/{city}/final/points.shp"
#     # os.makedirs(output_pts, exist_ok=True)
#     # output_cs = f"output/cross_sections/{city}/final/cross_sections.shp"
#     # os.makedirs(output_cs, exist_ok=True)
#     # bgt_folder = f"input/BGT/{city}"
#     # unique_landuses_dir = f"output/unique_landuses"
#     # os.makedirs(unique_landuses_dir, exist_ok=True)
#     # unique_landuses_output = f"output/unique_landuses/{city}.csv"
#
#     # RUN
#     # cross_section_extraction_added_riverwidth(river, interval, width, output_cs, river_midpoints)
#     # create_cross_section_points_added_riverwidth(output_cs, width, river_midpoints, output_pts)
#     # add_landuse_to_shapefile(output_pts, bgt_folder)
#     # extract_unique_landuses(output_pts, unique_landuses_output)
#     # add_elevation_from_tiles(output_pts, "input/flood/middelgrote_kans", 'flood_dept')
#     #
#     # add_elevation_from_tiles(output_pts, tiles_folder_dtm, 'elev_dtm')
#     # add_elevation_from_tiles(output_pts, tiles_folder_dsm, 'elev_dsm')



# for idx, rv in enumerate(main_rivers):
#     for city in their_cities[idx]:
#         river = f"input/river/{rv}/{city}/{city}.shp"
#         tiles_folder_dtm = f"input/AHN/{rv}/{city}/DTM"
#         tiles_folder_dsm = f"input/AHN/{rv}/{city}/DSM"
#         # FILES
#         output_cs_preproces = f"output/cross_sections/{rv}/{city}/preprocess/cross_sections.shp"
#         points_cs_preprocess = f"output/cross_sections/{rv}/{city}/preprocess/points.shp"
#         river_midpoints = f"output/river/{rv}/{city}/{city}_mid.shp"
#         embankments_file_left = f"output/embankment/{rv}/{city}/left/left.shp"
#         embankments_file_right = f"output/embankment/{rv}/{city}/right/right.shp"
#         embankments_file = f"output/embankment/{rv}/{city}/embankments.shp"
#         # FILES
#         output_pts = f"output/cross_sections/{rv}/{city}/final/points.shp"
#         os.makedirs(output_pts, exist_ok=True)
#         output_cs = f"output/cross_sections/{rv}/{city}/final/cross_sections.shp"
#         os.makedirs(output_cs, exist_ok=True)
#         unique_landuses_dir = f"output/unique_landuses"
#         os.makedirs(unique_landuses_dir, exist_ok=True)
#         unique_landuses_output = f"output/unique_landuses/{city}.csv"
#         bgt_folder = f"input/BGT/{city}"
#         # PARAMETERS
#         interval = 100
#         width_preprocess = 500
#         user_defined_height = 1.75
#         width = 200
#         interval = 100
#
#         # RUN
#         # cross_section_extraction_added_riverwidth(river, interval, width, output_cs, river_midpoints)
#         # create_cross_section_points_added_riverwidth(output_cs, width, river_midpoints, output_pts)
#         # add_landuse_to_shapefile(output_pts, bgt_folder)
#         # extract_unique_landuses(output_pts, unique_landuses_output)
#         # add_elevation_from_tiles(output_pts, "input/flood/middelgrote_kans", 'flood_dept')
#         #
#         # add_elevation_from_tiles(output_pts, tiles_folder_dtm, 'elev_dtm')
#         # add_elevation_from_tiles(output_pts, tiles_folder_dsm, 'elev_dsm')
# RUN
# add_landuse_to_shapefile(output_pts, bgt_folder)
# extract_unique_landuses(output_pts, unique_landuses_output)
# add_elevation_from_tiles(output_pts, "input/flood/middelgrote_kans", 'flood_dept')
#
# add_elevation_from_tiles(output_pts, tiles_folder_dtm, 'elev_dtm')
# add_elevation_from_tiles(output_pts, tiles_folder_dsm, 'elev_dsm')

# add_raster_column(shapefile_path=output_pts, raster_path=visible_raster, column_name="visible")
main_rivers = ['maas/', '', '',  '', '', '']
their_cities = [['maastricht'],  ['BovenMark'], ['Harinxmakanaal'], ['RijnSchieKanaal'], ['schie'], ['vliet']]
# for idx, rv in enumerate(main_rivers):
#     for idy, city in enumerate(their_cities[idx]):
#         points_file = f"output/cross_sections/{rv}{city}/final/points.shp/points.shp"
#         viewshed_file = f'output/visibility/{rv}{city}/combined_viewshed.tif'
#
#         add_raster_column(shapefile_path=points_file, raster_path=viewshed_file, column_name='visible')
#         add_elevation_from_tiles(points_file, "input/flood/middelgrote_kans", 'flood_dept')

main_rivers_segments = ['ar/', 'ijssel/', '', 'waal/', '']
segments_subrivers = [['amsterdam', 'utrecht'], ['deventer'], ['lek'], ['nijmegen', 'zaltbommel'], ['Winschoterdiep']]
segs = [[['_0_99', '_100_199', '_200_255'], ['_0_99', '_100_199', '_200_299', '_300_326']], [['_0_99', '_100_199', '_200_299', '_300_399', '_400_499', '_500_599', '_600_642']],[['_0_99', '_100_199', '_200_299', '_300_394']], [['_0_99', '_100_199', '_200_299', '_300_399', '_400_456'],['_0_99', '_100_178']], [['_0_99', '_100_199', '_200_299', '_300_348']] ]

# imperv_raster = f"input/imperviousness/MERGED_reproj_28992.tif"
# add_raster_column(shapefile_path=output_pts, raster_path=imperv_raster, column_name='imperv')
# for idx, rv in enumerate(main_rivers_segments):
#     for idy, city in enumerate(segments_subrivers[idx]):
#         for seg in segs[idx][idy]:
#             parts = seg.rsplit('_', 1)  # Split from the right, splitting only once
#             prefix, number = parts[0], parts[1]
#             # Increment the numeric part
#             new_number = int(number) + 1
#             new_seg =  f"{prefix}_{new_number}"
#             viewshed_file = f'output/visibility/{rv}{city}/viewsheds/viewshed{new_seg}/combined_viewshed.tif'
#             points_file = f"output/cross_sections/{rv}{city}/final/points{seg}/points{seg}.shp"
#             # add_raster_column(shapefile_path=points_file, raster_path=viewshed_file, column_name='visible')
#             add_elevation_from_tiles(points_file, "input/flood/middelgrote_kans", 'flood_dept')

# This groups the points shapefiles into segments
# for idx, rv in enumerate(main_rivers_segments):
#     for city in segments_subrivers[idx]:
#         print(f"Processing {city}")
#
#         output_pts = f"output/cross_sections/{rv}{city}/final/cross_sections.shp"
#         max_ids = 100
#         points_gdf = gpd.read_file(output_pts)
#         output_dir = os.path.dirname(output_pts)
#
#         grouped = points_gdf.groupby('FID')
#
#
#         def chunk_groupby(grouped, max_ids):
#             iterator = iter(grouped)
#             while True:
#                 chunk = list(islice(iterator, max_ids))
#                 if not chunk:
#                     break
#                 yield chunk
#
#
#         for i, chunk in enumerate(chunk_groupby(grouped, max_ids)):
#             # Combine the groups in the chunk into a single GeoDataFrame
#             chunk_gdf = gpd.GeoDataFrame(pd.concat([group for _, group in chunk]), crs=points_gdf.crs)
#
#             # Determine start and end indices of IDs
#             start_id = chunk_gdf['FID'].iloc[0]
#             end_id = chunk_gdf['FID'].iloc[-1]
#
#             # Create a unique folder for this shapefile
#             subfolder_name = f"cross_sections_{start_id}_{end_id}"
#             subfolder_path = os.path.join(output_dir, subfolder_name)
#             os.makedirs(subfolder_path, exist_ok=True)
#
#             # Determine output filename within the folder
#             output_filename = os.path.join(subfolder_path, f"{subfolder_name}.shp")
#             # Save the chunk
#             chunk_gdf.to_file(output_filename)
#             print(f"Saved {output_filename}")
#
#
#         print(f"Done processing {city}")


# SPLIT SECTIONS INTO LEFT AND RIGHT-----------------------------------------------------------------------------------
"""
The cross-sections are split in the middle, mainly for the purpose of finding building intersections.
left and right can then be processed seperatly. Side 0 is left, side 1 is right
Cross-section halves are saved to a seperate 'halves' file
Split points is added in columns to the points shapefile. 
columns: ['id', 'h_distance', 'visible', 'imperv', 'elev_dtm', 'elev_dsm',
       'landuse', 'flood_dept', 'side', 'split_h_di', 'geometry']
Split cross-sections shapefile are used for the building intersections
columns: ['id', 'side', 'geometry']
Split points shapefile is used for further metric computation
"""
def split_cross_sections(midpoints_shp, cross_sections_shp):
    """
    Splits the cross-section geometries in half and adds the left and right linestrings
    to a new shapefile.

    Args:
        midpoints_shp (str): File path to the midpoints shapefile.
        cross_sections_shp (str): File path to the cross-sections shapefile.

    Returns:
        None
    """
    midpoints = gpd.read_file(midpoints_shp)
    cross_sections = gpd.read_file(cross_sections_shp)

    # Create lists to store the data for the output GeoDataFrame
    # ids = []
    # sides = []
    geometries = []

    for idx, row in cross_sections.iterrows():
        cross_section = row.geometry
        if cross_section is not None:
            midpoint = midpoints[midpoints['FID'] == row['FID']].iloc[0].geometry
            coords = list(cross_section.coords)
            left_line = LineString([midpoint, coords[0]])
            right_line = LineString([midpoint, coords[-1]])

            # Append the left and right linestrings to the lists
            # ids.extend([row['FID'], row['FID']])
            # sides.extend([0, 1])
            # geometries.extend([left_line, right_line])
            print(f"i append {row['FID']}, 0 and {left_line}")
            geometries.append([row['FID'], 0, left_line])
            geometries.append([row['FID'], 1, right_line])

    # Create the output GeoDataFrame
    # output_df = gpd.GeoDataFrame({'id': ids, 'side': sides, 'geometry': geometries}, crs=midpoints.crs)
    output_df = gpd.GeoDataFrame(geometries, columns=['id', 'side', 'geometry'])
    output_df.set_crs("EPSG:28992", inplace=True)
    # Save the output GeoDataFrame to a new shapefile

    output_df.to_file(midpoints_shp.replace('.shp', '_halves.shp'))


def split_points(points_shp):
    """
    Splits the points file in the two half by creating columns 'side' and 'split_h_di'
    Splitpoint is determined as half of max h_distance


    side: 1 for right side, 0 for left side
    split_h_di give the distance along line from midpoint
    Args:
        points_shp: shapefile of points with parameters

    Returns:

    """
    gdf = gpd.read_file(points_shp)
    grouped = gdf.groupby('id')

    for idx, section in tqdm(grouped, desc="Processing sections"):
        max_h_distance = section['h_distance'].max()
        splitpoint = max_h_distance / 2

        # boolean to determine side. True is converted by astype(int) to 1, and False to 0 integers.
        section['side'] = (section['h_distance'] >= splitpoint).astype(int)

        # Create the new h_distance column as the absolute difference from the splitpoint
        section['split_h_di'] = (section['h_distance'] - splitpoint).abs()

        # Update the main GeoDataFrame with the modified 'section'
        gdf.loc[gdf['id'] == idx, ['side', 'split_h_di']] = section[['side', 'split_h_di']]

    gdf.to_file(points_shp)
    return


# split_points(output_pts)
# split_cross_sections(midpoints_shp=river_midpoints, cross_sections_shp=output_cs)
main_rivers_segments = ['ar/', 'ijssel/', '', 'waal/', '']
segments_subrivers = [['amsterdam', 'utrecht'], ['deventer'], ['lek'], ['nijmegen', 'zaltbommel'], ['Winschoterdiep']]
segs = [[['_0_99', '_100_199', '_200_255'], ['_0_99', '_100_199', '_200_299', '_300_326']], [['_0_99', '_100_199', '_200_299', '_300_399', '_400_499', '_500_599', '_600_642']],[['_0_99', '_100_199', '_200_299', '_300_394']], [['_0_99', '_100_199', '_200_299', '_300_399', '_400_456'],['_0_99', '_100_178']], [['_0_99', '_100_199', '_200_299', '_300_348']] ]

# for idx, rv in enumerate(main_rivers_segments):
#     for idy, city in enumerate(segments_subrivers[idx]):
#         for seg in segs[idx][idy]:
#             parts = seg.rsplit('_', 1)  # Split from the right, splitting only once
#             prefix, number = parts[0], parts[1]
#             new_number = int(number) + 1
#             new_seg =  f"{prefix}_{new_number}"
#             midpoints_file = f'output/visibility/segments/{rv}{city}/mid/{city}_mid_segment{new_seg}.shp'
#             points_file = f"output/cross_sections/{rv}{city}/final/points{seg}/points{seg}.shp"
#             cs_file = f"output/cross_sections/{rv}{city}/final/cross_sections{seg}/cross_sections{seg}.shp"
#
#             split_points(points_file)
#             split_cross_sections(midpoints_shp=midpoints_file, cross_sections_shp=cs_file)
# BUILDING INTERSECTION------------------------------------------------------------------------------------------------
"""
Get the intersections of the cross-sections with buildings.
Adds the first point of intersection and the max building height of this intersection to the shapefile containing the half cross-sections
Produces a shapefile with intersections with columns id and geometry (linestring)
"""
def building_parameters(halves_shp, building_gpkg, intersection_output_file, closest_intersection_output_file):
    """
    Compute parameters related to building intersections with the river space.
    This version handles cross-sections that have already been split into left and right parts, with the input file in a custom format.

    Parameters:
    cs_halvex_shp (str): Path to the Shapefile containing the pre-split cross-sections.
    building_gpkg (str): Path to the GeoPackage containing the building data.
    river_shp (str): Path to the Shapefile containing the river geometry.

    Returns:
    GeoDataFrame: The input cs_halvex_shp GeoDataFrame with additional columns:
        buil1 (Point): The first intersection point
        height1 (float): The maximum building height of the first intersection
        buil_int (MultiLineString): The geometry of all intersection points
    """
    # Load data
    halves = gpd.read_file(halves_shp)
    # buildings = gpd.read_file(building_gpkg, layer="pand")
    buildings = gpd.read_file(building_gpkg, layer="lod12_2d")
    # river = gpd.read_file(river_shp)

    # Prepare additional columns in the cs_halvex GeoDataFrame
    # cs_halvex['build1'] = None
    # cs_halvex['height1'] = None

    intersections_list = []
    closest_intersections_list = []

    for _, row in halves.iterrows():
        line = row.geometry
        side = row['side']
        id = row['id']

        # point is [intersection_point, max_height, building['identificatie']]
        point, total = get_intersection(line, buildings)

        if point is not None:
            # cs_halvex.loc[_, 'build1'] = point
            # cs_halvex.loc[_, 'height1'] = point.z
            closest_intersections_list.append([id, side, point[0], point[1]])
            for intersection in total:
                intersections_list.append([id,side,intersection])


    # cs_halvex.to_file(cs_halvex_shp)
    gdf = gpd.GeoDataFrame(intersections_list, columns=["id","side","geometry"])
    gdf.set_crs("EPSG:28992", inplace=True)
    gdf.to_file(intersection_output_file, driver="ESRI Shapefile")

    gdf_clos = gpd.GeoDataFrame(closest_intersections_list, columns=['id', 'side', 'geometry', 'height'])
    gdf_clos.set_crs("EPSG:28992", inplace=True)
    gdf_clos.to_file(closest_intersection_output_file, driver="ESRI Shapefile")

    return

def get_intersection(line, buildings):
    # Get intersecting buildings directly using spatial operation
    intersecting_buildings = buildings[buildings.intersects(line)]

    if len(intersecting_buildings) == 0:
        return None, []

    intersections = []
    total_intersections = []
    for idx, building in intersecting_buildings.iterrows():
        try:
            intersection = line.intersection(building.geometry)
            total_intersections.append(intersection)

            if isinstance(intersection, LineString):
                intersection_point = Point(intersection.coords[0])
            elif isinstance(intersection, MultiPolygon):
                # Take the centroid of the first polygon
                intersection_point = Point(list(intersection.geoms)[0].centroid)
            elif isinstance(intersection, Polygon):
                intersection_point = Point(intersection.centroid)
            elif isinstance(intersection, Point):
                intersection_point = intersection
            elif isinstance(intersection, MultiLineString):
                # Get the first LineString in the MultiLineString
                first_line = intersection.geoms[0]
                # Get the first coordinate of this LineString
                first_coord = first_line.coords[0]
                # Create a Point from the first coordinate
                intersection_point = Point(first_coord)
            else:
                print(f"Unexpected intersection type: {type(intersection)}")
                continue

            # max_height = get_max_height(building.geometry)
            max_height = building['b3_h_max']
            intersections.append([intersection_point, max_height, building['identificatie']])

        except Exception as e:
            print(f"Error processing building {idx}: {e}")

    if intersections:
        # Use the actual line for projection
        closest_intersection = min(intersections, key=lambda x: line.project(x[0]))
        # print(f"intersections {intersections}, {closest_intersection[0]}, {total_intersections}")
        return  closest_intersection, total_intersections
    return  None, []

def get_max_height(geometry):
    if isinstance(geometry, Polygon):
        return np.max([coord[2] for coord in geometry.exterior.coords])
    elif isinstance(geometry, MultiPolygon):
        return np.max([np.max([coord[2] for coord in poly.exterior.coords]) for poly in geometry.geoms])
    else:
        raise ValueError(f"Unexpected geometry type: {type(geometry)}")

main_rivers = ['maas/', '', '',  '', '', '']
their_cities = [['cuijk', 'roermond', 'venlo','maastricht'],  ['BovenMark'], ['Harinxmakanaal'], ['RijnSchieKanaal'], ['schie'], ['vliet']]
for idx, rv in enumerate(main_rivers):
    for city in their_cities[idx]:
        midpoints_file = f"output/river/{rv}{city}/{city}_mid.shp"
        halves = f"output/river/{rv}{city}/{city}_mid_halves.shp"
        points_file = f"output/cross_sections/{rv}{city}/final/points.shp/points.shp"
        cs_file = f"output/cross_sections/{rv}{city}/final/cross_sections.shp/cross_sections.shp"

        split_points(points_file)
        split_cross_sections(midpoints_shp=midpoints_file, cross_sections_shp=cs_file)

        intersections = f"output/buildings/{rv}{city}/building_intersections"
        os.makedirs(intersections, exist_ok=True)
        closest_intersections = f"output/buildings/{rv}{city}/closest_building_intersections"
        os.makedirs(closest_intersections, exist_ok=True)
        buildings = f'input/3DBAG/{rv}{city}/{city}/combined_tiles/combined.gpkg'
        building_parameters(halves, buildings, intersections, closest_intersections)

# halves = f"output/river/{rv}/{city}/halves.shp"
# # buildings = "input/3DBAG/KanaalVanWalcheren/3DBAG_combined_tiles/combined_3DBAG.gpkg"
# intersections = f"output/buildings/{rv}/{city}/building_intersections"
# closest_intersections = f"output/buildings/{rv}/{city}/closest_building_intersections"
# for idx, rv in enumerate(main_rivers_segments):
#     for idy, city in enumerate(segments_subrivers[idx]):
#         for seg in segs[idx][idy]:
#             parts = seg.rsplit('_', 1)  # Split from the right, splitting only once
#             prefix, number = parts[0], parts[1]
#             new_number = int(number) + 1
#             new_seg =  f"{prefix}_{new_number}"
#             midpoints_file = f'output/visibility/segments/{rv}{city}/mid/{city}_mid_segment{new_seg}.shp'
#             halve_file = f'output/visibility/segments/{rv}{city}/mid/{city}_mid_segment{new_seg}_halves.shp'
#             points_file = f"output/cross_sections/{rv}{city}/final/points{seg}/points{seg}.shp"
#             cs_file = f"output/cross_sections/{rv}{city}/final/cross_sections{seg}/cross_sections{seg}.shp"
#
#             intersections = f"output/buildings/{rv}{city}/{city}{seg}/building_intersections"
#             os.makedirs(intersections, exist_ok=True)
#             closest_intersections = f"output/buildings/{rv}{city}/{city}{seg}/closest_building_intersections"
#             os.makedirs(closest_intersections, exist_ok=True)
#
#             buildings = f'input/3DBAG/{rv}{city}/{city}{seg}/combined_tiles/combined.gpkg'
#             building_parameters(halve_file, buildings, intersections, closest_intersections)


# building_parameters(halves, buildings, intersections, closest_intersections)
# inters = gpd.read_file(closest_intersections)
# print(inters.columns)
# print(inters.head(10))

