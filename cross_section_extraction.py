"""
This script computes the cross-sectional lines, begin point and end point, along a shapefile line, in this case a river.
First it computes short crosssectional lines crossing the riverline. This is to find the embankment points.
This is taking forever, we need to optimize the code.
1. Compute cross-sections for pre-processing. Get midpoints of the river.
2. Compute points along these cross-sections with given interval
3. Get elevation from DTM tiles to points. Find embankment points and river width.
4. Compute cross-sections again but now with riverwidth added to the width.
5. Compute points along these cross-sections with computed interval (0.5)
5. Get elevation values from DSM and DTM
Using these embankment points, we compute lateral sections, only extending from this point, for the left and right side of the river. These are the sections we use in our further analysis.
The preprocess cross-sections and points can be removed then.
TODO: clean up
TODO: make imperviousness function geenrla, only name change. It is raster reading funciton. I also use it for visibility
"""
import os
import fiona
import rasterio
import pandas as pd
from tqdm import tqdm
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
from shapely.geometry import Point, box
from concurrent.futures import ProcessPoolExecutor, as_completed


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

    for index, row in projected_gdf.iterrows():
        riverline = row['geometry']

        print("index, riverline length: ", index, riverline.length)

        for distance_along_line in np.arange(0, riverline.length, interval):
            cross_section, point_on_line = get_perpendicular_cross_section(riverline, distance_along_line, width)
            cross_sections.append(cross_section)
            river_points.append(point_on_line)

    # Save cross-sections to a Shapefile (.shp)
    gdf = gpd.GeoDataFrame(geometry=cross_sections)
    gdf.set_crs(epsg=28992, inplace=True)
    gdf.to_file(output_file_cs, driver='ESRI Shapefile')

    # Save river midpoints to shapefile (.shp)
    river_points_gdf = gpd.GeoDataFrame(geometry=river_points)
    river_points_gdf.set_crs(epsg=28992, inplace=True)
    river_points_gdf.to_file(output_river_mid, driver='ESRI Shapefile')

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
        with rasterio.open(tif_path) as src:
            tif_bounds = src.bounds
            for idx, row in points_gdf.iterrows():
                if row.geometry is not None:
                    point = row.geometry

                    minx, miny, maxx, maxy = tif_bounds
                    if minx <= point.x <= maxx and miny <= point.y <= maxy:
                        try:
                            py, px = src.index(point.x, point.y)
                            window = rasterio.windows.Window(px - 1, py - 1, 3, 3)
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


# def add_elevation_from_tiles(shapefile_path, tiles_folder, elevation_column_name):
#     """
#     Adds elevation data from tiles to existing shapefile and updates it.
#
#     Parameters:
#     shapefile_path: Path to the existing shapefile with points
#     tiles_folder: Folder containing elevation tiles (.tif files)
#     elevation_column_name: Name of the column to store elevation values
#
#     Returns:
#     GeoDataFrame with added elevation column and count of missing points
#     """
#     # Read existing shapefile
#     missing_points = 0
#     nodata_points = 0
#     outside_coverage = 0
#     points_gdf = gpd.read_file(shapefile_path)
#     print(f"Loaded {len(points_gdf)} points from {shapefile_path}")
#
#     # Initialize elevation column with NaN
#     points_gdf[elevation_column_name] = np.nan
#
#     # Get list of tif files
#     tif_files = [f for f in os.listdir(tiles_folder) if f.endswith(".tif")]
#
#     # Process each cross-section separately
#     cross_sections = points_gdf.groupby('id')
#
#     # Loop through each cross-section group
#     for cross_section_id, section_gdf in tqdm(cross_sections, desc="Processing cross sections"):
#         # Create bounding box for the cross-section
#         bounds = section_gdf.total_bounds
#         section_bbox = box(*bounds)
#
#         # Find relevant TIF files that intersect with this cross-section
#         relevant_tifs = []
#         for tif_file in tif_files:
#             tif_path = os.path.join(tiles_folder, tif_file)
#             with rasterio.open(tif_path) as src:
#                 tile_bbox = box(*src.bounds)
#                 if section_bbox.intersects(tile_bbox):
#                     relevant_tifs.append(tif_file)
#
#         if not relevant_tifs:
#             print(f"Warning: No TIF files found covering cross section {cross_section_id}")
#             outside_coverage += len(section_gdf)
#             continue
#
#         # Process each point in the cross-section
#         for idx, point in section_gdf.iterrows():
#             point_x, point_y = point.geometry.x, point.geometry.y
#             elevation_found = False
#             is_nodata = False
#
#             # Try each relevant TIF file until we find an elevation
#             for tif_file in relevant_tifs:
#                 tif_path = os.path.join(tiles_folder, tif_file)
#                 with rasterio.open(tif_path) as src:
#                     # Check if point is within this tile's bounds
#                     if (src.bounds.left <= point_x <= src.bounds.right and
#                             src.bounds.bottom <= point_y <= src.bounds.top):
#
#                         try:
#                             # Convert point coordinates to pixel coordinates
#                             py, px = src.index(point_x, point_y)
#
#                             # Read the elevation value
#                             window = rasterio.windows.Window(px - 1, py - 1, 3, 3)
#                             data = src.read(1, window=window)
#
#                             # Check center pixel
#                             center_value = data[1, 1]
#                             if center_value == src.nodata:
#                                 is_nodata = True
#                             elif data.size > 0:
#                                 points_gdf.at[idx, elevation_column_name] = float(center_value)
#                                 elevation_found = True
#                                 break
#
#                         except (IndexError, ValueError):
#                             continue
#
#             if not elevation_found:
#                 missing_points += 1
#                 if is_nodata:
#                     nodata_points += 1
#                     print(f"NoData value found for point {idx} at ({point_x}, {point_y})")
#                 else:
#                     print(
#                         f"No elevation data found for point {idx} at ({point_x}, {point_y}) - point may be between tiles")
#
#     # Save the updated GeoDataFrame
#     points_gdf.to_file(shapefile_path, driver='ESRI Shapefile')
#     print(f"\nUpdated shapefile with {elevation_column_name} at: {shapefile_path}")
#     print(f"Total missing points: {missing_points}")
#
#     return points_gdf


# Get max elevation of the embankment for viewshed analysis. Get embankment points.-------------------------------------
def extract_max_elevation_values(shapefile, midpoints_file, embankments_file_left, embankments_file_right,
                                 user_defined_height):
    """
    Extracts the elevation values of the embankments, and selects the max value for the viewshed analysis
    This max value is then added up with user_defined_height.
    Takes embankment h_dist and selects max, multiplies this my 2, then add this as 'width' to add to radius in viewshed analysis
    Args:
        shapefile: parameter shapefile
        midpoints_file: cross-section midpoint file that is altered
        user_defined_height: viewpoint height to peform viewshed analysis from

    Returns: saves values left, right (embankment), max, and (viewpoint) height to midpoints_file
    TODO: now when the river extend my buffer, no boundary point gets added. Fix to edge of buffer, or last point
    """
    parameters = gpd.read_file(shapefile)
    midpoints = gpd.read_file(midpoints_file)

    # Initialize new columns in midpoints for output
    midpoints['left'] = np.nan
    midpoints['right'] = np.nan
    midpoints['max'] = np.nan
    midpoints['height'] = np.nan
    midpoints['width'] = np.nan  # Extra width for viewshed

    # Initialize embankments points
    embankment_points_left = gpd.GeoDataFrame(columns=['geometry', 'h_distance'])
    embankment_points_left['geometry'] = np.nan
    embankment_points_left['h_distance'] = np.nan
    embankment_points_left.set_crs("EPSG:28992", inplace=True)
    embankment_points_right = gpd.GeoDataFrame(columns=['geometry', 'h_distance'])
    embankment_points_right['geometry'] = np.nan
    embankment_points_right['h_distance'] = np.nan
    embankment_points_right.set_crs("EPSG:28992", inplace=True)

    # Group by cross-section ID
    grouped = parameters.groupby('id')
    for idx, row in tqdm(midpoints.iterrows(), total=midpoints.shape[0], desc="Processing midpoints"):
        midpoint_geom = row.geometry
        cross_section_id = row['FID']

        # Select corresponding cross-section points from 'parameters' shapefile
        cross_section_points = grouped.get_group(cross_section_id)

        # Ensure points are sorted by h_distance
        cross_section_points = cross_section_points.sort_values('h_distance').reset_index(drop=True)

        # Calculate approximate midpoint in h_distance
        max_h_distance = cross_section_points['h_distance'].max()
        target_midpoint = max_h_distance / 2

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
                # print(f"left geom is {left_geom} and type {type(left_geom)}")
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
        embankment_points_left.at[idx, 'geometry'] = left_geom
        embankment_points_right.at[idx, 'geometry'] = right_geom
        embankment_points_left.at[idx, 'h_distance'] = left_hdist
        embankment_points_right.at[idx, 'h_distance'] = right_hdist
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
            print(f"distance {abs(right_hdist - left_hdist)}")

        # Save the result to a new shapefile
    midpoints.to_file(midpoints_file, driver='ESRI Shapefile')
    embankment_points_left.to_file(embankments_file_left, driver="ESRI Shapefile")
    print(f"Results saved to {midpoints_file} and {embankments_file_left}")
    embankment_points_right.to_file(embankments_file_right, driver="ESRI Shapefile")
    print(f"Results saved to {midpoints_file} and {embankments_file_right}")


# Get correct cross-section outward from embankment points--------------------------------------------------------------
# x
# def get_outward_cross_sections(riverline, embank_points_left, embankmet_points_right, width, output_file):
#     """
#     Gets the outward cross-sections from the boundary points pointing outwards with respect to the river
#     Args:
#         riverline: Shapefile of river
#         boundary_points: Shapefile of embankment points
#         width: width of section (100m)
#         output_file: Shapefile to write output to
#
#     Returns:
#
#     """
#     gdf_riv = gpd.read_file(riverline)
#     gdf_left = gpd.read_file(embank_points_left)
#     gdf_right = gpd.read_file(embankmet_points_right)
#     # print(f"gdf points {gdf_pts}")
#     cross_sections = []
#     for index, row in gdf_left.iterrows():
#         geometry = row['geometry']
#         # print(f"point {pt}")
#         # pt_geom = pt.iloc['geometry']
#         # print(f"geometry point {pt_geom}")
#         cs = get_outward_cross_section(gdf_riv, geometry, width)
#         cross_sections.append(cs)
#
#     gdf = gpd.GeoDataFrame(geometry=cross_sections)
#     gdf.set_crs(epsg=28992, inplace=True)
#     gdf.to_file(output_file, driver='ESRI Shapefile')
#     return
#
# # x
# def get_outward_cross_section(line, embank_point, width):
#     """
#     Generates a cross-Section that points outward from the riverline at a given boundary point.
#
#     Parameters:
#     line (LineString): The river centerline geometry.
#     boundary_point (Point): The point on the river boundary where the cross-section should originate.
#     width (float): The desired width of the cross-section.
#
#     Returns:
#     LineString: The outward-pointing cross-section geometry.
#     """
#     # Get the tangent angle at the boundary point
#     # print(f"line is {type(line)} and geometry of first is {type(line.iloc[0]['geometry'])}")
#     # print(f"point {type(embank_point)}")
#     line_geom = line.iloc[0]['geometry']
#     small_distance = 0.001
#     if line_geom.project(embank_point) + small_distance <= line_geom.length:
#         point_ahead = line_geom.interpolate(line_geom.project(embank_point) + small_distance)
#     else:
#         point_ahead = line_geom.interpolate(line_geom.length)
#
#     if embank_point:
#         x0, y0 = embank_point.x, embank_point.y
#         x1, y1 = point_ahead.x, point_ahead.y
#         angle = np.arctan2(y1 - y0, x1 - x0)
#
#     # Calculate the outward-pointing line endpoints
#         dx = width / 2 * np.cos(angle)
#         dy = width / 2 * np.sin(angle)
#         p1 = Point(x0 + dx, y0 + dy)
#         p2 = Point(x0 - dx, y0 - dy)
#         return LineString([p1, p2])
#     else:
#         print("embankment point is none ")
#         return None
#
#     return LineString([p1, p2])

# riverwidth
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
            print(f"index {indx} and river width {river_width}")
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
        print('Processing cross-section', ind)
        if row.geometry is not None:
            print('Valid cs', ind)
            river_width = midpoint_gdf.iloc[ind]['width']
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

# Parameters, input and output files-----------------------------------------------------------------------------------
river = "input/river/longest_river.shp"
interval = 50
width1 = 200
output_cs_preproces = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_cs_pre.shp"
river_midpoints = "output/river/KanaalVanWalcheren/KanaalVanWalcheren_mid.shp"
n_points = 2 * width1
points_cs_preprocess = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_pts_cs_pre.shp"
tiles_folder_dtm = "input/AHN/KanaalVanWalcheren/DTM_test"
tiles_folder_dsm = "input/AHN/KanaalVanWalcheren/DSM_test"
embankments_file_left = "output/embankment/KanaalVanWalcheren/left/left.shp"
embankments_file_right = "output/embankment/KanaalVanWalcheren/right/right.shp"
user_defined_height = 1.75
width = 100
number_points = 2 * width
width2 = 200
output_cs_l = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_cs_left.shp"
output_cs_r = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_cs_right.shp"
output_pts_l = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_points_left.shp"
output_pts_r = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_points_right.shp"


output_pts = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_points.shp"
output_cs = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_cs_final.shp"

# RUN-------------------------------------------------------------------------------------------------------------------
# cross_section_extraction(river, interval, width1, output_cs_preproces, river_midpoints)
# create_cross_section_points(output_cs_preproces, n_points, points_cs_preprocess)
# add_elevation_from_tiles(points_cs_preprocess, tiles_folder_dtm, 'elev_dtm')

# gdf = gpd.read_file(points_cs_preprocess)
# gdf.to_csv("test.csv", index=False)
# extract_max_elevation_values(points_cs_preprocess, river_midpoints, embankments_file_left, embankments_file_right, user_defined_height)


# cross_section_extraction_added_riverwidth(river, interval, 200, output_cs, river_midpoints)
###### get_outward_cross_sections(river, embankments_file_right, width, output_cs_r)
# create_cross_section_points_added_riverwidth(output_cs, width, river_midpoints, output_pts)
# add_elevation_from_tiles(output_pts, tiles_folder_dtm, 'elev_dtm')
# add_elevation_from_tiles(output_pts, tiles_folder_dsm, 'elev_dsm')
# add_elevation_from_tiles(output_pts, "input/flood/middelgrote_kans", 'flood_dept')

# ###get_outward_cross_sections(river, embankments_file_left, width, output_cs_l)
##### create_cross_section_points(output_cs_l, number_points, output_pts_l)
#### add_elevation_from_tiles(output_pts_l, tiles_folder_dtm, 'elev_dtm')
#### add_elevation_from_tiles(output_pts_l, tiles_folder_dsm, 'elev_dsm')

# LANDUSE---------------------------------------------------------------------------------------------------------------

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
    print('I have found the bbox of points: ', bbox)
    print('total number of points: ', len(points_gdf))

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
                    print(f'No features in bbox for layer {layer_name}')
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
    Extract unique landuse values from a shapefile.

    Args:
        shapefile_path: Path to the shapefile containing landuse information
        output_file: Path to save the unique landuses CSV
    """
    # Read the shapefile as a GeoDataFrame
    points_gdf = gpd.read_file(shapefile_path)

    # Extract unique landuse values
    unique_landuses = points_gdf['landuse'].dropna().unique()

    # Convert to DataFrame and save to CSV
    landuses_df = pd.DataFrame(unique_landuses, columns=['landuse'])
    landuses_df.to_csv(output_file, index=False)

    print(f"Unique landuses saved to: {output_file}")


# RUN
# I work with bgt_area now. I do not convert the file anymoer because the conversion gives me errors.
# I loop over all .gml files. Results in missing values sometimes.
# add_landuse_to_shapefile(output_pts, 'input/BGT/bgt_area')
# extract_unique_landuses(output_pts, 'output/parameters/unique_landuses.csv')

# IMPERVIOUSNESS----------------------------------------------------------------------------------------------
def load_raster_data(imperviousness_raster):
    with rasterio.open(imperviousness_raster) as src:
        # Load entire raster into memory
        data = src.read(1)  # Read the first band
        transform = src.transform
    return data, transform


def check_imperviousness(location, raster_data, transform):
    if location is None or location.is_empty or location.geom_type != 'Point':
        print(f"location is wrong, not point or empty, location is {location}")
        return np.nan
    # Extract coordinates from the Point object
    x, y = location.x, location.y

    # Get the row, col index of the pixel corresponding to the point
    row, col = ~transform * (x, y)
    row, col = int(row), int(col)

    # Ensure indices are within bounds
    if 0 <= row < raster_data.shape[0] and 0 <= col < raster_data.shape[1]:
        pixel_value = raster_data[row, col]

    else:
        pixel_value = np.nan  # Or another value indicating out-of-bounds
        print('out of bounds!')

    return pixel_value


def compute_imperviousness(row, raster_data, transform):
    location = row['geometry']
    return check_imperviousness(location, raster_data, transform)


def add_imperviousness_column(shapefile, imperviousness_raster, columnname):
    # Load raster data and transform once
    raster_data, transform = load_raster_data(imperviousness_raster)

    # Load shapefile data
    gdf = gpd.read_file(shapefile)

    # Use tqdm with progress_apply
    tqdm.pandas(desc="Computing imperviousness for each point")

    # Pass raster_data and transform to avoid reopening the file each time
    gdf[columnname] = gdf.progress_apply(
        compute_imperviousness,
        axis=1,
        raster_data=raster_data,
        transform=transform
    )

    # Save updated shapefile
    gdf.to_file(shapefile)
    print(f"Updated shapefile saved to: {shapefile}")


# imperviouness_data = 'input/imperviousness/imperv_reproj_28992.tif'
# add_imperviousness_column(output_pts, imperviouness_data, 'imperv')


# gdf = gpd.read_file(output_pts)
# print("Columns in the shapefile:", gdf.columns.tolist())
# missing_imperv_count = gdf['imperv'].isna().sum()
# print(f"Missing imperviousness is {missing_imperv_count} ")
# gdf.to_csv("only_to_check.csv", index=False)
# gdf = gpd.read_file(output_pts)
# print("Columns in the shapefile:", gdf.columns.tolist())
# missing_imperv_count = gdf['imperv'].isna().sum()
# print(f"Missing imperviousness is {missing_imperv_count} ")
# unique_counts = gdf['imperv'].value_counts()
# print(unique_counts)
# gdf.to_csv("only_to_check.csv", index=False)

# VISIBILITY------------------------------------------------------------------------------------------------------------
# Before running visibility, it needs to run the viewshed function first, which implies running the metric function first
visiblitlity_data = 'output/visibility/KanaalVanWalcheren/combined_viewshed.tif'
add_imperviousness_column(output_pts, visiblitlity_data, 'visible')

gdf = gpd.read_file(output_pts)
print("Columns in the shapefile:", gdf.columns.tolist())
visible_counts = gdf['visible'].value_counts(dropna=False)

# Get the counts for 0.0, 1.0, and other values (including NaN)
count_0 = visible_counts.get(0.0, 0)
count_1 = visible_counts.get(1.0, 0)
count_nan = gdf['visible'].isna().sum()  # Count NaN values
count_other = visible_counts.sum() - count_0 - count_1

# Print the results
print(f"Count of rows with 'visible' = 0.0: {count_0}")
print(f"Count of rows with 'visible' = 1.0: {count_1}")
print(f"Count of rows with 'visible' = NaN (missing): {count_nan}")
print(f"Count of rows with 'visible' = other or missing: {count_other}")

gdf.to_csv('test.csv', index = False)