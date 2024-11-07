"""
This script computes the cross-sectional lines, begin point and end point, along a shapefile line, in this case a river.


input: River line shapefile
output: shapefiles and gdfs of all cross-sections

"""
import os
import rasterio
import pandas as pd
from tqdm import tqdm
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
from shapely.geometry import Point, box


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
    midpoints['width'] = np.nan #Extra width for viewshed

    # Initialize embankments points
    embankment_points_left = gpd.GeoDataFrame(columns=[])
    embankment_points_left['geometry'] = np.nan
    embankment_points_left['h_distance'] = np.nan
    embankment_points_left.set_crs("EPSG:28992", inplace=True)
    embankment_points_right = gpd.GeoDataFrame(columns=[])
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

        # Search left of the midpoint for the first valid elevation value
        for i in range(midpoint_idx - 1, -1, -1):
            if not pd.isna(cross_section_points.loc[i, 'elev_dtm']):
                left_value = cross_section_points.loc[i, 'elev_dtm']
                left_geom = cross_section_points.loc[i, 'geometry']
                left_hdist = cross_section_points.loc[i, 'h_distance']
                print(f"left geom is {left_geom} and type {type(left_geom)}")
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
        max_value_hdist = max(
            filter(None, [left_hdist, right_hdist])) if left_hdist is not None or right_hdist is not None else None
        midpoints.at[idx, 'max'] = max_value
        midpoints.at[idx, 'height'] = max_value + user_defined_height if max_value is not None else None
        # Add this width to radius of viewshed
        midpoints.at[idx, 'width'] = 2 * max_value_hdist

        # Save the result to a new shapefile
    midpoints.to_file(midpoints_file, driver='ESRI Shapefile')
    embankment_points_left.to_file(embankments_file_left, driver="ESRI Shapefile")
    print(f"Results saved to {midpoints_file} and {embankments_file_left}")
    embankment_points_right.to_file(embankments_file_right, driver="ESRI Shapefile")
    print(f"Results saved to {midpoints_file} and {embankments_file_right}")


# Get correct cross-section outward from embankment points--------------------------------------------------------------
def get_outward_cross_sections(riverline, embank_points, width, output_file):
    """
    Gets the outward cross-sections from the boundary points pointing outwards with respect to the river
    Args:
        riverline: Shapefile of river
        boundary_points: Shapefile of embankment points
        width: width of section (100m)
        output_file: Shapefile to write output to

    Returns:

    """
    gdf_riv = gpd.read_file(riverline)
    gdf_pts = gpd.read_file(embank_points)
    cross_sections = []
    for pt in gdf_pts:
        cs = get_outward_cross_section(gdf_riv, pt, width)
        cross_sections.append(cs)

    gdf = gpd.GeoDataFrame(geometry=cross_sections)
    gdf.set_crs(epsg=28992, inplace=True)
    gdf.to_file(output_file, driver='ESRI Shapefile')
    return


def get_outward_cross_section(line, embank_point, width):
    """
    Generates a cross-Section that points outward from the riverline at a given boundary point.

    Parameters:
    line (LineString): The river centerline geometry.
    boundary_point (Point): The point on the river boundary where the cross-section should originate.
    width (float): The desired width of the cross-section.

    Returns:
    LineString: The outward-pointing cross-section geometry.
    """
    # Get the tangent angle at the boundary point
    small_distance = 0.001
    if line.project(embank_point) + small_distance <= line.length:
        point_ahead = line.interpolate(line.project(embank_point) + small_distance)
    else:
        point_ahead = line.interpolate(line.length)

    x0, y0 = embank_point.x, embank_point.y
    x1, y1 = point_ahead.x, point_ahead.y
    angle = np.arctan2(y1 - y0, x1 - x0)

    # Calculate the outward-pointing line endpoints
    dx = width / 2 * np.cos(angle)
    dy = width / 2 * np.sin(angle)
    p1 = Point(x0 + dx, y0 + dy)
    p2 = Point(x0 - dx, y0 - dy)

    return LineString([p1, p2])

# Parameters, input and output files-----------------------------------------------------------------------------------
river = "input/river/longest_river.shp"
interval = 50
width1 = 200
output_cs_preproces = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_cs_pre.shp"
river_midpoints = "output/river/KanaalVanWalcheren/KanaalVanWalcheren_mid.shp"
n_points = 2 * width1
points_cs_preprocess = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_pts_cs_pre.shp"
tiles_folder_dtm = "input/AHN/KanaalVanWalcheren/DTM"
tiles_folder_dsm = "input/AHN/KanaalVanWalcheren/DSM"
embankments_file_left = "output/embankment/KanaalVanWalcheren/left.shp"
embankments_file_right = "output/embankment/KanaalVanWalcheren/right.shp"
user_defined_height = 1.75
width = 100
number_points = 2 * width
output_cs_l = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_cs_left.shp"
output_cs_r = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_cs_right.shp"
output_pts_l = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_points_left.shp"
output_pts_r = "output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_points_right.shp"


# RUN-------------------------------------------------------------------------------------------------------------------
cross_section_extraction(river, interval, width1, output_cs_preproces, river_midpoints)
create_cross_section_points(output_cs_preproces, n_points, points_cs_preprocess)
add_elevation_from_tiles(points_cs_preprocess, tiles_folder_dtm, 'elev_dtm')
extract_max_elevation_values(points_cs_preprocess, river_midpoints, embankments_file_left, embankments_file_right, user_defined_height)

# get_outward_cross_sections(river, embankments_file_right, width, output_cs_r)
# create_cross_section_points(output_cs_r, number_points, output_pts_r)
# add_elevation_from_tiles(output_pts_r, tiles_folder_dtm, 'elev_dtm')
# add_elevation_from_tiles(output_pts_r, tiles_folder_dsm, 'elev_dsm')
#
# get_outward_cross_sections(river, embankments_file_left, width, output_cs_l)
# create_cross_section_points(output_cs_l, number_points, output_pts_l)
# add_elevation_from_tiles(output_pts_l, tiles_folder_dtm, 'elev_dtm')
# add_elevation_from_tiles(output_pts_l, tiles_folder_dsm, 'elev_dsm')
