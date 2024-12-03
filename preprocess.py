"""
Execute data_retrieval_river script before to get river input file
This script gets all the cross-sections, retrieves the data, and adds all the parameters.

execute visibility after

INPUT:
river and city names, river shapefile.


This script performs preprocess:
1. download dtm,dsm,bgt,3dbag files
2. Compute wide cross-sections and points
3. Adds elevation to the dataframe elev_dtm
4. Finds embankment points by selecting the first value elev_dtm that is not None from the midpoint. When midpoint has value, we assume its a bridge and segment is invalid
5. Cut segment 100 meters from embankment points



OUTPUT:
Folder of DTM tif files
midpoint file
embankment file
cross-section segments
points of each segment
todo: add elevation now also takes the merged dsm, maybe I should put that in different folder. But thats why it takes a bit longer
"""
import math
import os
import rasterio
import pandas as pd
import statistics
from tqdm import tqdm
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np


# Get cross_sections intersecting with river---------------------------------------------------------------------------
def cross_section_extraction(river_file, interval, width, output_file_cs, output_river_mid):
    """

    Args:
        river_file: shapefile path of river
        interval: distance between segments
        width: width of segment
        output_file_cs: file path to write cross-sections to
        output_river_mid: file path to write midpoints to

    Returns: cross-sections shapefile, midpoints shapefile

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

    Args:
        line: LineString of river
        distance_along_line: Position on river we are at
        width: Desired width of cross-section

    Returns: LineString containing cross-section, midpoint of cross-section

    """
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

    Args:
        cross_sections_shapefile: Path to shapefile containging cross-section lines
        width: widht of crossesction. Used to determine number of points along lines. Points are seperated by 0.5m
        output_shapefile: Path where the resulting points shapefile will be saved

    Returns: GeoDataFrame containing points with h_distance

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
    GeoDataFrame with added elevation column. Prints count of missing points
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
def extract_max_elevation_values(points, midpoints_file,
                                 user_defined_height, embankments_shp):
    """
    Extracts the elevation values of the embankments, and selects the max value for the viewshed analysis
    This max value is then added up with user_defined_height.
    Takes embankment h_dist and selects max, multiplies this my 2, then add this as 'width' to add to radius in viewshed analysis
    Args:
        shapefile: points shapefile path
        midpoints_file: cross-section midpoint file that is altered
        user_defined_height: viewpoint height to perform viewshed analysis from
        embankments_shp: path to file to write embankment points to

    Returns: saves values left, right (embankment), max, and (viewpoint) height to midpoints_file
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
        # midpoint_geom = row.geometry
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
        l_split_h_di = None
        r_split_h_di = None

        if not pd.isna(cross_section_points.loc[midpoint_idx, 'elev_dtm']):
            print(f"Segments is probably a bridge. id is {midpoint_idx}")
            embankment_pts.append([idx, 0, left_geom, left_value, left_hdist, l_split_h_di])
            embankment_pts.append([idx, 1, right_geom, right_value, right_hdist, r_split_h_di])
            continue

        # Search left of the midpoint for the first valid elevation value
        for i in range(midpoint_idx - 1, -1, -1):
            if not pd.isna(cross_section_points.loc[i, 'elev_dtm']):
                left_value = cross_section_points.loc[i, 'elev_dtm']
                left_geom = cross_section_points.loc[i, 'geometry']
                left_hdist = cross_section_points.loc[i, 'h_distance']
                break

        # Search right
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

    # Save the result to shapefile
    midpoints.to_file(midpoints_file, driver='ESRI Shapefile')
    print(f"average width river is {statistics.mean(distance_list)} ")

    gdf_clos = gpd.GeoDataFrame(embankment_pts, columns=['id', 'side', 'geometry', 'height', 'h_dist', 'split_h_di'])
    gdf_clos.set_crs("EPSG:28992", inplace=True)
    gdf_clos.to_file(embankments_shp, driver="ESRI Shapefile")


def cut_segment(embankments_shp, points_shp, pts_outputfile, cs_outputfile, segment_width):
    """
    Cuts of ends longer than 100m from embankement
    Args:
        embankments_shp:
        points_shp:
        pts_outputfile:
        cs_outputfile:
        segment_width:

    Returns:

    """
    embankment = gpd.read_file(embankments_shp)
    points = gpd.read_file(points_shp)

    filtered_cs = []
    filtered_pts = []
    filtered_embankments = []

    # grouped_pts = points.groupby('id')
    # for idx, segment in enumerate(grouped_pts):
    for id, segment in points.groupby('id'):
        # id = segment.iloc[0]['id']
        print(f'id is {id}')
        # left_embankment_hdist = embankment.loc[(embankment['id'] == id) & (embankment['side'] == 0), 'h_dist'].iloc[0]
        # right_embankment_hdist = embankment.loc[(embankment['id'] == id) & (embankment['side'] == 1), 'h_dist'].iloc[0]
        left_embankment = embankment.loc[(embankment['id'] == id) & (embankment['side'] == 0)]
        right_embankment = embankment.loc[(embankment['id'] == id) & (embankment['side'] == 1)]

        # if left_embankment_hdist is None:
        #     #invalid segment, skip
        #     print(f"left embankment distance value is {left_embankment_hdist}, thus bridge?")
        #     continue
        if math.isnan(left_embankment['h_dist'].iloc[0]):
            #invalid segment, skip
            print(f"left embankment is math.nan")
            continue

        left_embankment_hdist = left_embankment['h_dist'].iloc[0]
        right_embankment_hdist = right_embankment['h_dist'].iloc[0]

        min_distance = left_embankment_hdist - segment_width
        max_distance = right_embankment_hdist + segment_width
        print(f"min and max distances: {min_distance}, {max_distance}")

        points_filtered = segment[(segment['h_distance'] < max_distance) & (segment['h_distance'] > min_distance)]

        #re-start h_diatnce at 0 by subtracting min value
        min_h_distance = points_filtered['h_distance'].min()
        points_filtered['h_distance'] = points_filtered['h_distance'] - min_h_distance
        filtered_pts.append(points_filtered)
        # also for embankment
        left_embankment_filtered = left_embankment.copy()
        right_embankment_filtered = right_embankment.copy()

        left_embankment_filtered['h_dist'] = left_embankment_filtered['h_dist'] - min_h_distance
        right_embankment_filtered['h_dist'] = right_embankment_filtered['h_dist'] - min_h_distance

        filtered_embankments.append(left_embankment_filtered)
        filtered_embankments.append(right_embankment_filtered)

        # Cross-sections
        max_id, min_id = points_filtered['h_distance'].idxmax(), points_filtered['h_distance'].idxmin()
        geometry_max, geometry_min = points_filtered.loc[max_id, 'geometry'], points_filtered.loc[min_id, 'geometry']
        cs = LineString([geometry_min, geometry_max])
        filtered_cs.append([id, cs])


    all_filtered_pts = pd.concat(filtered_pts)
    gdf_filtered_pts = gpd.GeoDataFrame(all_filtered_pts, geometry='geometry')
    gdf_filtered_pts.to_file(pts_outputfile, driver= 'ESRI Shapefile')


    all_filtered_embankments = pd.concat(filtered_embankments)
    gdf_filtered_embankments = gpd.GeoDataFrame(all_filtered_embankments, geometry='geometry')
    gdf_filtered_embankments.to_file(embankments_shp, driver='ESRI Shapefile')

    df_cs = pd.DataFrame(filtered_cs, columns=['id', 'geometry'])
    gdf_cs = gpd.GeoDataFrame(df_cs, geometry='geometry')
    gdf_cs.to_file(cs_outputfile, driver='ESRI Shapefile')


def run_preprocess(river, city):
    """
    Runs functions in this script
    Args:
        river: string
        city: string

    Returns:

    """
    river_file = f'input/river/{river}/{city}/{city}.shp'

    # PARAMETERS
    interval = 100
    width_preprocess = 700 #river can be 450 meters wide
    user_defined_height = 1.75
    width_segment = 100

    #FILE PATHS
    directories = [f"output/cross_sections/{river}/{city}/preprocess", f"output/river/{river}/{city}", f"output/cross_sections/{river}/{city}/final", f"output/embankment/{river}/{city}"]
    for dir in directories:
        os.makedirs(dir, exist_ok=True)

    cs_preprocess = f"output/cross_sections/{river}/{city}/preprocess/cross_sections.shp"
    pts_preprocess = f"output/cross_sections/{river}/{city}/preprocess/points.shp"
    cs_final = f"output/cross_sections/{river}/{city}/final/cross_sections.shp"
    pts_final = f"output/cross_sections/{river}/{city}/final/points.shp"

    river_midpoints = f"output/river/{river}/{city}/{city}_mid.shp"
    embankments_file = f"output/embankment/{river}/{city}/embankments.shp"
    tiles_folder_dtm = f'input/AHN/{river}/{city}/DTM'
    os.makedirs(tiles_folder_dtm, exist_ok=True)
    tiles_folder_dsm = f"input/AHN/{river}/{city}/DSM"
    os.makedirs(tiles_folder_dsm, exist_ok=True)


    # RUN
    # Get data
    # from data_retrieval import run_data_retrieval
    # run_data_retrieval(river, city)

    cross_section_extraction(river_file, interval, width_preprocess, cs_preprocess, river_midpoints)
    create_cross_section_points(cs_preprocess, width_preprocess, pts_preprocess)
    add_elevation_from_tiles(pts_preprocess, tiles_folder_dtm, 'elev_dtm')
    add_elevation_from_tiles(pts_preprocess, tiles_folder_dsm, 'elev_dsm')
    extract_max_elevation_values(pts_preprocess, river_midpoints,user_defined_height, embankments_file)
    cut_segment(embankments_file, pts_preprocess, pts_final, cs_final, width_segment)



if __name__ == "__main__":
    river = 'dommel'
    city = 'eindhoven'
    run_preprocess(river, city)