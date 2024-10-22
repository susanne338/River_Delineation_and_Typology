"""
Intersect cross-sections with buildigs
Input: building geopackage and river line shapefile. interval. buffer width
Output:

TO DO: boundary line doesn't work because cross-sections do not have correct orientation

"""
import geopandas as gpd
from shapely.geometry import box
from shapely import LineString, MultiPolygon
from shapely.geometry import Polygon, Point, MultiLineString
from shapely.ops import transform
import fiona
import os
import numpy as np


def drop_z(geom):
    """
    Drops the Z dimension from a shapely geometry.
    """
    if geom.has_z:
        return transform(lambda x, y, z=None: (x, y), geom)
    return geom


# GET THE INTERSECTION OF THE CROSS-SECTION WITH BUILDING DATA----------------------------------------------------------
def tile_intersection(cross_section, fgb_path, folder_path):
    """
    Gives the filenames of the tiles that intersect with a lineastring (cross_section)
    :param cross_section: Linestring
    :param fgb_path: indices
    :param folder_path: all transformed tiles
    :return: list of names of files of tiles
    """
    cross_section_gdf = gpd.GeoDataFrame(geometry=[cross_section])
    cross_section_geometry = cross_section_gdf.geometry.unary_union

    minx, miny, maxx, maxy = cross_section_geometry.bounds
    bbox = box(minx, miny, maxx, maxy)

    # Load the tile index from the .fgb file
    with fiona.open(fgb_path) as src:
        tile_index = gpd.GeoDataFrame.from_features(src)
    # print("Column names in the GeoDataFrame:", tile_index.columns.tolist())
    # print(tile_index.head())
    # # Display the summary of the GeoDataFrame
    # tile_index.info()

    # Check which tiles intersect with the bounding box
    intersecting_tiles = tile_index[tile_index.geometry.intersects(bbox)]
    num_items = len(intersecting_tiles)
    # print(f'Number of intersecting tiles: {num_items}')
    # print('first index ', intersecting_tiles)

    tile_ids = []
    for index, row in intersecting_tiles.iterrows():
        tile_id = row['tile_id']
        tile_id = tile_id.replace("/", "-")
        # print('tile id ', tile_id)
        tile_ids.append(tile_id)

    all_files = os.listdir(folder_path)
    full_filenames = []
    for partial_name in tile_ids:
        for file in all_files:
            if file.startswith(partial_name) and file.endswith('_transformed.gpkg'):
                full_filenames.append(file)

    # print('full filenames ', full_filenames)
    return full_filenames


# tile_intersection(line, tile_index_path, '3dbag_tiles_layers')
# tile_name = '7-80-352_transformed.gpkg'

def get_max_height(geometry):
    """Helper function to get max height from a Polygon or MultiPolygon"""
    if isinstance(geometry, Polygon):
        return np.max([coord[2] for coord in geometry.exterior.coords])
    elif isinstance(geometry, MultiPolygon):
        return np.max([np.max([coord[2] for coord in poly.exterior.coords]) for poly in geometry.geoms])
    else:
        raise ValueError(f"Unexpected geometry type: {type(geometry)}")


def get_intersection(cross_section_half, fgb_path, folder_path_transformed_tiles, crs="EPSG:28992"):
    """
    Finds the cross-section intersection with building. First finds the tiles the cross_section intersects with by using tile_intersection.
    :param cross_section: Linestring of the cross-section
    :param fgb_path: path to fgb file describing the tile extends and ids
    :param folder_path_transformed_tiles: Folder with all the tiles I retrieve before that intersect with the bbox of the river tiles of 3DBAG
    :param crs:
    :return: List of intersection: [intersection linestring, max height, building identification] and the closes_intersection point
    TODO: I am intersecting with the 2D building data now, but interseting with 3D data gies me a geometrycollection
    closest intersection now takes a linestring because we work with footprints.
    """
    tiles = tile_intersection(cross_section_half, fgb_path, folder_path_transformed_tiles)
    # tiles = ['7-80-352_transformed.gpkg']
    intersections = []

    for tile in tiles:
        gdf_lod13 = gpd.read_file(folder_path_transformed_tiles + '/' + tile, layer='lod13_2d')
        gdf_lod22 = gpd.read_file(folder_path_transformed_tiles + '/' + tile, layer='lod22_3d')

        # Ensure both GeoDataFrames have the same CRS as the cross_section
        gdf_lod13 = gdf_lod13.to_crs(crs)
        gdf_lod22 = gdf_lod22.to_crs(crs)

        # Find intersecting buildings (with footprint, if i use lod22 i get a geometrycollection and am not completely sure what is in there, line line point with different heights)
        intersecting_buildings = gdf_lod13[gdf_lod13.intersects(cross_section_half)]
        print('intersecting buildings ', intersecting_buildings)

        for _, building in intersecting_buildings.iterrows():
            intersectionTEST = cross_section_half.intersection(building.geometry)
            print('building intersection test ', intersectionTEST)
            # Get ground height from lod13
            ground_height = get_max_height(building.geometry)
            # print('building ', building)
            print('type building ', type(building))

            # print('ground height ', ground_height)

            # Get max height from lod22_3d
            lod22_building = gdf_lod22[gdf_lod22['identificatie'] == building['identificatie']]
            if not lod22_building.empty:
                max_height = get_max_height(lod22_building.iloc[0].geometry)  # Use the helper function
                print('max height ', max_height)
            else:
                # If no matching building in lod22, use the max height from lod13
                max_height = ground_height

            # Get the intersection
            intersection = building.geometry.intersection(cross_section_half)
            intersections.append([intersection, max_height, building['identificatie']])
            print('intersection ', intersection)

    # To get the closest intersection, first select the first point in the cross section and then take the nearest
    first_point = Point(cross_section_half.coords[0])
    closest_intersection = None
    min_distance = float('inf')

    #  I need to check if there is a linestring or a multilinestring in it
    for i in intersections:
        linestring = i[0]  # Access the intersection part (LineString or MultiLineString)

        # Check if it's a LineString or MultiLineString
        if isinstance(linestring, LineString):
            # Process the single LineString
            for coord in linestring.coords:
                point = Point(coord)
                distance = first_point.distance(point)
                if distance < min_distance:
                    min_distance = distance
                    closest_intersection = point

        elif isinstance(linestring, MultiLineString):
            # Process each LineString within the MultiLineString
            for sub_linestring in linestring.geoms:
                for coord in sub_linestring.coords:
                    point = Point(coord)
                    distance = first_point.distance(point)
                    if distance < min_distance:
                        min_distance = distance
                        closest_intersection = point

    print('Intersections ', intersections)
    print('Closest intersection ', closest_intersection)

    gdf_point = gpd.GeoDataFrame(geometry=[closest_intersection])
    gdf_point.set_crs(epsg=28992, inplace=True)
    # shapefilepath = 'TESTPOINTBOOOEEEHDELETEME.shp'
    # gdf_point.to_file(shapefilepath)

    return intersections, closest_intersection


# get_intersection(line, tile_index_path, '3dbag_tiles_layers')

def process_cross_sections(cross_sections, riverline, buffer_distance, buffer_file, fgb_path, folder_path_transformed_tiles, bp_file):
    """
    Processes all the cross-sections by checking for intersections with buildings and then returns the  boundary points of the river space delineated by the first line of buildings.
    :param cross_sections: gdf of all cross-sections
    :param riverline: gdf of 1 river
    :param buffer_distance: set 100m
    :param buffer_file: File to output the buffer polygon to (.shp)
    :param fgb_path: the path to the file that contains the tile indices for the tile_intersection function
    :param folder_path_transformed_tiles: Folder containing the tiles with the transformed nodata values to -246 (i think)
    :param bp_file: output file for the boundary points (.shp)
    :return: geodataframe of boundary points, boundary_points list
    """
    boundary_points = []
    river = riverline.geometry.iloc[0]
    river_buffer = river.buffer(buffer_distance)
    print("type river buffer and riverline ", type(river_buffer), type(riverline.geometry))

    # save buffer to shapefile
    river_buffer_gdf = gpd.GeoDataFrame(geometry=[river_buffer], crs="EPSG:28992")
    river_buffer_gdf.to_file(buffer_file)
    print(" made the river buffer :)")

    # Loop through the cross-sections
    for index, line in cross_sections.iterrows():
        line_geom = line.geometry
        print('index ', index)
        print('line geom  ', line_geom)

        river_point = river.interpolate(river.project(Point(line_geom.coords[0])))
        intersections, closest_building_point = get_intersection(line_geom, fgb_path, folder_path_transformed_tiles)
        print('closest building point, type ', closest_building_point, type(closest_building_point))

        if closest_building_point == None:
            # print('Closest boundary point is none')
            # buffer_intersection = line.intersection(river_buffer.iloc[0].boundary)
            buffer_intersection = line_geom.intersection(river_buffer.boundary)
            # Check if buffer_intersection is empty
            if buffer_intersection.is_empty:
                print("Warning: Buffer intersection is empty, not appending anything.")
                continue  # Skip the rest of the loop

            print('Buffer intersection is ', buffer_intersection)
            if isinstance(buffer_intersection, Point):
                boundary_points.append(buffer_intersection)
            elif isinstance(buffer_intersection, LineString):
                points = list(buffer_intersection.coords)
                if not points:
                    continue
                else:
                    points.sort(key=lambda p: Point(p).distance(river_point))
                    boundary_points.append(Point(points[0]))
            else:
                print(f"Unexpected buffer intersection type: {type(buffer_intersection)}")
                continue
        else:
            boundary_points.append(closest_building_point)


    # Create a GeoDataFrame from the list of Point objects
    gdf = gpd.GeoDataFrame(geometry=boundary_points, crs=28992)

    # Save the GeoDataFrame (as a shapefile)
    gdf.to_file(bp_file)
    return gdf, boundary_points

# def order_points_along_river_buffer(points, riverline):
#     """
#     Order points by projecting them onto the boundary of a river buffer polygon.
#
#     :param points: List of Point objects representing boundary points
#     :param river_buffer: Polygon object representing the river buffer
#     :return: LineString of ordered points projected onto the buffer boundary
#     """
#     combined_river = riverline.unary_union
#     river_buffer = combined_river.buffer(buffer_distance)
#     buffer_boundary = river_buffer.exterior
#
#     # Project each point onto the buffer boundary using list comprehension
#     projected_points = [nearest_points(point, buffer_boundary)[1] for point in points]
#
#     # Calculate the distance of each projected point along the buffer boundary
#     distances = [buffer_boundary.project(point) for point in projected_points]
#
#     # Sort the projected points based on these distances
#     sorted_data = sorted(zip(distances, projected_points), key=lambda x: x[0])
#
#     # Create a LineString from the sorted, projected points
#     ordered_boundary = LineString([point for _, point in sorted_data])
#
#     return ordered_boundary
def order_points_along_river(points, river_line):
    # Project each point onto the river line
    projected_points = [river_line.interpolate(river_line.project(point)) for point in points]
    print('points ', points)

    # Calculate the distance of each projected point along the river line
    distances = [river_line.project(point) for point in projected_points]
    distances = [dist.iloc[0] for dist in distances]
    print('distances ', distances)

    # Sort the original points based on these distances
    # sorted_points = [point for _, point in sorted(zip(distances, points))]
    sorted_points = [point for _, point in sorted(zip(distances, points), key=lambda x: x[0])]
    # key lambda means it will sort on the first element of the tuple (distance, point)
    # _ is used for ignoring certain values when unpacking tuples or lists. so we ignore distances here

    return sorted_points

def determine_side(line, river_line):
    # Use the midpoint of the line to determine which side it's on
    midpoint = Point(line.interpolate(0.5, normalized=True))
    print("midpoint, line, riverline, riverline project ", midpoint, line, river_line, river_line.project(midpoint))
    river_point = Point(river_line.interpolate(river_line.project(midpoint)))

    # Create a vector from the river point to the midpoint
    vector = np.array([midpoint.x - river_point.x, midpoint.y - river_point.y])

    # Get the direction of the river at this point
    river_direction = np.array(river_line.coords[1]) - np.array(river_line.coords[0])
    # vector_3d = np.append(vector, 0)
    # river_direction_3d = np.append(river_direction, 0)
    # Calculate the cross product
    cross_product = np.cross(river_direction, vector)

    # If cross product is positive, the line is on the left; if negative, it's on the right
    return 'left' if cross_product > 0 else 'right'

#  This function calls order_points_along_river and determine_side, and is called by boundary_line_smooth
# def create_boundary_line(points, river_line):
#     side = 'left'
#     # Order the points along the river
#     ordered_points = order_points_along_river(points, river_line)
#
#     # Create a line from the ordered points
#     boundary_line = LineString(ordered_points)
#
#     # Determine which side the boundary line is on
#     # current_side = determine_side(boundary_line, river_line)
#
#     # If it's not on the correct side, reverse the line
#     # if current_side != side:
#     #     boundary_line = LineString(reversed(ordered_points))
#
#     return boundary_line


# SMOOTHING-------------------------------------------------------------------------------------------------------------
def is_visible(point, river_point, buildings):
    """
    Determines if a point is visible from the river by checking if the line between the river and the point
    intersects any buildings.
    """

    # point = (point.x, point.y)
    # print("point before ", point)
    # print('riverpoint before ', river_point)
    # Check if point and river_point are 3D (Point Z) or 2D (Point)
    if point.has_z:  # Drop the z-coordinate if present
        point = (point.x, point.y)
    else:
        point = (point.x, point.y)

    if river_point.has_z:  # Drop the z-coordinate if present
        river_point = (river_point.x, river_point.y)
    else:
        river_point = (river_point.x, river_point.y)
    ray = LineString([river_point, point])
    # print("point after", point)
    # print('riverpoint after ', river_point)

    for building in buildings.geometry:
        if ray.intersects(building):
            return False  # The point is occluded by a building
    return True  # The point is visible


def boundary_line_smooth(boundary_points, riverline, buildings, output_file):
    print('length points  ', len(boundary_points))

    filtered_boundary_points = []
    for i, point in enumerate(boundary_points):
        river = riverline.geometry.iloc[0]
        river_point = river.interpolate(river.project(point))
        if is_visible(point, river_point, buildings):
            filtered_boundary_points.append(point)
            print(i, ' visible')
        else:
            print(i, ' not visible')
    filtered_boundary_points_gdf = gpd.GeoDataFrame(geometry=filtered_boundary_points)
    filtered_boundary_points_gdf.to_file('filtered_bp.shp')
    boundary_smooth = order_points_along_river(filtered_boundary_points, riverline)

    gdf = gpd.GeoDataFrame(geometry=boundary_smooth, crs="EPSG:28992")
    gdf.to_file(output_file)

    return boundary_smooth

def boundary_line_polygon(boundary_line, river, output_file):
    """
    Create a polygon by connecting the entire river line and the boundary line of the building.

    Parameters:
    - boundary_line: The building's boundary side (as a LineString).
    - river_line: The river's geometry (LineString).

    Returns:
    - Polygon object representing the area between the boundary and the river.
    TODO: this needs to take river edge instead i think
    """
    # Get the points of the boundary and river lines
    boundary_points = list(boundary_line.coords)
    river_points = list(river.coords)

    # Combine the boundary points and the river points (in reverse order to close the polygon correctly)
    # polygon_points = boundary_points + river_points[::-1]
    polygon_points = boundary_points + river_points
    polygon = Polygon(polygon_points)

    gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=28992)
    gdf.to_file(output_file)

    # Create and return the closed polygon
    return polygon



buffer_distance = 100
buildings = '3DBAG_combined_tiles/combined_3DBAG.gpkg'
river = "river_shapefiles/longest_river.shp"
gdf_river = gpd.read_file(river)
gdf_cs1 = gpd.read_file('cross_sections/cs_1_interval0_5/cs_1_interval0_5.shp')
gdf_cs2 = gpd.read_file('cross_sections/cs_2_interval0_5/cs_2_interval0_5.shp')
folder_3DBAG_transformed_tiles = "3DBAG_tiles"
path_to_fgb_tile_index = '../needed_files/tile_index.fgb'
output_file_boundary_points= "boundary_points_buildings/cs_1_bp"
buffer_output_file = "../boundary_points_buildings/buffer.shp"

gdf, boundary_points = process_cross_sections(gdf_cs1, gdf_river, buffer_distance, buffer_output_file, path_to_fgb_tile_index, folder_3DBAG_transformed_tiles, output_file_boundary_points)


#     boundary_line_1 = boundary_line_smooth(boundary_points_1, gdf_river, buildings, '3DBAGDATATEST/boundarylinesmooth1.shp', index_list_out_1)
#     # river_space_polygons_1 = boundary_line_polygon(boundary_line_1, gdf_river, '3DBAGDATATEST/polygon1')
#
#     # boundary_points_2 = process_cross_sections(cross_sec_2, buildings, gdf_river, buffer_distance,tile_index_path, folder)
#     # boundary_line_2 = boundary_line_smooth(boundary_points_2, river, buildings, '3DBAGDATATEST/boundarylinesmooth2.shp')
#     # river_space_polygons_2 = boundary_line_polygon(boundary_line_2, river, '3DBAGDATATEST/polygon2')
