"""
Intersect cross-section with building data.
Input: cross-section and building data folder
Output: List [intersection linestring, max height, building id] and closest_intersection Point

TO DO:

"""
import geopandas as gpd
from shapely.geometry import box
from shapely import LineString, MultiPolygon, wkt
from shapely.geometry import Polygon, Point, MultiLineString, MultiPoint
from shapely.ops import transform, nearest_points
import fiona
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cross_section_extraction import cross_section_extraction
from profile_extraction import profile_extraction
from OSM_data_river_retrieval import fetch_river_overpass


def tile_intersection(cross_section, fgb_path, folder_path):
    """
    Gives the filenames of the tiles that intersect with a linestring (cross_section)
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

    # Check which tiles intersect with the bounding box
    intersecting_tiles = tile_index[tile_index.geometry.intersects(bbox)]
    # num_items = len(intersecting_tiles)
    # print(f'Number of intersecting tiles: {num_items}')

    tile_ids = []
    for index, row in intersecting_tiles.iterrows():
        tile_id = row['tile_id']
        tile_id = tile_id.replace("/", "-")
        tile_ids.append(tile_id)

    all_files = os.listdir(folder_path)
    full_filenames = []
    for partial_name in tile_ids:
        for file in all_files:
            if file.startswith(partial_name) and file.endswith('_transformed.gpkg'):
                full_filenames.append(file)

    return full_filenames


def get_max_height(geometry):
    """Helper function to get max height from a Polygon or MultiPolygon"""
    if isinstance(geometry, Polygon):
        return np.max([coord[2] for coord in geometry.exterior.coords])
    elif isinstance(geometry, MultiPolygon):
        return np.max([np.max([coord[2] for coord in poly.exterior.coords]) for poly in geometry.geoms])
    else:
        raise ValueError(f"Unexpected geometry type: {type(geometry)}")


def get_intersection(cross_section, fgb_path, folder_path_transformed_tiles, crs="EPSG:28992"):
    """
    Finds the cross-section intersection with building. First finds the tiles the cross_section intersects with by using tile_intersection.
    Intersection is a linestring as we work with footprints (lod13_2d). Intersecting with 3d layer gives a geometrycollection
    :param cross_section: Linestring of the cross-section
    :param fgb_path: path to fgb file describing the tile extends and ids
    :param folder_path_transformed_tiles: Folder with all the tiles I retrieve before that intersect with the bbox of the river
    :param crs: 28992
    :return: List of intersection: [intersection linestring, max height, building identification] and the closes_intersection Point object
    TODO:
    """
    tiles = tile_intersection(cross_section, fgb_path, folder_path_transformed_tiles)
    # tiles = ['7-80-352_transformed.gpkg']
    intersections = []

    for tile in tiles:
        # Load layers of the tile
        gdf_lod13 = gpd.read_file(folder_path_transformed_tiles + '/' + tile, layer='lod13_2d')
        gdf_lod22 = gpd.read_file(folder_path_transformed_tiles + '/' + tile, layer='lod22_3d')

        # Ensure both GeoDataFrames have the same CRS as the cross_section
        gdf_lod13 = gdf_lod13.to_crs(crs)
        gdf_lod22 = gdf_lod22.to_crs(crs)

        # Find intersecting buildings (with footprint, if i use lod22 i get a geometrycollection and am not completely sure what is in there, line line point with different heights)
        intersecting_buildings = gdf_lod13[gdf_lod13.intersects(cross_section)]
        print('intersecting buildings ', intersecting_buildings)

        for _, building in intersecting_buildings.iterrows():
            # intersection = cross_section.intersection(building.geometry)

            # # Get max height from lod13_2d (what does this height mean)
            ground_height = get_max_height(building.geometry)

            # Get max height from lod22_3d
            lod22_building = gdf_lod22[gdf_lod22['identificatie'] == building['identificatie']]
            if not lod22_building.empty:
                max_height = get_max_height(lod22_building.iloc[0].geometry)  # Use the helper function
            else:
                # If no matching building in lod22, use the max height from lod13
                max_height = ground_height

            # Get the intersection
            intersection = building.geometry.intersection(cross_section)
            intersections.append([intersection, max_height, building['identificatie']])

    # To get the closest intersection, first select the first point in the cross-section and then take the nearest
    first_point = Point(cross_section.coords[0])
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

    # print('Intersections ', intersections)
    # print('Closest intersection ', closest_intersection)

    # This saves the closest_intersection point as a shapefile. IDK i don't need this.
    gdf_point = gpd.GeoDataFrame(geometry=[closest_intersection])
    gdf_point.set_crs(epsg=28992, inplace=True)
    # shapefilepath = 'TESTPOINTBOOOEEEHDELETEME.shp'
    # gdf_point.to_file(shapefilepath)

    return intersections, closest_intersection

