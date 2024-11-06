"""
This script computes the cross-sectional lines, begin point and end point, along a shapefile line, in this case a river.


input: River line shapefile
output: shapefiles and gdfs of all cross-sections

"""

import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np


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

    return gdf


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


# Collecting a lot of cross-sections for river space delineation
output_file_river = "input/river/maas.shp"
cross_section_extraction(output_file_river, 100, 250, "output/cross_sections/maas/cross_sections.shp", "output/cross_sections/maas/cross_sections_midpoints.shp" )
