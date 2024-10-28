"""
This script computes the cross-sectional lines, begin point and end point, along a shapefile line, in this case a river.


input: River line shapefile
output: shapefiles and gdfs of all cross-sections

"""

import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np


def cross_section_extraction(river, interval, width, output_gdf, output_file_1, output_file_2, output_river_mid):
    """
    Get cross-sections
    :param output_file_2:  output shapefile for cross-sections 2
    :param output_file_1:  output shapefile for cross-sections 1
    :param output_gdf: output shapefile for all cross-sections
    :param width: width of total cross-section
    :param interval: width between cross-sections
    :param river: River geodataframe
    :return: gdf, gdf_1, gdf_2, cross_sections_1, cross_sections_2, index_list
    """

    projected_gdf = river.to_crs(epsg=28992)
    # riverline = projected_gdf.geometry.iloc[0] # This takes the LineString from the geopanda geodataframe
    # print("riverline legth: ", riverline.length)
    cross_sections = []
    # cross_sections_1 = []
    # cross_sections_2 = []
    # index_list = []
    river_points = []
    for index, row in projected_gdf.iterrows():
        riverline = row['geometry']

        print("index, riverline length: ", index, riverline.length)

        for distance_along_line in np.arange(0, riverline.length, interval):
            cross_section, point_on_line = get_perpendicular_cross_section(riverline, distance_along_line, width)
            cross_sections.append(cross_section)
            river_points.append(point_on_line)

            # print('type ', type(cross_section)) #linestring
            # Split in two
            # p1 = cross_section.coords[0]
            # # print('point and type ', p1, type(p1)) #point is a tuple
            # p2 = cross_section.coords[-1]
            # midpoint = Point((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            # line1 = LineString([midpoint, p1])  # From the first point to the midpoint
            # line2 = LineString([midpoint, p2])  # From the midpoint to the second point
            # cross_sections_1.append(line1)
            # cross_sections_2.append(line2)
            # index_list.append(index)

    gdf = gpd.GeoDataFrame(geometry=cross_sections)
    gdf.set_crs(epsg=28992, inplace=True)
    # gdf_1 = gpd.GeoDataFrame(geometry=cross_sections_1)
    # # Set the correct Coordinate Reference System (CRS) - replace with your appropriate EPSG code
    # gdf_1.set_crs(epsg=28992, inplace=True)
    # gdf_2 = gpd.GeoDataFrame(geometry=cross_sections_2)
    # # Set the correct Coordinate Reference System (CRS) - replace with your appropriate EPSG code
    # gdf_2.set_crs(epsg=28992, inplace=True)

    # Save to a Shapefile (.shp)
    gdf.to_file(output_gdf, driver='ESRI Shapefile')
    # gdf_1.to_file(output_file_1, driver='ESRI Shapefile')
    # gdf_2.to_file(output_file_2, driver='ESRI Shapefile')

    # Save river mispoints to file
    river_points_gdf = gpd.GeoDataFrame(geometry=river_points)
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
output_file_river = "../river_shapefiles/longest_river.shp"
gdf_river = gpd.read_file(output_file_river)
cross_section_extraction(gdf_river, 0.5, 250, 'cross_sections/cs_interval0_5', '../cross_sections/cs_1_interval0_5',
                         'cross_sections/cs_2_interval0_5')
