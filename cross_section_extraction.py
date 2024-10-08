"""
This script computes the cross-sectional lines, begin point and end point, along a shapefile line, in this case a river.
It actually doesn't really use the DTM data, i SHOULD REMOVE THIS FROM THE SCRIPT AND PUT IT IN FUNCTION SUCH THAT i CAN CALL IT IN FIRSTLINEOFBUILDINGS

input: River line shapefile
output: shapefile of all cross-sections

TO DO:
remove DTM
Maybe here cut off cross-sections
"""

import geopandas as gpd
import rasterio
from shapely.geometry import LineString, Point
import numpy as np
from OSM_data_river_retrieval import fetch_river_overpass



def cross_section_extraction(river, interval, width):
    """
    Get cross-sections
    :param river: River geodataframe
    :return: crosssections: as geodataframe to put into profile_extraction
    """

    # if river[river.geometry.geom_type != 'lines']:
    #     print('river is not a line but probably polygon')
    #     return

    projected_gdf = river.to_crs(epsg=28992)
    # riverline = projected_gdf.geometry.iloc[0] # This takes the LineString from the geopanda geodataframe
    # print("riverline legth: ", riverline.length)

    cross_sections = []
    for index, row in projected_gdf.iterrows():
        riverline = row['geometry']
        print("index, riverline legth: ", index, riverline.length)
        for distance_along_line in np.arange(0, riverline.length, interval):
            cross_section = get_perpendicular_cross_section(riverline, distance_along_line, width)
            cross_sections.append(cross_section)

    gdf = gpd.GeoDataFrame(geometry=cross_sections)

    # Set the correct Coordinate Reference System (CRS) - replace with your appropriate EPSG code
    gdf.set_crs(epsg=28992, inplace=True)

    # Save to a Shapefile (.shp)
    # output_path = 'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/cross_sections_246.shp'
    # gdf.to_file(output_path, driver='ESRI Shapefile')
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
    # TO DOOOOOOOOOOOOOOO: p2 needs to be defined by the first line of buildings! Use linestring intersection to get this point p2.

    return LineString([p1, p2])


