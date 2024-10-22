"""
Uses the computes viewsheds to compute the visible points ratio for river midpoints
and can check if a point in the profile is visible from midpoint
TODO: loop for all viewsheds, first compute all viewsheds
"""
import rasterio
import numpy as np
from rasterio.features import geometry_mask
from shapely.geometry import Point
from shapely.geometry import mapping
import geopandas as gpd
import pandas as pd
from shapely.wkt import loads


def ratio_visibility(viewshed_file, radius, midpoints_file_used_for_visibility, point):
    """
    Computes the ratio of visibility within the radius of the viewshed for one point from which the viewshed is computes, so for now its the centerpoint of the cross-section (river midpoints)
    :param viewshed_file: viewshed computes in batch process qgis
    :param radius: set to same as viewshed = 100m
    :param midpoints_file_used_for_visibility: midpoints of river
    :param point: index of point to compute
    :return: visible percentage float
    """
    gdf = gpd.read_file(midpoints_file_used_for_visibility)
    pt = gdf.iloc[point]
    center_point = (pt.geometry.x, pt.geometry.y)
    print('center point ', center_point)

    # Open the viewshed raster
    with rasterio.open(viewshed_file) as src:
        # Read the viewshed data
        viewshed_data = src.read(1)  # Assuming single band
        transform = src.transform  # Get the transform for georeferencing

        # Create a circular mask for the specified radius
        center_pixel = src.index(center_point[0], center_point[1])
        y_indices, x_indices = np.ogrid[:viewshed_data.shape[0], :viewshed_data.shape[1]]
        distance = np.sqrt((x_indices - center_pixel[1]) ** 2 + (y_indices - center_pixel[0]) ** 2)

        # Create a mask where distance is less than the radius
        mask = distance <= radius

        # Apply the mask to the viewshed data
        limited_viewshed = np.where(mask, viewshed_data, np.nan)

        # Calculate the percentage of visible area within the radius
        visible_count = np.count_nonzero(limited_viewshed == 1)  # Assuming 1 is visible
        total_count = np.count_nonzero(~np.isnan(limited_viewshed))  # Count of valid cells
        percentage_visible = (visible_count / total_count) * 100 if total_count > 0 else 0

        # Print results
        print(f"Percentage visible within {radius}m: {percentage_visible:.2f}%")
        return percentage_visible

# ratio_visibility('visibility/viewshed_0.tif', 100, 'river_shapefiles/river_midpoints_elev_2m.shp', 0)


def visible_point(viewshed_file, point):
    """
    Checks if point is visible
    :param viewshed_file:
    :param point: (x,y)
    :return:
    """
    with rasterio.open(viewshed_file) as src:
        # Read the viewshed data
        viewshed_file= src.read(1)
        row, col = src.index(point[0], point[1])

        # Read the value at the pixel location
        pixel_value = src.read(1)[row, col]  # Read the first band

        # Check if the point is visible
        if pixel_value == 1:
            print("The point is visible.")
        elif pixel_value == 0:
            print("The point is not visible.")
        else:
            print(f"The pixel value at the point is: {pixel_value}")
    return pixel_value


def extract_point_coordinates_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    coordinates = []
    for geom in df['geometry']:
        # Parse the WKT POINT into a Shapely Point object
        point = loads(geom)  # This assumes the geometry is in WKT format
        coordinates.append((point.x, point.y))
    return coordinates



landuse_profile = 'profiles/left/0_left.csv'
pt = extract_point_coordinates_from_csv(landuse_profile)[0]
print('point ', pt)
visible_point('visibility/viewshed_0.tif', pt)