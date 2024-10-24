"""
Computes the viewsheds and combines them.
Uses the computes viewsheds to compute the visible points ratio for river midpoints
and can check if a point in the profile is visible from midpoint
TODO: loop over combined_viewsheds for each point. Fix profiles. There are too many profiles, probably stemming from not using longest_river
"""
import rasterio
import numpy as np
from rasterio.features import geometry_mask
from shapely.geometry import mapping
import pandas as pd
from shapely.wkt import loads
import os
import fiona
import glob
from rasterio import features
from shapely.geometry import shape, Polygon, MultiPolygon, Point
from shapely.ops import unary_union
import geopandas as gpd
from shapely import wkt #Creates geometries from the Well-Known Text (WKT) representation.

# First run the visibility_batch_qgis.py script in the python console within QGIS. This gives all binary viewsheds for the river from the midpoints.

# COMBINE VIEWSHEDS INTO ONE TIF----------------------------------------------------------------------------
def combine_viewsheds(input_dir, output_path):
    """
        Combine multiple viewshed rasters using rasterio by taking the max value at each pixel.
        """
    # Get all binary viewshed files
    viewshed_files = glob.glob(os.path.join(input_dir, 'binary_viewshed_*.tif'))

    if not viewshed_files:
        print("No viewshed files found! :(")
        return

    print(f"Found {len(viewshed_files)} viewshed files")

    # Read first file to get metadata height width
    with rasterio.open(viewshed_files[0]) as src:
        profile = src.profile.copy()
        combined = np.zeros((src.height, src.width), dtype=np.uint8)

    # Process each viewshed
    for file in viewshed_files:
        print(f"Processing {os.path.basename(file)}! :)")
        with rasterio.open(file) as src:
            data = src.read(1)  # Read the first band
            combined = np.maximum(combined, data) #max between 0 and 1s

    # Update metadata profile for output
    profile.update(
        dtype=rasterio.uint8, #datatype in raster uint8 = integers from 0-255
        count=1, #number of bands in raster
        nodata=None  # Remove nodata value since we're using binary data
    #     Keep other profile information: width, height, crs, transform
    )

    # Write output to file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(combined, 1)

    print(f"Combined viewshed created at: {output_path}")

# Run function
input_directory = 'C:/Users/susan/Documents/thesis/Thesis-terminal/thesis_output/visibility'
output_file = os.path.join(input_directory, 'combined_viewshed.tif')

# combine_viewsheds(input_directory, output_file)

# DELINEATE VISIBLE AREA------------------------------------------------------------------------------------
def filter_small_polygons(polygons, min_area):
    filtered = []
    total_original = len(polygons)

    for poly in polygons:
        if poly.area >= min_area:
            filtered.append(poly)

    print(f"Filtered out {total_original - len(filtered)} small polygons")
    print(f"Remaining polygons: {len(filtered)}")
    return filtered

def remove_small_holes(polygon, min_area):

    filtered = []

    for poly in polygon:
        exterior = poly.exterior
        interiors = poly.interiors
        print('amount of holes ', len(interiors))
        retained_holes = []
        removed_holes_count = 0

        # Check each interior ring (hole)
        for interior in interiors:
            hole = Polygon(interior)
            hole_area = hole.area

            if hole_area > min_area:
                # Keep the hole if it's larger than the threshold
                retained_holes.append(interior)
            else:
                # Print message when removing a hole
                print(f"Removed hole with area: {hole_area:.4f}")
                removed_holes_count += 1

        if removed_holes_count == 0:
            print("No holes removed.")

        filtered.append(Polygon(exterior, retained_holes))
    return filtered

def raster_to_polygon(input_raster, output_shapefile, smooth_tolerance=1.0, min_area=100):
    """
    Conversion of my binary raster to a polygon shapefile with smoothed boundaries
    :param input_raster: path to input raster
    :param output_shapefile: path to output file
    :param smooth_tolerance: level of smoothing. Higher value means smoother but less precise
    :return: -
    """

    # Read the raster
    with rasterio.open(input_raster) as src:
        # Read the first band as a numpy array
        image = src.read(1)

        # Get the transform: A transform in geospatial data (also called a geotransform) is a set of parameters that defines how to convert between pixel coordinates (row/column in the image) and real-world coordinates (like latitude/longitude or meters in a projected coordinate system).
        transform = src.transform
        # Get CRS
        crs = src.crs

        print("Raster loaded successfully :)")

        # Create shapes from the raster where value is 1
        shapes = features.shapes(
            image,
            mask=image == 1,  # Only convert pixels with value 1
            transform=transform
        )

        print("Shapes extracted from raster")

        # Convert shapes to polygons
        polygons = []
        for geom, value in shapes:
            if value == 1:  # Only process visible areas
                poly = shape(geom)
                if poly.is_valid:
                    polygons.append(poly)

        print(f"Created {len(polygons)} initial polygons")
        # Filter out small polygons
        polygons = filter_small_polygons(polygons, min_area)
        polygons= remove_small_holes(polygons, min_area)
        # Combine all polygons into one
        combined_polygon = unary_union(polygons)
        print("Polygons combined")

        # Smooth the combined polygon
        smoothed_polygon = combined_polygon.simplify(smooth_tolerance, preserve_topology=True)
        print("Polygon smoothed")

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'geometry': [smoothed_polygon]},
            crs=crs
        )

        # Save to shapefile
        gdf.to_file(output_shapefile)
        print(f"Saved polygon to {output_shapefile}")
        if isinstance(smoothed_polygon, MultiPolygon):
            print(f"Final result contains {len(smoothed_polygon.geoms)} polygons")
        else:
            print("Final result is a single polygon")
        print(f"Total area: {smoothed_polygon.area:.2f} square units")

# Run the conversion to delineate visible space
input_raster = 'C:/Users/susan/Documents/thesis/Thesis-terminal/thesis_output/visibility/combined_viewshed.tif'
output_dir = 'C:/Users/susan/Documents/thesis/Thesis-terminal/thesis_output/visibility'
output_shapefile = os.path.join(output_dir, 'viewshed_boundary.shp')

# 1 gives very patchy
smooth_tolerance = 1.0
min_area = 100


# raster_to_polygon(input_raster, output_shapefile, smooth_tolerance, min_area)

# VISIBILITY PARAMETERS------------------------------------------------------------------------------------
# ONE VALUE FOR WHOLE CROSS-SECTION
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

# CHECK VISIBILITY PER POINT

def check_visibility(location, viewshed_file):
    """
    Checks the visibility of a point based on the viewshed raster.

    Parameters:
        location (Point): The geometry (Shapely Point object) representing the point's location.
        viewshed_file (str): The file path to the viewshed raster.

    Returns:
        int: Pixel value at the point's location (1 for visible, 0 for not visible, other for different values).
    """
    with rasterio.open(viewshed_file) as src:
        # Read the viewshed data from the first band
        viewshed_data = src.read(1)

        # Extract coordinates from the Point object
        x, y = location.x, location.y

        # Get the row, col index of the pixel corresponding to the point
        row, col = src.index(x, y)

        # Read the pixel value at the point's location
        pixel_value = viewshed_data[row, col]

        # Check the visibility based on the pixel value
        if pixel_value == 1:
            print(f"The point at ({x}, {y}) is visible.")
        elif pixel_value == 0:
            print(f"The point at ({x}, {y}) is not visible.")
        else:
            print(f"The pixel value at ({x}, {y}) is: {pixel_value}")

    return pixel_value


# Function to apply visibility check for each row in the GeoDataFrame
def compute_visibility(row, viewshed_file):
    """
    Compute visibility for a given row in the GeoDataFrame using the viewshed raster.

    Parameters:
        row: A row of the GeoDataFrame.
        viewshed_file (str): The file path to the viewshed raster.

    Returns:
        int: The visibility value (pixel value from the viewshed raster).
    """
    location = row['geometry']  # The Point object (geometry)

    # Call the visibility function using the location and the viewshed file
    return check_visibility(location, viewshed_file)


# Apply the compute_visibility function to each row in the GeoDataFrame
def add_visibility_column(gdf, viewshed_file):
    """
    Adds a 'visibility' column to the GeoDataFrame by checking the visibility of each point.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing point geometries.
        viewshed_file (str): The file path to the viewshed raster.

    Returns:
        GeoDataFrame: The updated GeoDataFrame with a new 'visibility' column.
    """
    gdf['visibility'] = gdf.apply(compute_visibility, axis=1, viewshed_file=viewshed_file)
    return gdf


def load_csv_to_geodataframe(csv_file):
    """
    Loads a CSV file into a GeoDataFrame, converting coordinate columns into Point geometries.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        GeoDataFrame: A GeoDataFrame with Point geometries.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file)
    # geometry is a string but needs to be POINT objects
    df['geometry'] = df['geometry'].apply(wkt.loads)

    # Convert the DataFrame into a GeoDataFrame, specifying that the 'geometry' column contains Point geometries
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    return gdf

landuse_profile_csv = 'thesis_output/profiles/left/0_left.csv'
gdf_csv = load_csv_to_geodataframe(landuse_profile_csv)
viewshed_file = 'thesis_output/visibility/viewshed_0.tif'
gdf_with_vis = add_visibility_column(gdf_csv, viewshed_file)
gdf_with_vis.to_csv('thesis_output/trash/Testcsv.csv', index=False)

gdf_with_vis.to_file('thesis_output/trash/test_shp.shp', driver='ESRI Shapefile')