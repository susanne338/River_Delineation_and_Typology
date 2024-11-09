"""
Computes the viewsheds and combines them.
Uses the computes viewsheds to compute the visible points ratio for river midpoints
and can check if a point in the profile is visible from midpoint
TODO: make ratio_visibility add to metric table
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
from tqdm import tqdm
from rasterio.windows import Window
from collections import Counter


#Creates geometries from the Well-Known Text (WKT) representation.

# VIEWSHED BATCH
# First run the visibility_batch_qgis.py script in the python console within QGIS. This gives all binary viewsheds for the river from the midpoints.

# COMBINE VIEWSHEDS INTO ONE TIF----------------------------------------------------------------------------
def get_reference_dimensions(viewshed_files):
    """
    Get the most common dimensions from all viewshed files
    """
    dimensions = []
    for file in viewshed_files:
        with rasterio.open(file) as src:
            dimensions.append((src.height, src.width))

    # Get most common dimensions
    common_dims = Counter(dimensions).most_common(1)[0][0]
    print(f"Reference dimensions (height, width): {common_dims}")

    # Also return a list of files that don't match these dimensions
    files_to_crop = [(f, dim) for f, dim in zip(viewshed_files, dimensions) if dim != common_dims]
    if files_to_crop:
        print(f"\nFound {len(files_to_crop)} files that need cropping:")
        for f, dim in files_to_crop:
            print(f"- {os.path.basename(f)}: {dim}")

    return common_dims, files_to_crop


def crop_to_reference(src, ref_height, ref_width, ref_bounds):
    """
    Crop a larger raster to match reference dimensions, aligning with reference bounds
    """
    # Get the coordinates of the reference center
    ref_center_x = (ref_bounds.left + ref_bounds.right) / 2
    ref_center_y = (ref_bounds.top + ref_bounds.bottom) / 2

    # Convert reference center to pixel coordinates in the larger raster
    center_px_x, center_px_y = src.index(ref_center_x, ref_center_y)

    # Calculate the start positions to align with reference center
    start_y = int(center_px_y - ref_height // 2)
    start_x = int(center_px_x - ref_width // 2)

    # Ensure we don't go out of bounds
    start_y = max(0, min(start_y, src.height - ref_height))
    start_x = max(0, min(start_x, src.width - ref_width))

    # Read the cropped portion
    window = Window(start_x, start_y, ref_width, ref_height)
    data = src.read(1, window=window)

    return data


def combine_viewsheds(input_dir, output_path):
    """
        Combine multiple viewshed rasters using rasterio by taking the max value at each pixel.
        """
    # Get all binary viewshed files
    viewshed_files = glob.glob(os.path.join(input_dir, 'viewshed_*.tif'))

    if not viewshed_files:
        print("No viewshed files found! :(")
        return

    print(f"Found {len(viewshed_files)} viewshed files")

    (ref_height, ref_width), files_to_crop = get_reference_dimensions(viewshed_files)
    files_to_crop = dict(files_to_crop)

    # Get reference bounds from a file with correct dimensions
    ref_file = next(f for f in viewshed_files if f not in files_to_crop)

    with rasterio.open(ref_file) as src:
        ref_bounds = src.bounds
        ref_profile = src.profile.copy()

    # Initialize result array
    result = None

    # Initialize counters for statistics
    processed_files = 0
    cropped_files = 0
    error_files = 0

    for file in tqdm(viewshed_files, desc="Merging viewsheds", unit="file"):
        try:
            with rasterio.open(file) as src:
                # Check if file needs cropping
                if file in files_to_crop:
                    print(f"\nCropping {os.path.basename(file)}")
                    print(f"Original shape: {files_to_crop[file]}")
                    data = crop_to_reference(src, ref_height, ref_width, ref_bounds)
                    print(f"Cropped to: {data.shape}")
                    cropped_files += 1
                else:
                    data = src.read(1)

                # Verify dimensions
                if data.shape != (ref_height, ref_width):
                    print(f"\nError: Shape mismatch after processing {os.path.basename(file)}")
                    print(f"Expected shape: ({ref_height}, {ref_width}), got: {data.shape}")
                    error_files += 1
                    continue

                # Convert to binary
                binary_data = (data > 0).astype(np.uint8)

                # Initialize or update result
                if result is None:
                    result = binary_data
                else:
                    result = np.logical_or(result, binary_data).astype(np.uint8)

                processed_files += 1

        except Exception as e:
            print(f"\nError processing {file}: {str(e)}")
            error_files += 1
            continue

    if result is None:
        print("No valid data to merge!")
        return

    # Update metadata profile for output
    ref_profile.update(
        dtype=rasterio.uint8, #datatype in raster uint8 = integers from 0-255
        count=1, #number of bands in raster
        nodata=None  # Remove nodata value since we're using binary data
    #     Keep other profile information: width, height, crs, transform
    )

    print("\nWriting merged viewshed to file...")
    with rasterio.open(output_path, 'w', **ref_profile) as dst:
        dst.write(result, 1)

    # Print final statistics
    print(f"\nMerged viewshed created at: {output_path}")
    print(f"Successfully processed: {processed_files}")
    print(f"Files cropped: {cropped_files}")
    print(f"Errors: {error_files}")

    # Calculate visibility statistics
    visible_pixels = np.sum(result == 1)
    total_pixels = result.size
    visibility_percentage = (visible_pixels / total_pixels) * 100
    print(f"\nVisibility Statistics:")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Visible pixels: {visible_pixels:,}")
    print(f"Visibility percentage: {visibility_percentage:.2f}%")


# Run function
input_directory = 'output/visibility/KanaalVanWalcheren/viewsheds'
directory_vis = 'output/visibility/KanaalVanWalcheren'
output_file = os.path.join(directory_vis, 'combined_viewshed.tif')

combine_viewsheds(input_directory, output_file)

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
input_raster = 'output/visibility/combined_viewshed.tif'
output_shapefile = os.path.join(directory_vis, 'viewshed_boundary.shp')

# 1 gives kinda patchy result?
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

river = 'output/river/KanaalVanWalcheren/KanaalVanWalcheren_mid.shp'
gdf = gpd.read_file(river)

# Get the union of the geometries in the shapefile and create a buffer
river_geometry = gdf.geometry.unary_union
buffered_area = river_geometry.buffer(200)

# Create a GeoDataFrame from the buffered area, ensuring it has the correct CRS
# gdfbuf = gpd.GeoDataFrame(geometry=[buffered_area], crs="EPSG:28992")

# Write the buffered area to a new shapefile
# gdfbuf.to_file('output/river/KanaalVanWalcheren/buffer.shp', driver='ESRI Shapefile')