import os
import numpy as np
import rasterio
import glob
from rasterio import features
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
print('hello world')

# COMBBINE VIEWSHEDS INTO ONE TIF
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
input_directory = 'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/pythonProject/visibility/'
output_file = os.path.join(input_directory, 'combined_viewshed.tif')

# combine_viewsheds(input_directory, output_file)

# DELINEATE-------------------------------------------------
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
input_raster = 'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/pythonProject/visibility/combined_viewshed.tif'
output_dir = 'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/pythonProject/visibility/'
output_shapefile = os.path.join(output_dir, 'viewshed_boundary.shp')

# 1 gives very patchy
smooth_tolerance = 1.0
min_area = 100


raster_to_polygon(input_raster, output_shapefile, smooth_tolerance, min_area)