"""
Delineate flood
"""
from owslib.wms import WebMapService
import requests
from PIL import Image
from io import BytesIO
import geopandas as gpd
import numpy as np
import os
import rasterio
# from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.transform import Affine
from rasterio.transform import rowcol
from shapely.geometry import Point, Polygon, MultiPolygon
import alphashape
import matplotlib.pyplot as plt
from pyproj import CRS

# tif file containing the 100 year flood information
tif_file = "input/flood/KanaalVanWalcheren_clip.tif"


# RETRIEVE DATA VIA WMS-------------------------------------------------------------------------------------------------
def retrieve_flood_tif(wms_url, river_shapefile, buffer, flood_output_full_tif, flood_output_clip_tif):
    """
    There is a problem here with the coordinates. Theyre not there. Idk how to fix it.
    Args:
        wms_url:
        river_shapefile:
        buffer:
        flood_output_full_tif:
        flood_output_clip_tif:

    Returns:

    """

    wms = WebMapService(wms_url, version='1.3.0')
    # Check what layers are available
    # for layer in wms.contents:
    #     print(f"ID: {layer}, Name: {wms[layer].name}, Title: {wms[layer].title}")

    # check available crs
    # layer = wms['maximale_overstromingsdiepte_middelgrote_kans']
    # print("Available CRS for the layer:")
    # for crs in layer.crsOptions:
    #     print(crs)

    # Get river line
    gdf = gpd.read_file(river_shapefile)
    line = gdf.geometry.iloc[0]
    buffered_polygon = line.buffer(buffer)
    buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_polygon], crs=gdf.crs)
    minx, miny, maxx, maxy = buffered_gdf.to_crs(epsg=28992).total_bounds
    bbox = (minx, miny, maxx, maxy)
    print(f"bbox is {bbox}")

    # compute resolution
    bbox_width = maxx - minx
    bbox_height = maxy - miny
    resolution = 25  # meters per pixel
    width_pixels = int(bbox_width / resolution)
    height_pixels = int(bbox_height / resolution)
    size = (width_pixels, height_pixels)
    # size = (2048, 2048) #this is prob max

    response = wms.getmap(
        layers=['maximale_overstromingsdiepte_middelgrote_kans'],
        srs='EPSG:28992',
        bbox=bbox,
        size=size,
        format='image/tiff',
        transparent=True,
        dpi=96
    )

    # Save the full GeoTIFF file with correct metadata
    with rasterio.open(
            flood_output_full_tif, 'w',
            driver='GTiff',
            height=height_pixels,
            width=width_pixels,
            count=1,
            dtype=rasterio.float32,
            crs=CRS.from_epsg(28992),
            transform=Affine(resolution, 0, minx, 0, -resolution, maxy)
    ) as dst:
        dst.write(np.frombuffer(response.read(), dtype=rasterio.float32).reshape(height_pixels, width_pixels), 1)

    # with open(flood_output_full_tif, 'wb') as f:
    #     f.write(response.read())

    # I  TRY TO CLIP BUT IT DOESN'T WORK WELL
    # Open the unclipped tif file
    # with rasterio.open(flood_output_full_tif) as src:
    #     # Reproject buffered polygon to match raster CRS
    #     # buffered_gdf = buffered_gdf.to_crs(src.crs)
    #
    #     # Mask the raster with the buffered polygon
    #     out_image, out_transform = mask(src, buffered_gdf.geometry, crop=True)
    #     out_image[out_image == src.nodata] = np.nan  # Set no-data values to NaN
    #
    #     # Save the masked raster as a new tif
    # with rasterio.open(
    #         flood_output_clip_tif, 'w',
    #         driver='GTiff',
    #         height=out_image.shape[1],
    #         width=out_image.shape[2],
    #         count=1,
    #         dtype=out_image.dtype,
    #         crs=src.crs,
    #         transform=out_transform
    # ) as dest:
    #     dest.write(out_image, 1)
    #
    #
    #     # Remove unclipped tif
    # if os.path.exists(flood_output_clip_tif):
    #     os.remove(flood_output_full_tif)
    #     print(f"{flood_output_full_tif} has been deleted.")
    # else:
    #     print(f"Clipped TIFF not found. {flood_output_full_tif} was not deleted.")
    return


def clip_raster(flood_tif, river_shapefile, flood_tif_clip):
    gdf = gpd.read_file(river_shapefile)
    line = gdf.geometry.iloc[0]
    polygon = line.buffer(buffer)

    with rasterio.open(flood_tif) as src:
        flood_data = src.read(1)  # Read the first band
        affine = src.transform
        transform = src.transform
        crs = CRS.from_epsg(28992)

        # Convert your polygon to GeoJSON
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=crs)

        # Mask the raster
        out_image, out_transform = mask(src, gdf.geometry, crop=True)
        out_image = np.where(np.isnan(out_image), -9999, out_image)
        # out_image[out_image == src.nodata] = np.nan  # Replace nodata with NaN for easier analysis
        # Define metadata for the new file
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],  # Height of the array
            "width": out_image.shape[2],  # Width of the array
            "count": 1,
            "transform": transform,  # The transform from the mask function
            "dtype": "float32",  # Data type (e.g., np.float32, np.int16)
            "nodata": -9999
        })

        # Write the clipped data to a new TIFF file
        with rasterio.open(flood_tif_clip, "w", **out_meta) as dest:
            dest.write(out_image, 1)  # Write to the first band


# Get
wms_url = "https://apps.geodan.nl/public/data/org/gws/YWFMLMWERURF/kea_public/wms?request=getCapabilities"
shapefile_path = 'input/river/longest_river.shp'
flood_output_tif = "input/flood/KanaalVanWalcheren_full.tif"
flood_output_tif_clip = "input/flood/KanaalVanWalcheren_clip.tif"
buffer = 150


# retrieve_flood_tif(wms_url, shapefile_path, buffer, flood_output_tif, flood_output_tif_clip)
# clip_raster(flood_output_tif, shapefile_path, flood_output_tif_clip)

# Extract data per point------------------------------------------------------------------------------------------------

def floodable_points(flood_tif, output_file):
    with rasterio.open(flood_tif) as src:
        flood_data = src.read(1)  # Read the first band
        affine = src.transform  # Get affine transform to map pixel to coordinates
        height, width = flood_data.shape

        values = []
        for row in range(height):
            for col in range(width):
                elevation = flood_data[row, col]

                # Only process pixels that are not nodata (-9999)
                if elevation != -9999 and elevation != 0:

                    # Get the geographic coordinates of the pixel center
                    # x, y = affine * (col + 0.5, row + 0.5)

                    # Get coordinates of each pixels corners
                    top_left = affine * (col, row)
                    top_right = affine * (col + 1, row)
                    bottom_left = affine * (col, row + 1)
                    bottom_right = affine * (col + 1, row + 1)

                    # # Create a Point geometry with the x, y coordinates and the elevation
                    # point = Point(x, y)
                    # # Add the point with its elevation to the list
                    # values.append({'geometry': point, 'elevation': elevation})
                    values.append({'geometry': Point(top_left), 'elevation': elevation})
                    values.append({'geometry': Point(top_right), 'elevation': elevation})
                    values.append({'geometry': Point(bottom_left), 'elevation': elevation})
                    values.append({'geometry': Point(bottom_right), 'elevation': elevation})

    gdf = gpd.GeoDataFrame(values, crs=src.crs)
    gdf = gdf.set_crs("EPSG:28992", allow_override=True)

    # Save the GeoDataFrame as a shapefile
    gdf.to_file(output_file)
    return


floodable_points(tif_file, 'output/flood/floodable_points_corners.shp')

def alpha_shape(points_file, output_file, alpha):
    """
    Computes the alpha shape around a set of points from a shapefile, and saves to shapefile. This is used to delineate the floodable area.
    Note epsg has to be degree based for alpha shape computation
    Args:
        points_file: shapefile containing the points to make an alpha shape around
        output_file: shapefile to save the alpha shape to
        alpha: value for alpha parameter

    Returns:
    TODO: add river polygon to shape.
    TODO: extend are and then cut at buffer line
    """
    gdf = gpd.read_file(points_file)
    gdf = gdf.to_crs(epsg=4326)
    # it needs to be in 4326 as this uses degrees which works better
    points = gdf.geometry
    coordinates = [(point.x, point.y) for point in points]
    print(f"coordinates: {coordinates}")

    # alpha = 1200
    alpha_shape = alphashape.alphashape(coordinates, alpha)
    print(f"Alpha Shape Type: {type(alpha_shape)}")

    if isinstance(alpha_shape, (Polygon, MultiPolygon)):
        # If it's a valid geometry, plot the result
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        gdf.plot(ax=ax, color='blue', markersize=5)  # Plot the points
        if isinstance(alpha_shape, Polygon):
            x, y = alpha_shape.exterior.xy  # Get the exterior of the alpha shape
            ax.fill(x, y, alpha=0.3, color='red')  # Fill the alpha shape with color
            print(f"alpha shape is a polygon")
        elif isinstance(alpha_shape, MultiPolygon):
            print(f"alpha shape is a multipolygon")
            # Handle MultiPolygon by plotting each individual polygon
            for poly in alpha_shape:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.3, color='red')
        plt.show()
        print(f"Alpha Shape Type: {alpha_shape} ")
        alpha_shape_gdf = gpd.GeoDataFrame(geometry=[alpha_shape], crs=gdf.crs)
        print(f"Alpha Shape Type: {alpha_shape_gdf} ")
        print(alpha_shape_gdf.crs)
        alpha_shape_gdf = alpha_shape_gdf.to_crs(epsg=28992)
        print(alpha_shape_gdf.crs)
        alpha_shape_gdf.to_file(output_file)


    # Check if the alpha shape is a valid Polygon
    # if isinstance(alpha_shape, Polygon):
    #     # Plot the result
    #     fig, ax = plt.subplots()
    #     ax.set_aspect('equal')
    #     gdf.plot(ax=ax, color='blue', markersize=5)  # Plot the points
    #     x, y = alpha_shape.exterior.xy  # Get the exterior of the alpha shape
    #     ax.fill(x, y, alpha=0.3, color='red')  # Fill the alpha shape with color
    #     plt.show()
    #
    #     # Save the alpha shape as a new shapefile (optional)
    #     alpha_shape_gdf = gpd.GeoDataFrame(geometry=[alpha_shape], crs=gdf.crs)
    #     alpha_shape_gdf.to_file(output_file)

    else:
        print("The alpha shape is not a valid Polygon or multipolgon.")
        print(f"Geometry Details: {alpha_shape}")
        print(f"Is the geometry valid? {alpha_shape.is_valid}")


alpha_shape("output/flood/floodable_points_corners.shp", "output/flood/alpha_shape_corners.shp", 1200)
