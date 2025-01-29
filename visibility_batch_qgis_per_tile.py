import numpy as np
import processing
import os
from glob import glob
import geopandas as gpd
import sys
from qgis.core import QgsProject, QgsVectorLayer, QgsRectangle, QgsRasterLayer


# def get_wcs_layer():
#     # Get the WCS layer that's already loaded in QGIS
#     # You'll need to use the exact name of your WCS layer as it appears in QGIS
#     wcs_layer_name = "dsm"  # Change this to match your layer name
#     wcs_layer = QgsProject.instance().mapLayersByName(wcs_layer_name)
#
#     if not wcs_layer:
#         raise Exception(f"WCS layer '{wcs_layer_name}' not found in the QGIS project")
#
#     return wcs_layer[0]


tile_file = "C:/Users/susan/Documents/thesis/Thesis-terminal/output/CS/mid/tiles_processing_visibility.shp"
tiles_gdf = gpd.read_file(tile_file)

for idx, tile in tiles_gdf.iterrows():
    if tile['done'] == 1:
        print(f"Tile {tile['name']} has already been processed")
        continue

    tile_name = tile['name']
    temp_dir = f'C:/Users/susan/Documents/thesis/Thesis-terminal/temp/{tile_name}/'
    os.makedirs(temp_dir, exist_ok=True)

    # Load the point shapefile
    points_layer = QgsVectorLayer(
        f'C:/Users/susan/Documents/thesis/Thesis-terminal/output/CS/mid/split/{tile_name}.gpkg',
        'points',
        'ogr')

    if not points_layer.isValid():
        print("Layer failed to load!")
    else:
        print(f"Loaded {points_layer.featureCount()} features.")
        print(f"Points layer CRS: {points_layer.crs().authid()}")


    dsm_path = f"C:/Users/susan/Documents/thesis/Thesis-terminal/input/DATA/AHN/DSM/{tile_name}.tif"
    # Get the WCS layer
    # try:
    #     wcs_layer = get_wcs_layer()
    # except Exception as e:
    #     print(f"Error getting WCS layer: {str(e)}")
    #     continue

    # Load and process river layer as before
    # river_layer = QgsVectorLayer(
    #     'C:/Users/susan/Documents/thesis/Thesis-terminal/input/RIVERS/OSM_rivers_clipped_filtered_nowatervlaktes_noNonWaterdeel_dissolved_split_uniqueID.shp',
    #     'river', 'ogr')
    #
    # buffered_river_path = "C:/Users/susan/Documents/thesis/Thesis-terminal/input/DATA/BGT_rivers/extend_100mbuffer_around_waterdelen_for_bgt_download_NOHOLES.shp"
    #
    # # Instead of clipping a local DSM, we'll clip the WCS layer
    # clipped_dem_path = f"C:/Users/susan/Documents/thesis/Thesis-terminal/input/DATA/AHN/temp_{tile_name}_clip.tif"
    #
    # # Clip WCS layer using the buffered river
    # clip_params = {
    #     'INPUT': wcs_layer,
    #     'MASK': buffered_river_path,
    #     'OUTPUT': clipped_dem_path,
    #     'NODATA': None,
    #     'ALPHA_BAND': False,
    #     'CROP_TO_CUTLINE': True,
    #     'KEEP_RESOLUTION': True
    # }
    # processing.run("gdal:cliprasterbymasklayer", clip_params)
    # print(f"Clipped WCS layer saved to {clipped_dem_path}")
    dem_layer = QgsRasterLayer(dsm_path, "dsm")
    if not dem_layer.isValid():
        print(f"Failed to load DEM from {dsm_path}, probably cause it is not downloaded")
        continue

    # Set up the output directory
    output_dir = f'C:/Users/susan/Documents/thesis/Thesis-terminal/output/VIS/{tile_name}/viewsheds/'
    os.makedirs(output_dir, exist_ok=True)

    extent = points_layer.extent()
    buffer_size = 200  # 200m buffer
    extent_with_buffer = QgsRectangle(
        extent.xMinimum() - buffer_size,
        extent.yMinimum() - buffer_size,
        extent.xMaximum() + buffer_size,
        extent.yMaximum() + buffer_size
    )

    print(f"\nClipping extent:")
    print(f"xmin: {extent_with_buffer.xMinimum()}")
    print(f"ymin: {extent_with_buffer.yMinimum()}")
    print(f"xmax: {extent_with_buffer.xMaximum()}")
    print(f"ymax: {extent_with_buffer.yMaximum()}")

    clipped_dem_path = os.path.join(temp_dir, f'dem_{tile_name}.tif')

    # Create temporary extent polygon
    extent_layer = QgsVectorLayer("Polygon?crs=" + points_layer.crs().authid(), "extent", "memory")
    provider = extent_layer.dataProvider()

    feat = QgsFeature()
    geom = QgsGeometry.fromRect(extent_with_buffer)
    feat.setGeometry(geom)
    provider.addFeatures([feat])

    # Clip using mask layer
    clip_params = {
        'INPUT': dem_layer,
        'MASK': extent_layer,
        'SOURCE_CRS': dem_layer.crs().authid(),
        'TARGET_CRS': points_layer.crs().authid(),
        'NODATA': None,
        'ALPHA_BAND': False,
        'CROP_TO_CUTLINE': True,
        'KEEP_RESOLUTION': True,
        'OUTPUT': clipped_dem_path
    }

    try:
        print("\nAttempting to clip raster...")
        result = processing.run("gdal:cliprasterbymasklayer", clip_params)
        print(f"Clipping operation completed. Checking output...")

        if os.path.exists(clipped_dem_path):
            file_size = os.path.getsize(clipped_dem_path)
            print(f"Output file size: {file_size} bytes")
            if file_size == 0:
                print("Warning: Output file exists but is empty!")
                continue
            else:
                print("Output file created successfully")
        else:
            print("Error: Output file was not created!")
            continue

    except Exception as e:
        print(f"Error clipping raster: {str(e)}")
        continue



    memory_mb = 10240
    num_cores = 6
    print("Starting the viewshed loop")

    # Process viewsheds for each point
    for i, feature in enumerate(points_layer.getFeatures()):
        river_width = feature['vwidth']
        try:
            river_width = float(river_width) if river_width is not None else 200
        except (TypeError, ValueError):
            river_width = 200

        max_distance = 100 + 0.5 * river_width
        point_geom = feature.geometry().asPoint()

        height = feature['vheight'] + 1.75
        if not height:
            height = 1.75

        point_id = feature.id()
        print(f"river width is {river_width}, height: {height}, point_is {point_id}")

        params = {
            'input': clipped_dem_path,
            # 'input': wcs_layer,
            'coordinates': f'{point_geom.x()},{point_geom.y()}',
            'max_distance': max_distance,
            'observer_elevation': height,
            'target_elevation': 0,
            'output': f'{output_dir}viewshed_{point_id}.tif',
            '-b': True,
            'memory': memory_mb,
            'threads': num_cores,
            'refraction_coeff': 0.14286,
            'GRASS_REGION_CELLSIZE_PARAMETER': 1,
            'GRASS_REGION_READ_MAP_OPTS': '-s',
            'overwrite': True
        }

        processing.run('grass7:r.viewshed', params)
        if os.path.exists(params['output']):
            print(f"Viewshed output saved to: {params['output']}")
        else:
            print(f"Viewshed output failed for point ID: {point_id}")
            # sys.exit("Exiting the program because the viewshed was not made.")
    # Clean up temporary clipped DEM
    if os.path.exists(clipped_dem_path):
        os.remove(clipped_dem_path)

    # Update tile status
    tiles_gdf.at[idx, 'done'] = 1
    tiles_gdf.to_file(tile_file, driver='GeoJSON')

print("Batch process completed!")