""""
Run this script in the python console in qgis. Can't run from here because 'processing' uses an old python? One of the wheels is in python2x and can't be installed.
Change input paths as needed.
Change output path as needed.
"""

import processing

# Load the point shapefile into QGIS (make sure the path is correct)
points_layer = QgsVectorLayer('C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/pythonProject/river_shapefiles/river_midpoints_elev_2m.shp', 'points', 'ogr')
if not points_layer.isValid():
    print("Layer failed to load!")
else:
    print(f"Loaded {points_layer.featureCount()} features.")

# Path to the DEM raster for viewshed analysis
dem_path = 'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/pythonProject/AHN_tiles_DSM/clipped_total.tif'


# Loop through each point in the shapefile
for i, feature in enumerate(points_layer.getFeatures()):
    if   i <= 5:
        point_geom = feature.geometry().asPoint()  # Get the point's geometry as a coordinate (x, y)
        print('point geometry and feature id: ', point_geom, feature.id())
        point_id = feature.id()  # Get the ID of the point feature

        # Set up parameters for r.viewshed
        params = {
            'input': dem_path,  # The DEM raster
            'coordinates': f'{point_geom.x()},{point_geom.y()}',  # Observer coordinates (point)
            'max_distance': 100,  # Maximum visibility distance
            'observer_elevation': 2.00,  # Observer height (adjust as needed)
            'target_elevation': 0,  # Target height (adjust if needed)
            'output': f'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/pythonProject/visibility/viewshed_{point_id}.tif',  # Output file path for each point
            '-b': True
        }

        # Run the r.viewshed tool
        processing.run('grass7:r.viewshed', params)

        # Binary
        print(params.keys())
        viewshed_output_path = params['output']
        binary_output_path = f'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/pythonProject/visibility/binary_viewshed_{point_id}.tif'

        # Reclassify to create binary output
        reclass_params = {
            'INPUT_A': viewshed_output_path,  # Input raster A
            'BAND_A': 1,  # Band number for input A
            'FORMULA': 'A >= 1',  # Reclassification formula
            'OUTPUT': binary_output_path,
            'NO_DATA': None,
            'RTYPE': 1,  # Output raster type (Byte/Int8)
            'OPTIONS': ''
        }

        processing.run("gdal:rastercalculator", reclass_params)

print("Batch process completed!")

