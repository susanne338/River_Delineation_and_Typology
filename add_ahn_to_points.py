"""
I have my points divided into pieces. This file has to be executed in the QGIS console. The layer
AHN DTM or DSM has to be loaded in QGIS as WCS and the name of the layer has to be defined in the code.
It samples the values from the raster for each point.

"""

from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
import os
import glob



def extract_raster_values():
    print("Starting...")
    # Get the WCS layer - adjust the layer name to match yours
    raster_layer_name = 'dtm'
    raster_layer = QgsProject.instance().mapLayersByName(raster_layer_name)[0]

    # Get all gpkg files in the split folder
    print("Getting all gpkg files...")
    gpkg_files = glob.glob('C:/Users/susan/Documents/thesis/Thesis-terminal/output/PTS/split/*.gpkg')


    print(f"Got files. Starting the loop for {gpkg_files}")
    for gpkg_path in gpkg_files:
        # Load the vector layer
        layer_name = os.path.splitext(os.path.basename(gpkg_path))[0]
        print(f"working on {layer_name}")
        vector_layer = QgsVectorLayer(gpkg_path, layer_name, 'ogr')

        if not vector_layer.isValid():
            print(f"Layer {gpkg_path} failed to load!")
            continue

        # Start editing session
        vector_layer.startEditing()

        # Add a new field for raster values if it doesn't exist
        if 'dtm' not in [field.name() for field in vector_layer.fields()]:
            vector_layer.dataProvider().addAttributes([QgsField('dtm', QVariant.Double)])
            vector_layer.updateFields()

        raster_val_idx = vector_layer.fields().indexOf('dtm')

        # Create a data provider for raster sampling
        raster_data = raster_layer.dataProvider()

        # Process each feature
        for feature in vector_layer.getFeatures():
            point = feature.geometry().asPoint()

            # Get raster value at point
            value = raster_data.sample(point, 1)[0]

            # Update the feature
            vector_layer.changeAttributeValue(feature.id(), raster_val_idx, value)

        # Commit changes and save
        vector_layer.commitChanges()
        print(f"Processed {layer_name}")


# Run the function
extract_raster_values()