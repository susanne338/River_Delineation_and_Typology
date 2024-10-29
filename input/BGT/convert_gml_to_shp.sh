#!/bin/bash

# Directory containing the GML files
INPUT_FOLDER="bgt_kanaalvanWalcheren"  # Replace with the path to your folder containing .gml files
OUTPUT_FOLDER="bgt_kanaalvanWalchere_conv"  # Replace with your desired output folder

# Create the output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Loop over each .gml file in the input folder
for gml_file in "$INPUT_FOLDER"/*.gml; do
    # Check if the .gml file exists (in case there are no .gml files in the folder)
    if [[ -f "$gml_file" ]]; then
        # Get the base name of the file (without extension)
        base_name=$(basename "$gml_file" .gml)

        # Create a subfolder for the output Shapefile
        shp_output_folder="$OUTPUT_FOLDER/$base_name"
        mkdir -p "$shp_output_folder"

        # Set the path for the output .shp file
        shp_file="$shp_output_folder/$base_name.shp"

        # Convert .gml to .shp
        ogr2ogr -f "ESRI Shapefile" "$shp_file" "$gml_file"

        # Check if conversion was successful
        if [[ $? -eq 0 ]]; then
            echo "Converted $gml_file to $shp_file"
        else
            echo "Failed to convert $gml_file"
        fi
    else
        echo "No .gml files found in $INPUT_FOLDER"
        exit 1
    fi
done
