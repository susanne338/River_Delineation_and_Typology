#!/bin/bash
# THIS SCRIPT CONVERTS MY .gml FILES TO .gpkg FOR SPEEDING UP PROCESSING. THE .gml FILES ARE MANUALLY RETRIEVED
# The .gfs files are ignored


# Specify the directory where your .gml files are stored
input_folder="bgt_zeeland"
output_folder="bgt_zeeland_conv"  # Optional, use the same as input if you want

# Create output folder if it doesn't exist
mkdir -p "$output_folder"

# Loop over each .gml file in the input folder
for gml_file in "$input_folder"/*.gml; do
    # Get the base name of the file (without path and extension)
    base_name=$(basename "$gml_file" .gml)

    # Define output .gpkg file path
    output_file="$output_folder/$base_name.gpkg"

    # Convert .gml to .gpkg using ogr2ogr
    ogr2ogr -f "GPKG" "$output_file" "$gml_file"

    echo "Converted $gml_file to $output_file"
done
