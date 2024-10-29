def process_landuse(points_gdf, gpkg_folder):
    """
    Add landuse information to points GeoDataFrame from gpkg files from BGT data.

    Args:
        points: geodataframe of points
        gpkg_file: Folder containing .gpkg files

    Returns:
        GeoDataFrame: Updated points with landuse information
    """

    # Initialize landuse column
    points_gdf = points_gdf.to_crs(epsg=28992)
    points_gdf['landuse'] = None

    bbox = points_gdf.total_bounds
    print('I have found the bbox: ', bbox)

    # Loop through gpkg files
    i = 0
    for gpkg_file in os.listdir(gpkg_folder):
        print('index ', i)
        if not gpkg_file.endswith('.gml'):
            continue
        gpkg_path = os.path.join(gpkg_folder, gpkg_file)
        for layer_name in fiona.listlayers(gpkg_path):
            print(f'Processing layer: {layer_name} in file: {gpkg_file}')

            landuse_gdf = gpd.read_file(gpkg_path, layer=layer_name)
            bounds_landuse = landuse_gdf.total_bounds
            print('I have read the landuse file and its bounds are: ', bounds_landuse)

            landuse_filtered = landuse_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            print('I filtered the landuse to bbox')

            # Ensure CRS matches
            if landuse_filtered.crs != points_gdf.crs:
                print('crs landuse ', landuse_filtered.crs)
                print('crs points ', points_gdf.crs)
                landuse_filtered.set_crs("EPSG:28992", inplace=True)
                landuse_filtered = landuse_filtered.to_crs(points_gdf.crs)
                # points_gdf = points_gdf.to_crs(landuse_filtered.crs)
                print('crs landuse ', landuse_filtered.crs)
            print('points bbox now is ', points_gdf.total_bounds)
            print('landuse filtered columns are ', landuse_filtered.columns)
            print("Filtered landuse count:", len(landuse_filtered))

            # Process each point
            # I get a warning here that says I should work with a copy but I don't want to I want to write to this specific one no?
            joined_gdf = gpd.sjoin(points_gdf, landuse_filtered, how="left", predicate="intersects")
            # It's this line: that points_gdf is being altered instead of a copy of it
            points_gdf.loc[joined_gdf.index, 'landuse'] = layer_name

            # Process each point for CBS data
            # joined_gdf = gpd.sjoin(points_gdf, landuse_filtered, how="left", predicate="intersects")
            # points_gdf['landuse'] = joined_gdf['description']


            i += 1

    return points_gdf

# THIS IS FOR THE FOLDER OF GPKG BUT IT TAKES FOREVER
# def process_landuse(points_gdf, gpkg_folder):
#     """
#     Add landuse information to points GeoDataFrame from GeoPackage files.
#
#     Args:
#         points_gdf: GeoDataFrame containing points
#         gpkg_folder: Folder containing .gpkg files
#
#     Returns:
#         GeoDataFrame: Updated points with landuse information
#     """
#     # Initialize landuse column
#     points_gdf['landuse'] = None
#
#     # Process each point
#     for idx in tqdm(points_gdf.index, desc="Processing landuse"):
#         point = points_gdf.loc[[idx]]
#         found_intersection = False
#
#         # Loop through each GeoPackage file
#         for gpkg_file in os.listdir(gpkg_folder):
#             if not gpkg_file.endswith('.gpkg'):
#                 continue
#
#             gpkg_path = os.path.join(gpkg_folder, gpkg_file)
#
#             # Loop over each layer in the GeoPackage
#             layers = gpd.io.file.fiona.listlayers(gpkg_path)
#             for layer_name in layers:
#                 landuse_gdf = gpd.read_file(gpkg_path, layer=layer_name)
#
#                 # Check if the 'plus-type' column exists in this layer
#                 if 'plus-type' not in landuse_gdf.columns:
#                     continue
#
#                 # Ensure CRS matches
#                 if landuse_gdf.crs != points_gdf.crs:
#                     landuse_gdf = landuse_gdf.to_crs(points_gdf.crs)
#
#                 # Perform spatial join with the point
#                 joined = gpd.sjoin(point, landuse_gdf, how="left", predicate="within")
#
#                 # Check if we found an intersection
#                 if not joined['plus-type'].isna().all():
#                     points_gdf.loc[idx, 'landuse'] = joined['plus-type'].iloc[0]
#                     found_intersection = True
#                     break  # Exit layer loop since we found an intersection
#
#             if found_intersection:
#                 break  # Exit GeoPackage loop if we found an intersection
#
#         if not found_intersection:
#             print(f"Warning: No landuse intersection found for point {idx}")
#
#     return points_gdf

def add_landuse_to_shapefile(shapefile_path, gpkg_folder):
    """
    Reads a shapefile, adds landuse information from gpkg files, and saves the updated shapefile.

    Parameters:
        shapefile_path: Path to the shapefile containing all cross-section points
        gpkg_folder: Folder containing landuse .gpkg files
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    print('The shapefile has been read! its columns are: ', gdf.columns)

    # Add landuse information
    gdf = process_landuse(gdf, gpkg_folder)
    print("landuse is added to the geodataframe!")

    # Save updated shapefile
    gdf.to_file(shapefile_path)
    print(f"Updated shapefile saved to: {shapefile_path}")
