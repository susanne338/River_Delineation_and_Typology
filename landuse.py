import geopandas as gpd
import fiona
import pandas as pd
import pygeos
import shapely
from shapely.wkt import loads
from shapely.geometry import box
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# file = "landuse.gml"
file = "landuse.gpkg"
file_profile = "profiles/left/2_left.csv"
cross_section_points = "points_line.shp"
cross_section_all = "cross_sections/cross_sections_1_longest.shp"


# This is very slow because it reads the entire file
# landuse_gdf = gpd.read_file(file)
# print(landuse_gdf.head())

# with fiona.open("landuse.gml") as gml_file:
#     # Print the schema (which includes column names)
#     print(gml_file.schema)

def landuse_cross_section(gpkg_file, cross_sections_shp):
    """

    :param gpkg_file: vector polygon file (for landuse)
    :param cross_section: Linestring in shapefile
    :return:
    """
    # Load the points data and get coordinates
    pointss_gdf = gpd.read_file(cross_sections_shp)
    print("poitns gdf columns ", pointss_gdf.columns)
    points_gdf = pointss_gdf.iloc[[7]]
    print("points gdf ", points_gdf)
    print("points gdf columns ", points_gdf.columns)
    points_gdf_geom = points_gdf.geometry.iloc[0]
    start_coords, end_coords = list(points_gdf_geom.coords)[0], list(points_gdf_geom.coords)[1]

    # Calculate the bounding box
    min_x = min(start_coords[0], end_coords[0])
    max_x = max(start_coords[0], end_coords[0])
    min_y = min(start_coords[1], end_coords[1])
    max_y = max(start_coords[1], end_coords[1])
    # Create the bounding box geometry
    bbox = (min_x, min_y, max_x, max_y)
    print("bbox ", bbox)

    # Load the land use data from a .gpkg file
    landuse_gdf = gpd.read_file(gpkg_file)
    # print("columns of landuse ",landuse_gdf.columns)
    # columns of landuse  Index(['gml_id', 'description', 'name', 'localId', 'namespace',
    #        'beginLifespanVersion', 'hilucsPresence', 'specificPresence',
    #        'observationDate', 'validFrom', 'validTo', 'geometry'],
    #       dtype='object')
    landuse_filtered = landuse_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    # Ensure the CRS matches
    if points_gdf.crs != landuse_gdf.crs:
        points_gdf = points_gdf.to_crs(landuse_gdf.crs)

    # Perform spatial join to get land use for each point
    joined_gdf = gpd.sjoin(points_gdf, landuse_filtered, how="left", op="intersects")

    # Extract land use column and add it to the points GeoDataFrame
    # CHECK WHAT COLUMN TO USE
    points_gdf['landuse'] = joined_gdf['description']

    # Print the result
    print(points_gdf[['h_distance', 'elevation', 'geometry', 'landuse']].head())


def landuse_profile(gpkg_landuse, profile_csv):
    # Get and put data in gpd
    df = pd.read_csv(profile_csv)
    # Convert the 'geometry' column from WKT strings to Shapely geometries
    df['geometry'] = df['geometry'].apply(loads)
    geo_df = gpd.GeoDataFrame(df, geometry='geometry')
    geo_df.set_crs(epsg=28992, inplace=True)

    landuse_gdf = gpd.read_file(gpkg_landuse)
    bbox = geo_df.total_bounds
    bbox_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs=geo_df.crs)
    bbox_gdf = bbox_gdf.to_crs(landuse_gdf.crs)
    # Extract the new bounding box in the CRS of landuse_gdf
    new_bbox = bbox_gdf.total_bounds

    # BBOX
    # before setting crs bbox  [ 35730.7452951  395829.60645649  35809.59678684 395837.11225204]
    print("bbox ", new_bbox)

    # print("columns of landuse ",landuse_gdf.columns)
    # columns of landuse  Index(['gml_id', 'description', 'name', 'localId', 'namespace',
    #        'beginLifespanVersion', 'hilucsPresence', 'specificPresence',
    #        'observationDate', 'validFrom', 'validTo', 'geometry'],
    #       dtype='object')
    landuse_filtered = landuse_gdf.cx[new_bbox[0]:new_bbox[2], new_bbox[1]:new_bbox[3]]

    print("should give point: ", geo_df.geom_type.unique())  # Should show 'Point'
    print("should give polygon or multipolygon: ", landuse_filtered.geom_type.unique())

    if geo_df.crs != landuse_gdf.crs:
        print("landuse crs ", landuse_gdf.crs)
        geo_df = geo_df.to_crs(landuse_gdf.crs)

    # Perform spatial join to get land use for each point
    joined_gdf = gpd.sjoin(geo_df, landuse_filtered, how="left", predicate="intersects")

    print(joined_gdf.head())
    print(joined_gdf.columns)

    # Extract land use column and add it to the points GeoDataFrame
    # CHECK WHAT COLUMN TO USE
    # geo_df['landuse'] = joined_gdf['description']
    if 'description' in joined_gdf.columns:
        geo_df['landuse'] = joined_gdf['description']
    else:
        print("The column 'description' is not present in joined_gdf.")

    # Print the result
    print(geo_df[['h_distance', 'elevation', 'geometry', 'landuse']].head())
    geo_df.to_csv("landuse_profile/landuseTEST.csv", index=False)
    return


# landuse_profile(file, file_profile)

csv_file = "landuse_profile/landuseTEST.csv"


def plot_landuse(csv_file):
    data = pd.read_csv(csv_file)
    landuse_colors = {
        'Main road': 'red',
        'Other agricultural usage': 'orange',
        'Other inland water': 'cyan',
    }
    data['color'] = data['landuse'].map(landuse_colors)
    plt.figure(figsize=(12, 8))
    plt.scatter(data['h_distance'], data['elevation'], c=data['color'], alpha=0.6, edgecolors='w', s=100)
    plt.axis('equal')
    # Add labels and title
    plt.title('Elevation vs. Horizontal Distance Colored by Land Use')
    plt.xlabel('Horizontal Distance (h_distance)')
    plt.ylabel('Elevation (elevation)')

    # Create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
               for color in landuse_colors.values()]
    labels = landuse_colors.keys()
    plt.legend(handles, labels, title="Land Use", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# plot_landuse(csv_file)

# ALL CROSS-SECIOTNS processing-----------------------------------------------------------------------------
cross_Section_longest = "cross_sections/cross_sections_longest.shp"


def landuse_profiles(gpkg_landuse, profile_csv_folder, total_cross_section_shp, output_folder):
    # I want to sjoin my data only once cause this is the slowest step. So I take the data within the bbox of my whole river by using the all-cross-sections file
    # Load data and set correct crs of bbox
    total_cs = gpd.read_file(total_cross_section_shp)
    landuse_gdf = gpd.read_file(gpkg_landuse)

    if total_cs.crs != landuse_gdf.crs:
        total_cs = total_cs.to_crs(landuse_gdf.crs)
        print("Transformed total_cs to match the CRS of landuse_gdf.")

    # print("columns of landuse ",landuse_gdf.columns)
    # columns of landuse  Index(['gml_id', 'description', 'name', 'localId', 'namespace',
    #        'beginLifespanVersion', 'hilucsPresence', 'specificPresence',
    #        'observationDate', 'validFrom', 'validTo', 'geometry'],
    #       dtype='object')
    bbox = total_cs.total_bounds
    print("bbox ", bbox)
    landuse_filtered = landuse_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    # Get and put data in gpd and then process
    for filename in os.listdir(profile_csv_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(profile_csv_folder, filename)
            print("file path ", file_path)
            df = pd.read_csv(file_path)

            # Convert the  'geometry' column from WKT strings to Shapely geometries
            df['geometry'] = df['geometry'].apply(loads)
            geo_df = gpd.GeoDataFrame(df, geometry='geometry')
            geo_df.set_crs(epsg=28992, inplace=True)

            if geo_df.crs != landuse_gdf.crs:
                geo_df = geo_df.to_crs(landuse_gdf.crs)

            joined_gdf = gpd.sjoin(geo_df, landuse_filtered, how="left", predicate="intersects")

            # Extract land use column and add it to the points GeoDataFrame. I use 'description' now but am not sure if it is the best column
            geo_df['landuse'] = joined_gdf['description']

            # Print the result
            print(geo_df[['h_distance', 'elevation', 'geometry', 'landuse']].head())
            # output_place = output_folder + '/' + filename
            output_place = os.path.join(output_folder, filename)
            geo_df.to_csv(output_place, index=False)
    return


# landuse_profiles(file, 'profiles/right', cross_Section_longest, 'landuse_profile/right')


# GET ALL LANDUSES IN RIVER CROSS-SECTIONS-----------------------------------------------------------------------------
def extract_landuses(unique_landuses, folder, output_csv_file):
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):  # Ensure we're only processing CSV files
            file_path = os.path.join(folder, filename)  # Construct full file path

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check if the 'landuse' column exists in the DataFrame
            if 'landuse' in df.columns:
                # Add unique land use values to the set
                unique_landuses.update(df['landuse'].unique())

    unique_landuses_list = list(unique_landuses)
    landuses_df = pd.DataFrame(unique_landuses_list, columns=['landuse'])

    # Save the DataFrame to a CSV file
    landuses_df.to_csv(output_csv_file, index=False)


# Initialize an empty set to store unique land uses
unique_landuses = set()
output_csv_file = 'landuse_profile/unique_landuses.csv'
extract_landuses(unique_landuses, 'landuse_profile/left', output_csv_file)
extract_landuses(unique_landuses, 'landuse_profile/right', output_csv_file)
unique_landuses_list = list(unique_landuses)
print(f"Unique land uses found: {unique_landuses_list}")

#
# # Convert the list to a DataFrame with one column named 'landuse'
# landuses_df = pd.DataFrame(unique_landuses_list, columns=['landuse'])
#
# # Save the DataFrame to a CSV file
# landuses_df.to_csv(output_csv_file, index=False)
