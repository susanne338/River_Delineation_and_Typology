"""
Following a tutorial
https://www.youtube.com/watch?v=pAWB4qVFZ9w&ab_channel=GeoDeltaLabs

input: shapefile of all cross sections, DTM tif file
output: csv files of elevation profiles of cross sections

TO DO:
Cut at river space edge, either here or when retrieving the cross-sections
"""
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pandas as pd
import os
import rasterio
from AHN_data_retrievalDELETE import fetch_AHN_data_bbox, extract_elevation

wcs_url_dtm = 'https://api.ellipsis-drive.com/v3/ogc/wcs/8b60a159-42ed-480c-ba86-7f181dcf4b8a?request=getCapabilities&version=1.0.0&requestedEpsg=28992'
wcs_url_dsm = 'https://api.ellipsis-drive.com/v3/ogc/wcs/78080fff-8bcb-4258-bb43-be9de956b3e0?request=getCapabilities&version=1.0.0&requestedEpsg=28992'
cross_sections = gpd.read_file(
    r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/cross_sections.shp')
n_points = 50


def profile_extraction(cross_sections, n_points, wcs_url, output_folder_left, output_folder_right, ahn_tif, ahn_tif_mod):
    """

    :param cross_sections: geodataframe of all cross sections along river
    :param n_points: Amount of points we want on the cross section
    :param wcs_url: WMS link
    :return: lists of geodataframes: combined_gdf, combined_gdf_left, combined_gdf_right
    """
    elevation_profiles = []
    gdf_list = []
    gdf_list_left = []
    gdf_list_right = []
    for ind, row in cross_sections.iterrows():
        print('I am working on ', ind)

        # XS_ID = str(row['FID'])
        # print("id? ", XS_ID)

        start_coords = list([row.geometry][0].coords)[0]
        end_coords = list([row.geometry][0].coords)[1]

        # Calculate the bounding box
        min_x = min(start_coords[0], end_coords[0])
        max_x = max(start_coords[0], end_coords[0])
        min_y = min(start_coords[1], end_coords[1])
        max_y = max(start_coords[1], end_coords[1])

        # Create the bounding box geometry
        bbox = (min_x, min_y, max_x, max_y)
        # print('bbox ', bbox)

        # Create the points on the cross-sections
        lon = [start_coords[0]]
        lat = [start_coords[1]]

        for i in np.arange(1, n_points + 1):
            x_dist = end_coords[0] - start_coords[0]
            y_dist = end_coords[1] - start_coords[1]
            point = [(start_coords[0] + (x_dist / (n_points + 1)) * i),
                     (start_coords[1] + (y_dist / (n_points + 1)) * i)]
            lon.append(point[0])
            lat.append(point[1])

        lon.append(end_coords[0])
        lat.append(end_coords[1])


        dem, bbox_dem = fetch_AHN_data_bbox(wcs_url, bbox, ahn_tif + str(ind) + '.tif', ahn_tif_mod + str(ind) + '.tif') #temp files cause this is in a loop and they keep being overwritten
        points = list(zip(lon, lat))
        # print('Points ', points)
        elevation_profile = extract_elevation(dem, points, bbox_dem)
        elevation_profiles.append(elevation_profile)
        # print('Elevation profile ', elevation_profile)

        # Create a geodataframe
        geometry = [Point(xy) for xy in points]
        df = pd.DataFrame({'geometry': geometry, 'elevation': elevation_profile})
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf.set_crs(epsg=28992, inplace=True)  # Example EPSG code

        gdf['h_distance'] = 0.0  # float instead of integer 0
        # Add distance
        for index, row in gdf.iterrows():
            gdf.loc[index, 'h_distance'] = gdf.geometry[0].distance(gdf.geometry[index])
        # print(gdf)

        # Split in half, REMOVES NODATA POINTS -246
        # maybe save each elevation profile seperatly
        midpoint = len(gdf) // 2
        left_section = gdf.iloc[midpoint:].copy()
        left_section = left_section[left_section['elevation'] != -246]
        left_section.reset_index(drop=True, inplace=True)
        gdf_list_left.append(left_section)
        right_section = gdf.iloc[:midpoint + 1].copy()
        right_section = right_section[right_section['elevation'] != -246]
        right_section.reset_index(drop=True, inplace=True)
        gdf_list_right.append(right_section)

        gdf_list.append(gdf)
        # gdf.to_file(
        #     r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/extracted_sections_BIGTEST_shp/')

        # Save left section as a shapefile
        left_section_shapefile = left_section[['h_distance', 'elevation', 'geometry']]
        left_section_shapefile.to_csv(
            output_folder_left + '/'+ str(ind) + '_left.csv', index=False)

        # Save right section as a shapefile
        right_section_shapefile = right_section[['h_distance', 'elevation', 'geometry']]
        right_section_shapefile.to_csv(
            output_folder_right +'/' + str(ind) + '_right.csv', index=False)
        print('i saved the csv files of ', ind)

    # After the loop, concatenate all the GeoDataFrames
    combined_gdf = pd.concat(gdf_list, ignore_index=True)
    combined_gdf_left = pd.concat(gdf_list_left, ignore_index=True)
    combined_gdf_right = pd.concat(gdf_list_right, ignore_index=True)

    # Save the combined GeoDataFrame as a shapefile
    # combined_gdf.to_file(r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/extracted_sections_BIGTESTTEST_shp/')
    return combined_gdf, combined_gdf_left, combined_gdf_right

cross_sections_longest = gpd.read_file('cross_sections/cross_sections_longest.shp')
n_points = 50

# wcs_url_dsm = 'https://api.ellipsis-drive.com/v3/ogc/wcs/8b60a159-42ed-480c-ba86-7f181dcf4b8a?request=getCapabilities&version=1.0.0&requestedEpsg=28992'
folder_left = 'profiles/left_dsm'
folder_right = 'profiles/right_dsm'
print('Going to extract profiles!...')
profile_extraction(cross_sections_longest, n_points, wcs_url_dsm, folder_left, folder_right, 'trash/tempDELETE', 'trash/tempDELETEmod')


# unfinsihes?
def profile_extraction_tiles(cross_sections, n_points, tiles_folder, shapefile_path):

    for ind, row in cross_sections.iterrows():
        print('I am working on ', ind)

        #  column 'geometry' holds POINT objects in shapefile
        start_coords = list([row.geometry][0].coords)[0]
        end_coords = list([row.geometry][0].coords)[1]
        # Create the points on the cross-sections
        lon = [start_coords[0]]
        lat = [start_coords[1]]

        for i in np.arange(1, n_points + 1):
            x_dist = end_coords[0] - start_coords[0]
            y_dist = end_coords[1] - start_coords[1]
            point = [(start_coords[0] + (x_dist / (n_points + 1)) * i),
                     (start_coords[1] + (y_dist / (n_points + 1)) * i)]
            lon.append(point[0])
            lat.append(point[1])

        lon.append(end_coords[0])
        lat.append(end_coords[1])
        points = list(zip(lon, lat))


        elevations = []
        for tif_file in os.listdir(tiles_folder):
            if tif_file.endswith(".tif"):
                tif_path = os.path.join(tiles_folder, tif_file)

                with rasterio.open(tif_path) as src:
                    # Get the bounds of the current .tif file
                    bounds = src.bounds

                    # Check for each point if it falls within the bounds of this .tif file
                    for point in points:
                        lon, lat = point
                        if bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top:
                            # If the point is within the bounds, get the elevation value
                            row, col = src.index(lon, lat)
                            elevation = src.read(1)[row, col]
                            elevations.append({'geometry': Point(lon, lat), 'elevation': elevation})

        # Create a geodataframe
        gdf = gpd.GeoDataFrame(elevations)
        gdf.set_crs(epsg=28992, inplace=True)

        # Add a new column for horizontal distance, initialized to 0.0
        gdf['h_distance'] = 0.0  # Initialize the h_distance column

        # Calculate distances
        for index, row in gdf.iterrows():
            # Calculate the distance from the first point
            if index == 0:
                continue  # Skip the first point as distance to itself is 0
            gdf.loc[index, 'h_distance'] = gdf.geometry[0].distance(gdf.geometry[index])

        gdf.to_file(shapefile_path, driver='ESRI Shapefile')
        print(f"Shapefile saved to: {shapefile_path}")

