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
import rasterio
import matplotlib.pyplot as plt
from AHN_data_retrieval import fetch_AHN_data_bbox, extract_elevation

wcs_url = 'https://api.ellipsis-drive.com/v3/ogc/wcs/8b60a159-42ed-480c-ba86-7f181dcf4b8a?request=getCapabilities&version=1.0.0&requestedEpsg=28992'
cross_sections = gpd.read_file(
    r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/cross_sections.shp')
n_points = 50


def profile_extraction(cross_sections, n_points, wcs_url):
    """

    :param cross_sections: geodataframe of all cross sections along river
    :param n_points: Amount of points we want on the cross section
    :param wcs_url: WMS link
    :return:
    """
    elevation_profiles = []
    gdf_list = []
    gdf_list_left = []
    gdf_list_right = []
    for ind, row in cross_sections.iterrows():

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

        # create a geodataframe
        # df = pd.DataFrame({'Latitude': lat,
        #                    'Longitude': lon})
        #
        # gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
        # gdf.set_crs(epsg=28992, inplace=True)  # Updated CRS initialization method
        #
        # gdf_pcs = gdf.to_crs(epsg=28992) #unesearry?

        # Adding h_distance column (?)
        # initiate at 0
        # gdf_pcs['h_distance'] = 0.0  # float instead of integer 0
        # # Add distance
        # for index, row in gdf_pcs.iterrows():
        #     gdf_pcs.loc[index, 'h_distance'] = gdf_pcs.geometry[0].distance(gdf_pcs.geometry[index])
        #
        # # Extracting the elevations from the DEM
        # gdf_pcs['Elevation'] = 0.0

        dem, bbox_dem = fetch_AHN_data_bbox(wcs_url, bbox, 'temp_ahn.tif', 'temp_ahn_mod.tif',) #temp files cause this is in a loop and they keep being overwritten
        # print('dem ', dem)
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

        # # Save left section as a shapefile
        # left_section_shapefile = left_section[['h_distance', 'elevation', 'geometry']]
        # # left_section_shapefile.to_file(
        # #     r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/extracted_sections_shp/' + XS_ID + '_left.shp')
        #
        # # Save right section as a shapefile
        # right_section_shapefile = right_section[['h_distance', 'elevation', 'geometry']]
        # # right_section_shapefile.to_file(
        # #     r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/extracted_sections_shp/' + XS_ID + '_right.shp')

    # After the loop, concatenate all the GeoDataFrames
    combined_gdf = pd.concat(gdf_list, ignore_index=True)
    combined_gdf_left = pd.concat(gdf_list_left, ignore_index=True)
    combined_gdf_right = pd.concat(gdf_list_right, ignore_index=True)

    # Save the combined GeoDataFrame as a shapefile
    # combined_gdf.to_file(r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/extracted_sections_BIGTESTTEST_shp/')
    return combined_gdf, combined_gdf_left, combined_gdf_right

    # dem = rasterio.open(
    #     r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/water_middy/DTM_AHN_nodata_246_Middy.tif',
    #     mode='r')

    # for index, row in gdf_pcs.iterrows():
    #     row, col = dem.index(row['Longitude'], row['Latitude'])
    #     dem_data = dem.read(1)
    #
    #     # gdf_pcs['Elevation'].loc[index] = dem_data[row, col]
    #     gdf_pcs.loc[index, 'Elevation'] = dem_data[row, col]
    #
    # # Split in half, REMOVES NODATA POINTS -246
    # midpoint = len(gdf_pcs) // 2
    # left_section = gdf_pcs.iloc[midpoint:].copy()
    # left_section = left_section[left_section['Elevation'] != -246]
    # left_section.reset_index(drop=True,
    #                          inplace=True)  # Reset index for saving purposes. Ensures the indices start from 0 for both the left and right sections
    #
    # right_section = gdf_pcs.iloc[:midpoint + 1].copy()
    # right_section = right_section[right_section['Elevation'] != -246]
    # right_section.reset_index(drop=True, inplace=True)
    #
    # # Save left section
    # left_x_y_data = left_section[['h_distance', 'Elevation', 'Latitude', 'Longitude']]
    # left_x_y_data.to_csv(
    #     r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/extracted_sections/' + XS_ID + '_left.csv',
    #     index=False)
    #
    # # Save right section
    # right_x_y_data = right_section[['h_distance', 'Elevation', 'Latitude', 'Longitude']]
    # right_x_y_data.to_csv(
    #     r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/extracted_sections/' + XS_ID + '_right.csv',
    #     index=False)
    #
    # # Save left section as a shapefile
    # left_section_shapefile = left_section[['h_distance', 'Elevation', 'Latitude', 'Longitude', 'geometry']]
    # left_section_shapefile.to_file(
    #     r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/extracted_sections_shp/' + XS_ID + '_left.shp')
    #
    # # Save right section as a shapefile
    # right_section_shapefile = right_section[['h_distance', 'Elevation', 'Latitude', 'Longitude', 'geometry']]
    # right_section_shapefile.to_file(
    #     r'C:/Users/susan/OneDrive/Documenten/geomatics/Thesis_3D_Delineation/Python_test/TEST_cross_sections/extracted_sections_shp/' + XS_ID + '_right.shp')


profile_extraction(cross_sections, n_points, wcs_url)
