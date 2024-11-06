import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, LineString, Polygon

# ADD EMBANKMENT AND VIEWPOINT HEIGHT VALUES TO CROSS-SECTIONS-MIDPOINTS FILE------------------------------------------
def extract_max_elevation_values(shapefile, midpoints_file, embankments_file_left, embankments_file_right, user_defined_height):
    """
    Extracts the elevation values of the embankments, and selects the max value for the viewshed analysis
    This max value is then added up with user_defined_height.
    Args:
        shapefile: parameter shapefile
        midpoints_file: cross-section midpoint file that is altered
        user_defined_height: viewpoint height to peform viewshed analysis from

    Returns: saves values left, right (embankment), max, and (viewpoint) height to midpoints_file
    TODO: add location. what does this mean.
    TODO: now when the river extend my buffer, no boundary point gets added. Fix to edge of buffer, or last point
    """
    parameters = gpd.read_file(shapefile)
    midpoints = gpd.read_file(midpoints_file)

    # Initialize new columns in midpoints for output
    midpoints['left'] = np.nan
    midpoints['right'] = np.nan
    midpoints['max'] = np.nan
    midpoints['height'] = np.nan

    # Initialize embankments points
    embankment_points_left = gpd.GeoDataFrame(columns=[])
    embankment_points_left['geometry'] = np.nan
    embankment_points_left.set_crs("EPSG:28992", inplace=True)
    embankment_points_right = gpd.GeoDataFrame(columns=[])
    embankment_points_right['geometry'] = np.nan
    embankment_points_right.set_crs("EPSG:28992", inplace=True)

    # Group by cross-section ID
    grouped = parameters.groupby('id')
    for idx, row in tqdm(midpoints.iterrows(), total=midpoints.shape[0], desc="Processing midpoints"):
        midpoint_geom = row.geometry
        cross_section_id = row['FID']

        # Select corresponding cross-section points from 'parameters' shapefile
        cross_section_points = grouped.get_group(cross_section_id)

        # Ensure points are sorted by h_distance
        cross_section_points = cross_section_points.sort_values('h_distance').reset_index(drop=True)

        # Calculate approximate midpoint in h_distance
        max_h_distance = cross_section_points['h_distance'].max()
        target_midpoint = max_h_distance / 2

        # Find the closest point to this target midpoint
        midpoint_idx = (cross_section_points['h_distance'] - target_midpoint).abs().idxmin()

        # Initialize left and right values
        left_value = None
        right_value = None

        # Search left of the midpoint for the first valid elevation value
        for i in range(midpoint_idx - 1, -1, -1):
            if not pd.isna(cross_section_points.loc[i, 'elev_dtm']):
                left_value = cross_section_points.loc[i, 'elev_dtm']
                left_geom = cross_section_points.loc[i, 'geometry']
                print(f"left geom is {left_geom} and type {type(left_geom)}")
                break

        # Search right of the midpoint for the first valid elevation value
        for i in range(midpoint_idx + 1, len(cross_section_points)):
            if not pd.isna(cross_section_points.loc[i, 'elev_dtm']):
                right_value = cross_section_points.loc[i, 'elev_dtm']
                right_geom = cross_section_points.loc[i, 'geometry']
                break

        # Assign left, right, max, and height values to the midpoints DataFrame
        midpoints.at[idx, 'left'] = left_value
        midpoints.at[idx, 'right'] = right_value
        embankment_points_left.at[idx, 'geometry'] = left_geom
        embankment_points_right.at[idx, 'geometry'] = right_geom
        # embankment_points.at[idx, 'right_geom'] = right_geom

        # Calculate max and height and add to file
        max_value = max(
            filter(None, [left_value, right_value])) if left_value is not None or right_value is not None else None
        midpoints.at[idx, 'max'] = max_value
        midpoints.at[idx, 'height'] = max_value + user_defined_height if max_value is not None else None

        # Save the result to a new shapefile
    midpoints.to_file(midpoints_file, driver='ESRI Shapefile')
    embankment_points_left.to_file(embankments_file_left, driver="ESRI Shapefile")
    print(f"Results saved to {midpoints_file} and {embankments_file_left}")
    embankment_points_right.to_file(embankments_file_right, driver="ESRI Shapefile")
    print(f"Results saved to {midpoints_file} and {embankments_file_right}")


# extract_max_elevation_values('output/parameters/parameters_longest.shp',
#                              'output/cross_sections/cross_sections_midpoints.shp', 'output/metrics/embankment_points_left.shp', 'output/metrics/embankment_points_right.shp', 1.75)

# # SAVE THE FILE TO CSV TO TAKE A LOOK
# gdf = gpd.read_file('output/metrics/embankment_points_left.shp')
# print("Columns in the shapefile:", gdf.columns.tolist())
# gdf.to_csv("output/metrics/cross_section_midpoints_embankment.csv", index=False)
# gdf = gdf.drop(columns=['left_geom', 'right_geom'])
# print("Columns in the shapefile:", gdf.columns.tolist())
# gdf = gpd.read_file('output/metrics/embankment_points_right.shp')
# gdf = gdf.drop(columns=['left_geom', 'right_geom'])
# print("Columns in the shapefile:", gdf.columns.tolist())
# gdf = gpd.read_file('output/parameters/parameters_longest.shp')
# print("Columns in the shapefile:", gdf.columns.tolist())

def polygon_river(embankment_left, embankment_right, river_polygon_shapefile):
    gdf_right = gpd.read_file(embankment_right)
    gdf_left = gpd.read_file(embankment_left)
    points_right = [Point(row['geometry'].x, row['geometry'].y) for index, row in gdf_right.iterrows()]
    points_left = [Point(row['geometry'].x, row['geometry'].y) for index, row in gdf_left.iterrows()]
    # points_right = sorted(points_right, key=lambda p: (p.x, p.y))
    # points_left = sorted(points_left, key=lambda p: (p.x, p.y))

    line_right = LineString(points_right)
    line_left = LineString(points_left)
    if line_right.coords[-1] != line_left.coords[0]:
        line_right = LineString(list(reversed(line_right.coords)))

    coords = list(line_right.coords) + list(line_left.coords)
    print(f"coordinates {coords}")
    polygon = Polygon(coords)
    gdf_polygon = gpd.GeoDataFrame(geometry=[polygon])
    gdf_polygon.set_crs(gdf_right.crs, allow_override=True, inplace=True)
    gdf_polygon.to_file(river_polygon_shapefile)
    return

polygon_river('output/metrics/embankment_points_left.shp', 'output/metrics/embankment_points_right.shp', 'output/metrics/river_polygon.shp')