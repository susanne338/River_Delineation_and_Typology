"""

"""

import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, LineString, Polygon

# This gets the polygon of the river via the embankment points computed before
# TODO: add embankment points at buffer or add that into this function cause now it skips
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
# polygon_river('output/metrics/embankment_points_left.shp', 'output/metrics/embankment_points_right.shp', 'output/metrics/river_polygon.shp')

# METRIC VALUE COMPUTATION-------------------------------------------------------------------------
# Midpoints shp contains columns FID, left, right, max, height, width, geometry
midpoints_shp = "output/river/KanaalVanWalcheren/KanaalVanWalcheren_mid.shp"

def compute_ratio_flood(group, column):
    total_points = len(group)
    floodable = len(group[(group[column] != -9999) & (group[column].notna())])
    print(f"total points is {total_points} and floodable pts is  {floodable}")
    return floodable / total_points if total_points > 0 else 0.0

def compute_stats_flood(group, column):
    points_with_valid_depth = group[group[column] != -9999]
    depths = points_with_valid_depth[column]
    return(depths.mean(), depths.max(), depths.std())


def compute_landuse_ratios(section_points, unique_landuses):
    total_points = len(section_points)
    landuse_counts = {lu: 0 for lu in unique_landuses['landuse']}

    # Count occurrences of each landuse
    for landuses in section_points['landuse'].dropna():
        individual_landuses = [lu.strip() for lu in str(landuses).split(',')]
        for lu in individual_landuses:
            if lu in landuse_counts:
                landuse_counts[lu] += 1

    # Convert counts to ratios
    landuse_ratios = {f'landuse_{unique_landuses.loc[unique_landuses["landuse"] == lu, "index"].iloc[0]}':
                          count / total_points for lu, count in landuse_counts.items()}

    return landuse_ratios


def analyze_landuse_distribution(points_shp, unique_landuses_csv, metric_shp):
    parameters = gpd.read_file(points_shp)
    unique_landuses = pd.read_csv(unique_landuses_csv)

    # Group points by section ID
    grouped = parameters.groupby('id')

    # Initialize columns for metrics GeoDataFrame
    landuse_columns = [f'landuse_{idx}' for idx in range(len(unique_landuses))]
    columns = ['id', 'geometry'] + landuse_columns

    # Create metrics GeoDataFrame
    metrics = gpd.GeoDataFrame(columns=columns, crs=parameters.crs)

    # Process each section
    for idx, section in tqdm(grouped, desc="Processing sections"):
        # Get section ID and representative geometry
        id = section.iloc[0]['id']
        geometry = section.iloc[0]['geometry']

        # Compute landuse ratios
        landuse_ratios = compute_landuse_ratios(section, unique_landuses)

        # Create row data
        row_data = {'id': id, 'geometry': geometry}
        row_data.update(landuse_ratios)

        # Add row to metrics GeoDataFrame
        metrics.loc[idx] = row_data

    # Save to shapefile
    metrics.to_file(metric_shp, driver='ESRI Shapefile')
    return metrics

def compute_values(points_shp, metric_shp, midpoints_shp, unique_landuse_csv):
    parameters = gpd.read_file(points_shp)
    unique_landuses = gpd.read_file(unique_landuse_csv)
    # unique_landuses = extract_unique_landuses(points_shp, unique_landuse_csv)

    cs_shp = gpd.read_file(midpoints_shp)
    # columns: Index(['id', 'h_distance', 'elev_dtm', 'elev_dsm', 'flood_dept', 'landuse',
    #        'imperv', 'visible', 'geometry'],
    #       dtype='object')
    grouped = parameters.groupby('id')
    landuse_columns = [f'landuse_{idx}' for idx in range(len(unique_landuses))]
    base_columns = ['id', 'geometry', 'slope', 'barriers', 'rugosity', 'building',
                    'tree', 'landuse', 'land_div', 'imperv', 'flood_rat', 'flood_mean',
                    'flood_max', 'flood_std', 'buil_heig', 'rs_wid', 'visible', 'vis_layer']

    metrics = gpd.GeoDataFrame(
        columns=base_columns + landuse_columns,
        crs=parameters.crs
    )

    for idx, section in tqdm(grouped, desc="Processing sections"):
        print(f"idx is {idx}")

        id = section.iloc[0]['id']
        metrics.at[idx, 'id'] = id

        #TODO: Get riverwidth, embankmentpoints, etc from midpoints

        #flood metrics
        flood_ratio = compute_ratio_flood(section, 'flood_dept')
        flood_mean, flood_max, flood_std = compute_stats_flood(section, 'flood_dept')
        print(f"flood stats: {flood_max}, {flood_std}, {flood_mean}")
        print(f"flood ratio {flood_ratio}")
        #landuse metrics
        landuse_ratios = compute_landuse_ratios(section, unique_landuses)


        # This should update the metrics geodataframe
        # TODO: for some reason this doesnt work right now
        row_data = {
             'id': id,
            'geometry': None,
            'slope': None,
            'barriers': None,
            'rugosity': None,
            'building': None,
            'tree': None,
            'landuse': None,
            'land_div': None,
            'imperv': None,
            'flood_rat': flood_ratio,
            'flood_mean': flood_mean,
            'flood_max': flood_max,
            'flood_std': flood_std,
            'buil_heig': None,
            'rs_wid': None,
            'visible': None,
            'vis_layer': None
        }
        row_data.update(landuse_ratios)
        # metrics.loc[idx] = row_data
        metrics.loc[idx, row_data.keys()] = list(row_data.values())

    metrics.to_file(metric_shp, driver='ESRI Shapefile')

# compute_values(points_shp="output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_points_test.shp", metric_shp="metric_test.shp", midpoints_shp=midpoints_shp, unique_landuse_csv='output/parameters/unique_landuses.csv')



points ="output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_points_test.shp"
pts = gpd.read_file(points)
print(pts['flood_dept'].head(10))
metr = gpd.read_file("metric_test.shp")
print(metr[['flood_rat', 'flood_mean', 'flood_max', 'flood_std']].head(1))
metr.to_csv("metric_test.csv", index=False)