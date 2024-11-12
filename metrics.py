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

def compute_ratio_flood(group, column):
    total_points = len(group)
    floodable = len(group[(group[column] != -9999) & (group[column].notna())])
    # print(f"total points is {total_points} and floodable pts is  {floodable}")
    return floodable / total_points if total_points > 0 else 0.0

def compute_stats_flood(group, column):
    points_with_valid_depth = group[group[column] != -9999]
    depths = points_with_valid_depth[column]
    return(depths.mean(), depths.max(), depths.std())


def compute_landuse_ratios(section_points, unique_landuses):
    """
    Compute ratios for each landuse per section. Nr of type landuse divided by total nr of landuse
    Args:
        section_points: Points in cross-section
        unique_landuses: The total different landuses extracted from all points

    Returns: ratios of landuse per landuse per section, total up to 1

    """
    if len(section_points) == 0:
        return {f'landuse_{idx}': None for idx in range(len(unique_landuses))}

    total_points = len(section_points)
    landuse_counts = {lu: 0 for lu in unique_landuses['landuse']}
    total_count = 0

    for landuses in section_points['landuse'].dropna():
        individual_landuses = [lu.strip() for lu in str(landuses).split(',')]
        for lu in individual_landuses:
            if lu in landuse_counts:
                landuse_counts[lu] += 1
                total_count += 1

    # Convert counts to ratios: This is for dividing by number of points
    # landuse_ratios = {f'landuse_{unique_landuses.loc[unique_landuses["landuse"] == lu, "index"].iloc[0]}':
    #                       count / total_points for lu, count in landuse_counts.items()}
    # This is for dividing by number of landuses
    landuse_ratios = {f'landuse_{unique_landuses.loc[unique_landuses["landuse"] == lu, "index"].iloc[0]}':
                          count / total_count for lu, count in landuse_counts.items()}
    non_zero_landuse_types = sum(1 for ratio in landuse_ratios.values() if ratio > 0)

    return landuse_ratios, non_zero_landuse_types


def compute_landuse_ratios_nonequal_contr(section_points, unique_landuses):
    """
        Compute ratios for each landuse per section. Uses a non-equal contribution of landuse if a point has multiple landuses
        Args:
            section_points: Points in cross-section
            unique_landuses: The total different landuses extracted from all points

        Returns: ratios of landuse per landuse per section, total up to 1

        """
    landuse_counts = {lu: 0 for lu in unique_landuses['landuse']}
    total_count = 0

    # .dropna() drops entries in column that have NaN or None values
    for landuses in section_points['landuse'].dropna():
        individual_landuses = [lu.strip() for lu in str(landuses).split(',')]

        total_count += 1
        contribution = 1.0 / len(individual_landuses)

        for lu in individual_landuses:
            if lu in landuse_counts:
                landuse_counts[lu] += contribution

    landuse_ratios = {}
    for lu in unique_landuses['landuse']:
        idx = unique_landuses.loc[unique_landuses['landuse'] == lu, 'index'].iloc[0]
        ratio = landuse_counts[lu] / total_count if total_count > 0 else 0
        landuse_ratios[f'landuse_{idx}'] = ratio

    # Verification that ratios sum to approximately 1
    ratio_sum = sum(landuse_ratios.values())
    # print(f"Sum of ratios for section: {ratio_sum}")
    if not (0.99 <= ratio_sum <= 1.01):  # Allow for small floating point errors
        print("Warning: Landuse ratios don't sum to 1")

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

def compute_imperv_ratio(group):
    imperv_count = 0
    perv_count = 0
    valid_point_count = group['imperv'].dropna().count()
    for imperv_value in group['imperv'].dropna():
        # print(f"imperv value is {imperv_value}")
        if imperv_value <= 50:
            perv_count += 1
        else:
            imperv_count += 1
    ratio = imperv_count / valid_point_count
    return ratio




def building_metrics(group, closest_intersection, intersections, midpoint_row, embankment):
    first_intersection_point = None
    first_building_height = None
    river_space_width = None

    side_id = group.iloc[0]['id']
    side_side = group.iloc[0]['side']
    length_side = group['split_h_di'].max()

    # Find first intersection
    for _, row in closest_intersection.iterrows():
        if row['id'] == side_id and row['side'] == side_side:

            first_intersection_point = row['geometry']
            first_building_height = row['height']
            break

    if first_intersection_point:
        embank_row = embankment[(embankment['id'] == side_id) & (embankment['side'] == side_side)]
        embank_pt = embank_row['geometry'].iloc[0]
        river_space_width = first_intersection_point.distance(embank_pt)

    intersections_lengths = []

    # Find lengths of building intersections
    for _, row in intersections.iterrows():
        if row['id'] == side_id and row['side'] == side_side:
            intersections_lengths.append(row.geometry.length)

    total_length_intersections = sum(intersections_lengths)
    if length_side and total_length_intersections:
        ratio = total_length_intersections / length_side
    else:
        ratio = None

    return ratio, first_building_height, first_intersection_point, river_space_width


def visibility_ratio(group):
    count_1, count_0 = 0, 0
    valid_point_count = group['visible'].dropna().count()
    for value in group['visible'].dropna():
        if int(value) == 0:
            count_0 +=  1
            continue
        if int(value) == 1:
            count_1 += 1
            continue
        else: print(f"something is wrong")
    return count_1 / valid_point_count

def visibility_layerdness(group):
    visible_segments = 0
    in_visible_segment = False  # Track whether we're in a visible segment

    for index, row in group.iterrows():
        if row['visible'] == 1:
            if not in_visible_segment:  # We are starting a new visible segment
                visible_segments += 1
                in_visible_segment = True  # We're now in a visible segment
        else:
            in_visible_segment = False  # We're no longer in a visible segment
    return visible_segments



# RUN-------------------------------------------------------------------------------------------------------------------
def compute_values(points_shp, metric_shp, midpoints_shp, unique_landuse_csv, halves_shp, building_intersections_shp, closest_building_intersections_shp, embankment_shp):
    parameters = gpd.read_file(points_shp)
    # Index(['id', 'h_distance', 'visible', 'imperv', 'elev_dtm', 'elev_dsm',
    #        'landuse', 'flood_dept', 'side', 'split_h_di', 'geometry'],
    #       dtype='object')
    halves = gpd.read_file(halves_shp)
    # columns: Index(['id', 'side', 'height1', 'build1', 'geometry'], dtype='object')
    building_intersections = gpd.read_file(building_intersections_shp)
    #  columns: Index(['id', 'side', 'geometry'], dtype='object')
    closest_building_intersections = gpd.read_file(closest_building_intersections_shp)
    # Index(['id', 'side', 'height', 'geometry'], dtype='object')
    unique_landuses = pd.read_csv(unique_landuse_csv)
    cs_shp = gpd.read_file(midpoints_shp)
    # Index(['FID', 'left', 'right', 'max', 'height', 'width', 'geometry'], dtype='object')
    embankment = gpd.read_file(embankment_shp)
    # Index(['id', 'side', 'height', 'h_dist', 'geometry'], dtype='object')

    # check if 'side' exists
    if 'side' not in parameters.columns:
        print("Error: 'side' column is missing from the input data")
        return
    if parameters['side'].isnull().any():
        print("Warning: There are null values in the 'side' column")

    grouped = parameters.groupby(['id', 'side'])
    rows = []

    for (idx, side), group in tqdm(grouped, desc="Processing sections"):
        # print(f"Processing id: {idx}, side: {side}")
        # print(f"type of side: {type(side)}")  # Check that section is a DataFrame or GeoDataFrame

        matching_cs = cs_shp[cs_shp['FID'] == idx].iloc[0] if len(cs_shp[cs_shp['FID'] == idx]) > 0 else None

        # Validity check because some cross-sections are invalid
        is_valid = matching_cs is not None and matching_cs['width'] != 0.0
        geometry = matching_cs.geometry if matching_cs is not None else None

        if is_valid:
            # Compute metrics only for valid cross-sections
            flood_ratio = compute_ratio_flood(group, 'flood_dept')
            flood_mean, flood_max, flood_std = compute_stats_flood(group, 'flood_dept')
            landuse_ratios, landuse_diversity = compute_landuse_ratios(group, unique_landuses)
            imperv_ratio = compute_imperv_ratio(group)
            building_ratio, first_building_height, first_intersection_point, river_space_width = building_metrics(group, closest_building_intersections, building_intersections, matching_cs, embankment)
            vis_rat = visibility_ratio(group)
            vis_layerdness = visibility_layerdness(group)
        else:
            # Set all metrics to None for invalid cross-sections
            flood_ratio = None
            flood_mean, flood_max, flood_std = None, None, None
            landuse_ratios = {f'landuse_{idx}': None for idx in range(len(unique_landuses))}
            imperv_ratio = None
            building_ratio, first_building_height, first_intersection_point, river_space_width = None, None, None, None
            vis_rat, vis_layerdness = None, None

            # This should update the metrics geodataframe
        row_data = {
            'id': idx,
            'side': side,
            'geometry': geometry,
            'slope': None,
            'barriers': None,
            'rugosity': None,
            'building': building_ratio,
            'tree': None,
            'land_div': landuse_diversity,
            'imperv': imperv_ratio,
            'flood_rat': flood_ratio,
            'flood_mean': flood_mean,
            'flood_max': flood_max,
            'flood_std': flood_std,
            'buil_heig': first_building_height,
            'rs_wid': river_space_width,
            'visible': vis_rat,
            'vis_layer': vis_layerdness
        }
        row_data.update(landuse_ratios)
        rows.append(row_data)

    metrics = gpd.GeoDataFrame(rows, crs=parameters.crs)
    metrics.to_file(metric_shp, driver='ESRI Shapefile')
    return metrics




midpoints_shp = "output/river/KanaalVanWalcheren/KanaalVanWalcheren_mid.shp"
points_shp ="output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_points_test.shp"
halves_shp = 'output/river/KanaalVanWalcheren/KanaalVanWalcheren_mid_halves.shp'
intersections = "output/buildings/KanaalVanWalcheren/building_intersections/building_intersections.shp"
closest_intersections = "output/buildings/KanaalVanWalcheren/closest_building_intersections/closest_building_intersections.shp"
embankments_file = "output/embankment/KanaalVanWalcheren/embankments.shp"

compute_values(points_shp="output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_points_test.shp", metric_shp="metric_test.shp", midpoints_shp=midpoints_shp, unique_landuse_csv='output/parameters/unique_landuses.csv', halves_shp=halves_shp, building_intersections_shp=intersections, closest_building_intersections_shp=closest_intersections, embankment_shp=embankments_file)

# embankment = gpd.read_file(embankments_file)
# column_data = embankment[['geometry']]
# column_data.to_csv('embankment_geometry.csv', index=False)
# DEBUGGING
# midpts = gpd.read_file(midpoints_shp)
# print(midpts.columns)
# halves = gpd.read_file(halves_shp)
# print((halves.columns))
# points = gpd.read_file(points_shp)
# print(f"points columnc \n {points.columns}")
# intersections = gpd.read_file(intersections)
# Assuming 'gdf' is your GeoDataFrame and 'your_column_name' is the column you're checking

# pts = gpd.read_file(points)
# selected_col = pts[['id', 'imperv']]
# selected_col.to_csv("imperv.csv", index=False)
# print(pts.head(1))
metr = gpd.read_file("metric_test.shp")
# print(metr[['id', 'flood_rat', 'flood_mean', 'flood_max', 'flood_std']].head(1))
# print(metr[['landuse_0', 'landuse_1', 'landuse_2', 'landuse_3', 'landuse_4','landuse_5','landuse_6']].head(1))
# print(metr[['landuse_7','landuse_8','landuse_9','landuse_10','landuse_11','landuse_12','landuse_13']].head(1))
# print(metr[['id', 'imperv']].tail(10))
print(metr[['building', 'buil_heig', 'rs_wid', 'visible' , 'vis_layer']].tail(30))
# metr.to_csv("metric_test.csv", index=False)