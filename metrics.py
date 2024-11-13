"""

"""

import geopandas as gpd
import pandas as pd
import ruptures as rpt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_gradient_magnitude
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
    else:
        # If there is not intersection, then we take the distance to the end of the section as river space width
        embank_row = embankment[(embankment['id'] == side_id) & (embankment['side'] == side_side)]
        embank_pt = embank_row['split_h_di'].iloc[0]
        river_space_width = length_side - embank_pt

        first_building_height = first_intersection_point = 0

    intersections_lengths = []

    # Find lengths of building intersections
    for _, row in intersections.iterrows():
        if row['id'] == side_id and row['side'] == side_side:
            intersections_lengths.append(row.geometry.length)

    total_length_intersections = sum(intersections_lengths)
    if length_side and total_length_intersections:
        ratio = total_length_intersections / length_side
    else:
        ratio = 0

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
    ratio = count_1 / valid_point_count
    # print(f"ratio is {ratio} for {group.iloc[0]['id']}")
    return ratio

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


def tree_cover_ratio(group, intersections):
    tree_count = 0

    for idx, row in group.iterrows():
        dtm = row['elev_dtm']
        dsm = row['elev_dsm']
        if dtm is not None and dsm is not None:
            elev_diff = dsm - dtm
            is_tree = True
            if elev_diff > 2:
                id = row['id']
                side = row['side']
                point = row['geometry']
                intersect_rows = intersections[(intersections['id'] == side) & (intersections['side'] == side)]
                # print(f"intersections {intersect_rows} and type \n {type(intersect_rows)}")
                if not intersect_rows.empty:
                    for row in intersect_rows:
                        line = row['geometry']
                        if isinstance(line, LineString):
                            intersection = point.intersects(line)
                            if intersection:
                                is_tree = False
                                break
                if is_tree:
                    tree_count +=1

    # print(f"tree count {tree_count} length group {len(group)}")
    return tree_count / len(group)


def detect_barriers(group, threshold, min_change_threshold, embankment):

    # Only consider points on land so filter based on split_h_dist
    side_id = group.iloc[0]['id']
    side_side = group.iloc[0]['side']
    embank_row = embankment[(embankment['id'] == side_id) & (embankment['side'] == side_side)]
    embank_h_dist = embank_row['split_h_di'].values[0]


    filtered_group = group[group['split_h_di'] > embank_h_dist]
    # print(f"group length {len(group)} and filtered group lenth {len(filtered_group)}")

    # Extract the necessary data from the GeoDataFrame
    x = filtered_group.geometry.x
    y = filtered_group.geometry.y
    elevation = filtered_group['elev_dtm'].to_numpy()

    elevation_filled = griddata((x[~np.isnan(elevation)], y[~np.isnan(elevation)]),
                                elevation[~np.isnan(elevation)],
                                (x, y), method='nearest')

    # print(f"elevation \n {elevation}")
    amount_points = len(elevation_filled)
    close_to_river = amount_points / 3

    gradient_magnitude = gaussian_gradient_magnitude(elevation_filled, sigma=1)
    barrier_indices = np.where(gradient_magnitude > threshold)[0]
    barriers_close_to_river = sum(1 for index in barrier_indices if index < close_to_river)



    model = "l2"  # 'l2' model is used for detecting changes in mean or variance
    algo = rpt.Binseg(model=model).fit(elevation)
    # change_points = algo.predict(pen = change_point_penalty)
    change_points = algo.predict(n_bkps=5)  # Adjust `n_bkps` (number of breakpoints) as necessary
    # print(f" length group is {len(elevation)} barriers \n {barrier_indices}")

    significant_change_points = []
    for idx in change_points[:-1]:  # Exclude the last change point (end of profile)
        if abs(elevation[idx] - elevation[idx - 1]) > min_change_threshold:
            significant_change_points.append(idx)
    significant_change_points = np.array(significant_change_points, dtype=int)

    # if len(barrier_indices) != 0.0:
    #     print(f" group {group.iloc[0]['id']} side {group.iloc[0]['side']} has a barrier in it")
    #     print(f" amount is {len(barrier_indices)} and {len(significant_change_points)}")

    return len(barrier_indices), barriers_close_to_river, len(significant_change_points)


def rugosity(group):
    x_coords = group.geometry.x.to_numpy()
    y_coords = group.geometry.y.to_numpy()
    dtm_elev = group['elev_dtm'].to_numpy()
    dsm_elev = group['elev_dsm'].to_numpy()

    # Remove NaN values
    valid_mask_dtm = (~np.isnan(dtm_elev))
    valid_mask_dsm = (~np.isnan(dsm_elev))

    x_dsm = x_coords[valid_mask_dsm]
    y_dsm = y_coords[valid_mask_dsm]
    x_dtm = x_coords[valid_mask_dtm]
    y_dtm = y_coords[valid_mask_dtm]
    dtm_elev = dtm_elev[valid_mask_dtm]
    dsm_elev = dsm_elev[valid_mask_dsm]

    # Calculate total horizontal distances (start to end point)
    total_horizontal_distance_dtm = np.sqrt(
        (x_dtm[-1] - x_dtm[0]) ** 2 +
        (y_dtm[-1] - y_dtm[0]) ** 2
    )

    total_horizontal_distance_dsm = np.sqrt(
        (x_dsm[-1] - x_dsm[0]) ** 2 +
        (y_dsm[-1] - y_dsm[0]) ** 2
    )

    # Calculate surface distances by summing all segments
    dtm_surface_distances = np.sqrt(
        np.diff(x_dtm) ** 2 +
        np.diff(y_dtm) ** 2 +
        np.diff(dtm_elev) ** 2
    )

    dsm_surface_distances = np.sqrt(
        np.diff(x_dsm) ** 2 +
        np.diff(y_dsm) ** 2 +
        np.diff(dsm_elev) ** 2
    )

    # Sum up the surface distances
    total_dtm_surface_distance = np.sum(dtm_surface_distances)
    total_dsm_surface_distance = np.sum(dsm_surface_distances)

    # Compute rugosity indices
    rugosity_index_dtm = total_dtm_surface_distance / total_horizontal_distance_dtm
    rugosity_index_dsm = total_dsm_surface_distance / total_horizontal_distance_dsm

    return rugosity_index_dsm, rugosity_index_dtm


def analyze_slope_distribution(group, flat_threshold):
    valid_mask_dtm = ~np.isnan(group['elev_dtm'])
    group = group[valid_mask_dtm].sort_values('split_h_di')

    # Calculate slopes between consecutive points
    dx = group['split_h_di'].diff()
    dy = group['elev_dtm'].diff()

    # Calculate slope (rise over run)
    slopes = dy / dx

    # Categorize slopes
    flat_mask = (slopes.abs() <= flat_threshold)
    positive_mask = (slopes > flat_threshold)
    negative_mask = (slopes < -flat_threshold)

    # Count points in each category
    # Note: first point has no slope (NaN), so we categorize it based on the slope to the next point
    flat_count = flat_mask.sum()
    positive_count = positive_mask.sum()
    negative_count = negative_mask.sum()
    total_points = len(group) - 1  # Excluding first point which has no slope

    flat = round(flat_count / total_points * 100, 2)
    positive = round(positive_count / total_points * 100, 2)
    negative = round(negative_count / total_points * 100, 2)
    mean_slope = round(slopes.mean(), 4)
    max_slope = round(slopes.max(), 4)
    min_slope = round(slopes.min(), 4)

    def count_segments(mask):
        segments = 0
        in_segment = False

        for value in mask:
            if value:
                if not in_segment:
                    segments += 1
                    in_segment = True
            else:
                in_segment = False
        return segments

    flat_patches = count_segments(flat_mask)
    positive_patches = count_segments(positive_mask)
    negative_patches = count_segments(negative_mask)
    patches = flat_patches + positive_patches + negative_patches

    return flat, positive, negative, mean_slope, max_slope, min_slope, patches


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
    # embankment h_dist is in terms of h_distance
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
            tree_ratio = tree_cover_ratio(group, building_intersections)
            nr_barriers, nr_barriers_close_to_river, nr_change_points = detect_barriers(group, 0.5, 0.05, embankment)
            rug_dsm, rug_dtm = rugosity(group)
            flat, positive, negative, mean_slope, max_slope, min_slope, patches = analyze_slope_distribution(group, 0.1)
        else:
            # Set all metrics to None for invalid cross-sections
            flood_ratio = None
            flood_mean, flood_max, flood_std = None, None, None
            landuse_ratios = {f'landuse_{idx}': None for idx in range(len(unique_landuses))}
            imperv_ratio = None
            building_ratio, first_building_height, first_intersection_point, river_space_width = None, None, None, None
            vis_rat, vis_layerdness = None, None
            tree_ratio = None
            nr_barriers, nr_barriers_close_to_river, nr_change_points = None, None, None
            rug_dsm, rug_dtm = None, None
            flat, positive, negative, mean_slope, max_slope, min_slope, patches = None, None, None, None, None, None, None

            # This should update the metrics geodataframe
        row_data = {
            'id': idx,
            'side': side,
            'geometry': geometry,
            'slope_flat': flat,
            'slope_pos': positive,
            'slope_neg':negative,
            'slope_mean': mean_slope,
            'barriers': nr_barriers,
            'change_pts': nr_change_points,
            'rug_dtm': rug_dtm,
            'rug_dsm': rug_dsm,
            'building': building_ratio,
            'tree': tree_ratio,
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
metrics_output = "output/metrics/metrics/metrics.shp"

compute_values(points_shp="output/cross_sections/KanaalVanWalcheren/KanaalVanWalcheren_points_test.shp", metric_shp=metrics_output, midpoints_shp=midpoints_shp, unique_landuse_csv='output/parameters/unique_landuses.csv', halves_shp=halves_shp, building_intersections_shp=intersections, closest_building_intersections_shp=closest_intersections, embankment_shp=embankments_file)

# embankment = gpd.read_file(embankments_file)
# print(embankment.columns)
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
metr = gpd.read_file(metrics_output)
# print(metr[['id', 'flood_rat', 'flood_mean', 'flood_max', 'flood_std']].head(1))
# print(metr[['landuse_0', 'landuse_1', 'landuse_2', 'landuse_3', 'landuse_4','landuse_5','landuse_6']].head(1))
# print(metr[['landuse_7','landuse_8','landuse_9','landuse_10','landuse_11','landuse_12','landuse_13']].head(1))
# print(metr[['id', 'imperv']].tail(10))

print(metr[['id','barriers', 'slope_flat','slope_pos','slope_neg','slope_mean', 'rs_wid']].head(100))
max_value = metr['barriers'].max()
average_value = metr['barriers'].mean()
print(f"Max: {max_value}")
print(f"Average: {average_value}")

max_value_row = metr.loc[metr['barriers'] == metr['barriers'].max()]
max_id = max_value_row['id'].values[0]  # Assuming 'id' is the column name
max_side = max_value_row['side'].values[0]
print(f"Max Value: {metr['barriers'].max()}")
print(f"ID at Max: {max_id}")
print(f"Side at Max: {max_side}")

# ONLY IF YOU HAVE NAN FOR EVEYR COLUMN, IS IT A INVALID. ALDO WILL ONLY HAVE 1 ROW WITH THAT ID
# i mean, there is one row with id 88, but two for id 1
# metr.to_csv("metric_test.csv", index=False)