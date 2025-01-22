import os
from shapely import MultiLineString, MultiPoint, GeometryCollection
from shapely.ops import unary_union
from tqdm import tqdm
import overpy
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# River data retrieval--------------------------------------------------------------------------------------------------
def fetch_waterways_in_areas(areas_gdf, output_file):
    """
    Fetches all rivers and canals within specified polygon areas using OSM Overpass API. I use it to retrieve
    all osm rivers and canals that lay in urban areas in the netherlands

    Parameters:
    areas_gdf (GeoDataFrame): GeoDataFrame containing polygon geometries of non-urban areas.
    output_file (str): Path where to save the output shapefile

    Returns:
    """
    api = overpy.Overpass()
    all_lines = []

    # Convert areas to WGS84 if they aren't already
    if areas_gdf.crs != "EPSG:4326":
        areas_gdf = areas_gdf.to_crs("EPSG:4326")

    # Process each polygon
    for idx, area in areas_gdf.iterrows():
        # Convert polygon coordinates to string for query
        coords = []
        if isinstance(area.geometry, Polygon):
            exterior_coords = area.geometry.exterior.coords[:-1]  # Exclude last point as it's same as first
            coords = ' '.join([f'{lat} {lon}' for lon, lat in exterior_coords])
        elif isinstance(area.geometry, MultiPolygon):
            # Take the exterior coordinates of the largest polygon
            largest_poly = max(area.geometry.geoms, key=lambda p: p.area)
            exterior_coords = largest_poly.exterior.coords[:-1]
            coords = ' '.join([f'{lat} {lon}' for lon, lat in exterior_coords])

        # Overpass query for rivers and canals within the area
        query = f"""
            [out:json];
            (
                way["waterway"="river"](poly:"{coords}");
                way["waterway"="canal"](poly:"{coords}");
            );
            out body;
            >;
            out skel qt;
        """

        try:
            result = api.query(query)

            # Process ways
            for way in result.ways:
                coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
                if len(coords) >= 2:  # Only create valid lines
                    properties = {
                        'osm_id': way.id,
                        'waterway_type': way.tags.get('waterway'),
                        'name': way.tags.get('name', 'unknown')
                    }
                    all_lines.append({
                        'geometry': LineString(coords),
                        'properties': properties
                    })

            print(f"Processed area {idx} successfully")

        except Exception as e:
            print(f"Error processing area {idx}: {str(e)}")
            continue

    # Create GeoDataFrame from all collected lines
    if all_lines:
        geometries = [line['geometry'] for line in all_lines]
        properties = [line['properties'] for line in all_lines]

        gdf = gpd.GeoDataFrame(
            properties,
            geometry=geometries,
            crs="EPSG:4326"
        )

        # Convert to Dutch coordinate system
        gdf = gdf.to_crs("EPSG:28992")

        # Save to shapefile
        gdf.to_file(output_file, driver='ESRI Shapefile')
        print(f"Shapefile saved successfully: {output_file}")
        return gdf
    else:
        print("No waterways found in the specified areas")
        return None

    return

# CS preprocess STEP 1------------------------------------------------------------------------------------------------
# File keeping track of results
def append_to_line_in_file(filename, step, content_to_add):
    try:
        # Open the file in read mode and load all lines into a list
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write('\n')

        with open(filename, 'r') as file:
            lines = file.readlines()

        # Check if the step is within the valid range (0-based index)
        if 0 <= step < len(lines):
            # Completely replace the specific line at 'step'
            lines[step] = content_to_add + '\n'
        else:
            # If step is beyond the last line, append to the end
            lines.append(content_to_add + '\n')

        # Open the file in write mode and save the updated content
        with open(filename, 'w') as file:
            file.writelines(lines)

        print(f"Content successfully added for step {step}.")
    except Exception as e:
        print(f"Error: {e}")

#section extraction
def cross_section_extraction(river_file, interval, width, directory, step, txt_file):
    """

    Args:
        river_file (str): This is the path to cleaned river shapefile containing OSM rivers that lay in urban areas and not in bgt 'watervlaktes' except harbours
        interval (int): The distance in meters between each initial cross-section
        width (int): Width in meters of each initial cross-section
        output_file_cs (str): Output path shapefile .shp for initial cross-sections
        output_river_mid (str): Output path shapefile .shp for mid points of each cross-section

    Returns:

    """
    print("Reading file...")
    river = gpd.read_file(river_file)
    projected_gdf = river.to_crs(epsg=28992)
    cross_sections = []
    river_points = []
    total_length = 0

    # starting_point for cs id
    i = 0

    for index, row in tqdm(projected_gdf.iterrows(), total=projected_gdf.shape[0], desc="Processing each river"):
        riverline = row['geometry']
        # fid is an unique identifier OSM gives when retrieving the data. We use it here as the river id
        river_fid = row['fid']

        # print("index, riverline length: ", index, riverline.length)
        # distance = 0
        total_length += riverline.length

        # start 1 meter ahead to get sections. This is because sometimes starting a section on the beginning of the line causes problems, especially for clean_riverwidth_crossing_river, cause then it automatically crosses 2 rivers
        for distance_along_line in np.arange(1, riverline.length, interval):
            side_0, side_1, point_on_line = get_perpendicular_cross_section(riverline, distance_along_line, width)

            cross_sections.append({
                'geometry': side_0,
                'rv_id': river_fid,
                'cs_id': i,
                'side': 0,
                'corner': 0
            })
            cross_sections.append({
                'geometry': side_1,
                'rv_id': river_fid,
                'cs_id': i,
                'side': 1,
                'corner': 0
            })

            river_points.append({'geometry': point_on_line,
                                 'rv_id': river_fid,
                                 'cs_id': i,
                                 'corner': 0})
            i += 1
            # distance = distance_along_line

        # starting_point = interval - (riverline.length - distance)
        cornerlines, cornerpoints, i_value = extract_corners_lines(row, i, 170, width)
        for corner_line in cornerlines:
            cross_sections.append(corner_line)
        for corner_point in cornerpoints:
            river_points.append(corner_point)
        i = i_value

    print("Saving files...")
    # Save cross-sections to a Shapefile (.shp)
    gdf = gpd.GeoDataFrame(cross_sections)
    gdf.set_crs(epsg=28992, inplace=True)
    # segments_output_shp = os.path.join(directory, "CS/lines", f"cs_{step}")
    cs_output = cs_input(directory, step)
    gdf.to_file(cs_output, driver='ESRI Shapefile')

    # Save river midpoints to shapefile (.shp)
    river_points_gdf = gpd.GeoDataFrame(river_points)
    river_points_gdf.set_crs(epsg=28992, inplace=True)
    # midpoints_output_shp = os.path.join(directory, "CS/mid",f"mid_{step}" )
    mid_output = mid_input(directory, step)
    # os.makedirs(mid_input, exist_ok=True)
    river_points_gdf.to_file(mid_output, driver='ESRI Shapefile')

    # Get the current date and time
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    content = f"Step {step} "+ formatted_datetime +  f": Total length river {total_length} and sections amount is {i}"
    append_to_line_in_file(txt_file, step, content)

    return


def get_perpendicular_cross_section(line, distance_along_line, width):
    """
    Generates a cross-Section perpendicular to a riverline at a given distance along the line.
    """
    # Get the first point (at distance_along_line)
    point_on_line = line.interpolate(distance_along_line)  # Interpolated point on the riverline

    # Get another point slightly ahead for tangent calculation
    small_distance = 0.001  # A very small distance ahead (you can adjust this)
    if distance_along_line + small_distance <= line.length:
        point_ahead = line.interpolate(distance_along_line + small_distance)
    else:
        point_ahead = line.interpolate(line.length)  # If near the end, use the last point on the line

    # Extract coordinates from the points
    x0, y0 = point_on_line.x, point_on_line.y
    x1, y1 = point_ahead.x, point_ahead.y

    # Calculate the angle of the tangent using arctan2
    angle = np.arctan2(y1 - y0, x1 - x0)

    # create a perpendicular cross-section with the given width
    dx = width / 2 * np.sin(angle)  # X displacement (perpendicular)
    dy = width / 2 * np.cos(angle)  # Y displacement (perpendicular)

    # Two points to define the cross-section
    p1 = Point(x0 + dx, y0 - dy)  # One side of the cross-section
    p2 = Point(x0 - dx, y0 + dy)  # Other side of the cross-section

    side_1 = LineString([point_on_line, p1])
    side_2 = LineString([point_on_line, p2])

    return side_1, side_2, point_on_line


def extract_corners_lines(river_row, cs_id_value, angle_threshold, length):
    """
    Detects corners in a riverline and generates lines at half the corner angle.
    For each corner, only keeps the lines that point away from the bend.
    Side assignment matches perpendicular cross-sections.
    """
    corner_lines = []
    corner_points = []

    def is_outward_direction(p1, p2, p3, test_vector):
        """
        Determines if a vector points outward from the bend.
        """
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])

        # Get the inner angle vector (points into the bend)
        ab = a - b
        cb = c - b
        inner_angle = (ab + cb) / np.linalg.norm(ab + cb)

        # If test_vector points in roughly opposite direction of inner_angle,
        # it's pointing outward
        dot_product = np.dot(inner_angle, test_vector)
        return dot_product < 0

    riverline = river_row['geometry']
    fid = river_row['fid']

    if isinstance(riverline, MultiLineString):
        lines = riverline.geoms
    else:
        lines = [riverline]

    for line in lines:
        coords = list(line.coords)
        for i in range(1, len(coords) - 1):
            p1 = Point(coords[i - 1])
            p2 = Point(coords[i])
            p3 = Point(coords[i + 1])

            # Calculate angle at corner
            a = np.array([p1.x, p1.y])
            b = np.array([p2.x, p2.y])
            c = np.array([p3.x, p3.y])

            ab = a - b
            cb = c - b

            cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

            if angle < angle_threshold:
                # Always add the corner point
                corner_points.append({
                    'geometry': p2,
                    'rv_id': fid,
                    'cs_id': cs_id_value,
                    'corner': 1
                })

                # Calculate bisector vectors (one in each direction)
                length_bisector = length / 2
                ab_norm = ab / np.linalg.norm(ab)
                cb_norm = cb / np.linalg.norm(cb)
                bisector = (ab_norm + cb_norm)
                bisector = bisector / np.linalg.norm(bisector) * length_bisector

                # Get side from perpendicular cross-section logic
                side_0, side_1, _ = get_perpendicular_cross_section(line, line.project(p2), length)

                # Create both possible lines
                line1 = LineString([p2, Point(b + bisector)])
                line2 = LineString([p2, Point(b - bisector)])

                # Check which direction is outward
                if is_outward_direction(p1, p2, p3, bisector):
                    # The outward line points in the positive bisector direction
                    # Compare with side_0's direction to determine which side it matches
                    if np.dot(bisector, np.array([side_0.coords[-1][0] - p2.x,
                                                  side_0.coords[-1][1] - p2.y])) > 0:
                        side = 0
                    else:
                        side = 1
                    corner_lines.append({
                        'geometry': line1,
                        'rv_id': fid,
                        'cs_id': cs_id_value,
                        'side': side,
                        'corner': 1
                    })
                    cs_id_value += 1
                else:
                    # The outward line points in the negative bisector direction
                    if np.dot(-bisector, np.array([side_0.coords[-1][0] - p2.x,
                                                   side_0.coords[-1][1] - p2.y])) > 0:
                        side = 0
                    else:
                        side = 1
                    corner_lines.append({
                        'geometry': line2,
                        'rv_id': fid,
                        'cs_id': cs_id_value,
                        'side': side,
                        'corner': 1
                    })
                    cs_id_value += 1

                # Create extra perpendicular lines near the corner
                # distance_along_line = line.project(p2)
                # for offset in [-2, 2]:
                #     new_distance = distance_along_line + offset
                #     if 0 <= new_distance <= line.length:
                #         side_0, side_1, point_on_line = get_perpendicular_cross_section(
                #             line, new_distance, length
                #         )
                #
                #         corner_points.append({
                #             'geometry': point_on_line,
                #             'rv_id': fid,
                #             'cs_id': cs_id_value,
                #             'corner': 2
                #         })
                #
                #
                #         # Add the perpendicular line pointing outward
                #         vec_0 = np.array([side_0.coords[-1][0] - point_on_line.x,
                #                           side_0.coords[-1][1] - point_on_line.y])
                #         if is_outward_direction(p1, point_on_line, p3, vec_0):
                #             corner_lines.append({
                #                 'geometry': side_0,
                #                 'rv_id': fid,
                #                 'side': 0,
                #                 'cs_id': cs_id_value,
                #                 'corner': 2
                #             })
                #         else:
                #             corner_lines.append({
                #                 'geometry': side_1,
                #                 'rv_id': fid,
                #                 'side': 1,
                #                 'cs_id': cs_id_value,
                #                 'corner': 2
                #             })
                #         cs_id_value += 1



    return corner_lines, corner_points, cs_id_value


def cleaning_initial_sections(cs_in, mid_in, waterdelen_shp, overbruggingsdeel_shp, directory, step, txt_file):
    cs_shp = cs_input(directory, cs_in)
    mid_shp = mid_input(directory, mid_in)

    print("Reading files...")
    gdf_cs = gpd.read_file(cs_shp)
    gdf_mid = gpd.read_file(mid_shp)
    gdf_bridges = gpd.read_file(overbruggingsdeel_shp)
    water_gdf = gpd.read_file(waterdelen_shp)

    print("Executing cleaning functions")
    # print("Step 2.1")
    # content1, removed_points_nonwater = remove_midpoints_not_in_water(water_gdf, gdf_cs, gdf_mid, 2.1)
    # print("Step 2.2")
    content, removed_points_bridge, mid_gdf = clean_bridges(gdf_cs, gdf_mid, gdf_bridges, 2.2)


    removed_points = removed_points_bridge
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    total_content = f"Step {step}" + formatted_datetime + ": " + content

    append_to_line_in_file(txt_file, step, total_content)

    print("Filtering data...")
    filtered_mid = mid_gdf[~mid_gdf['cs_id'].isin(removed_points)]
    gdf_cs_cleaned = gdf_cs[~gdf_cs['cs_id'].isin(removed_points)]

    # Save the cleaned GeoDataFrame
    print("Saving cleaned section file...")
    gdf_cs_cleaned.set_crs(epsg=28992, inplace=True)
    segments_cleaned_path = os.path.join(directory, "CS/lines", f"cs_{step}")
    os.makedirs(segments_cleaned_path, exist_ok=True)
    gdf_cs_cleaned.to_file(segments_cleaned_path, driver="ESRI Shapefile")

    print("Saving cleaned midpoint file...")
    filtered_mid.set_crs(epsg=28992, inplace=True)
    midpoints_output_shp = os.path.join(directory, "CS/mid", f"mid_{step}")
    os.makedirs(midpoints_output_shp, exist_ok=True)
    filtered_mid.to_file(midpoints_output_shp, driver='ESRI Shapefile')

    return segments_cleaned_path, midpoints_output_shp


def clean_bridges(gdf_cs, gdf_mid, gdf_bridges, step):

    # Ensure crs
    gdf_cs = gdf_cs.to_crs(epsg=28992)
    gdf_mid = gdf_mid.to_crs(epsg=28992)
    gdf_bridges = gdf_bridges.to_crs(epsg=28992)

    removed_points = set()
    # initialize bridge column. 0 means no bridge, 1 means bridge
    gdf_mid['bridge'] = 0

    print("Building spatial index for midpoints...")
    mid_sindex = gdf_mid.sindex
    #this takes 15 seconds
    for index, row in tqdm(gdf_bridges.iterrows(), total=gdf_bridges.shape[0],
                           desc="Processing each bridge"):
        bridge_geom = row['geometry']

        possible_matches_index = list(mid_sindex.intersection(bridge_geom.bounds))
        possible_matches = gdf_mid.iloc[possible_matches_index]
        intersections = possible_matches[possible_matches.intersects(bridge_geom)]

        # intersections = gdf_mid[gdf_mid.intersects(bridge_geom)]

        if not intersections.empty:
            # print(f"Intersection found! for bridge {index}")

            for idx, point in intersections.iterrows():
                pt_id = point['cs_id']

                # Midpoint is marked as bridge
                removed_points.add(point['cs_id'])
                gdf_mid.loc[gdf_mid['cs_id']== pt_id, 'bridge'] = 1


    content = f"Amount of midpoints identified as bridges is {gdf_mid[gdf_mid['bridge'] == 1].shape[0]}" # 1271



    return content, removed_points, gdf_mid

# Find embankment STEP 3----------------------------------------------------------------------------------------------
def find_embankment_faster(cs_in, mid_in, dissolved_waterdelen_shp, directory, txt_file, step):
    cs_shp = cs_input(directory, cs_in)
    mid_shp = mid_input(directory, mid_in)

    print("Reading files...")
    gdf_cs = gpd.read_file(cs_shp)
    gdf_mid = gpd.read_file(mid_shp)
    gdf_water = gpd.read_file(dissolved_waterdelen_shp)

    # Create spatial index for cross-sections and midpoints
    cs_sindex = gdf_cs.sindex

    # Create dictionary for quick midpoint lookup
    mid_dict = dict(zip(gdf_mid['cs_id'], gdf_mid.geometry))

    # Dictionary to store all intersections for each cross-section
    # Key: (cs_id, side), Value: list of intersection points
    all_intersections = {}


    for idx, water_row in tqdm(gdf_water.iterrows(), total=len(gdf_water), desc="Processing water features"): # Takes 3 minutes
        water_boundary = water_row.geometry.boundary

        # Find potential cross-sections that intersect with the water boundary's bounding box
        possible_cs_idx = list(cs_sindex.intersection(water_boundary.bounds))
        if not possible_cs_idx:
            continue

        # Get relevant cross-sections
        potential_cs = gdf_cs.iloc[possible_cs_idx]

        # Process each potential cross-section
        for _, cs_row in potential_cs.iterrows():
            cs_id = cs_row['cs_id']
            side = cs_row['side']
            unique_id = (cs_id, side)

            # Skip if no corresponding midpoint
            if cs_id not in mid_dict:
                continue

            section_geom = cs_row.geometry

            try:
                if section_geom.intersects(water_boundary):
                    intersection = section_geom.intersection(water_boundary)

                    if intersection.is_empty:
                        continue

                    intersection_points = []
                    # Convert intersection to point(s)
                    if isinstance(intersection, (Point, LineString)):
                        intersection_points.append(
                            Point(intersection.coords[0] if isinstance(intersection, LineString) else intersection))
                    elif isinstance(intersection, (MultiPoint, MultiLineString)):
                        intersection_points.extend([Point(geom.coords[0] if isinstance(geom, LineString) else geom)
                                                    for geom in intersection.geoms])
                    elif isinstance(intersection, GeometryCollection):
                        for geom in intersection.geoms:
                            if isinstance(geom, Point):
                                intersection_points.append(geom)
                            elif isinstance(geom, LineString):
                                intersection_points.append(Point(geom.coords[0]))

                    if intersection_points:
                        # Add intersection points to the dictionary
                        if unique_id not in all_intersections:
                            all_intersections[unique_id] = []
                        all_intersections[unique_id].extend(intersection_points)

            except Exception as e:
                print(f"Error processing intersection for cs_id {cs_id}, side {side}: {e}")
                continue


    embankment_results = {}  # Key: (cs_id, side, corner), Value: (intersection_point, distance)

    # Process all intersections to find the closest point for each cross-section
    for unique_id, intersection_points in tqdm(all_intersections.items(), desc="Finding closest points"): #8 seconds
        cs_id, side,  = unique_id
        midpoint_geom = mid_dict[cs_id]

        # Find closest intersection point among all collected points
        closest_intersection = min(intersection_points, key=lambda x: midpoint_geom.distance(x))
        mid_intersection_distance = midpoint_geom.distance(closest_intersection)
        embankment_results[unique_id] = (closest_intersection, mid_intersection_distance)

    print("Making the embankment file...")
    embankments = []
    for (cs_id, side), (intersection_point, distance) in embankment_results.items():
        section_data = gdf_cs[(gdf_cs['cs_id'] == cs_id) & (gdf_cs['side'] == side)].iloc[0]
        embankments.append({
            'cs_id': cs_id,
            'rv_id': section_data['rv_id'],
            'side': side,
            'embank': intersection_point,
            'mid_dist': distance,
            'corner': section_data['corner']
        })

    content = f"Step {step}: total embankment points is {len(embankments)}, removed cs total is {len(gdf_cs) - len(embankments)}"
    append_to_line_in_file(txt_file, step, content)

    # Keep only cross-sections with embankment points
    mask1 = [(row['cs_id'], row['side']) in embankment_results for _, row in gdf_cs.iterrows()]
    gdf_cs = gdf_cs[mask1]
    mask2 = [(row['cs_id']) in embankment_results for _, row in gdf_mid.iterrows()]
    gdf_mid= gdf_mid[mask2]

    print("Saving cleaned files...")
    gdf_cs.set_crs(epsg=28992, inplace=True)
    cs_out_path = cs_input(directory, step)
    gdf_cs.to_file(cs_out_path, driver="ESRI Shapefile")

    gdf_mid.set_crs(epsg=28992, inplace=True)
    mid_out_path = mid_input(directory, step)
    gdf_mid.to_file(mid_out_path, driver="ESRI Shapefile")

    gdf_embank = gpd.GeoDataFrame(embankments, geometry='embank')
    gdf_embank.set_crs(epsg=28992, inplace=True)
    embank_out_path = embank_input(directory, step)
    gdf_embank.to_file(embank_out_path)


#RIVERWIDTH CREATION AND CLEANING STEP 4------------------------------------------------------------------------------
def create_and_clean_riverwidth(embank_in, mid_in, rivers_shp, directory, step, txt_file):
    embank_shp = embank_input(directory, embank_in)
    mid_shp = mid_input(directory, mid_in)

    print("Reading files...")
    embank_gdf = gpd.read_file(embank_shp)
    mid_gdf = gpd.read_file(mid_shp)
    rivers_gdf = gpd.read_file(rivers_shp)

    riverwidth_gdf = create_river_width_file(embank_gdf, mid_gdf)
    # # riverwidth_gdf.set_crs(epsg=28992, inplace=True)
    output_file_41 = riverwidth_input(directory, 1)
    riverwidth_gdf.to_file(output_file_41)

    riverwidth_gdf, embank_gdf, content1 = clean_riverwidth_crossing_river(riverwidth_gdf, embank_gdf, rivers_gdf)
    # riverwidth_gdf.set_crs(epsg=28992, inplace=True)
    output_file_42 = riverwidth_input(directory, 2)
    riverwidth_gdf.to_file(output_file_42)

    # riverwidth_gdf = gpd.read_file(output_file_42)

    riverwidth_gdf, embank_gdf, content2 = clean_large_riverwidth_(riverwidth_gdf, embank_gdf)
    # riverwidth_gdf.set_crs(epsg=28992, inplace=True)
    output_file_43 = riverwidth_input(directory, 3)
    riverwidth_gdf.to_file(output_file_43)

    # groups = "output/CS/embank/river_width_groups.shp"
    riverwidth_gdf, embank_gdf, content3 = clean_cross_sections_riverwidth_crossing_riverwidth_directions(riverwidth_gdf, embank_gdf)
    # riverwidth_gdf.set_crs(epsg=28992, inplace=True)
    output_file_44 = riverwidth_input(directory, 4)
    riverwidth_gdf.to_file(output_file_44, driver="ESRI Shapefile")

    # embank_gdf.set_crs(epsg=28992, inplace=True)
    output_file_embank = embank_input(directory, step)
    embank_gdf.to_file(output_file_embank, driver="ESRI Shapefile")


    total_content = f"Step 4.2: " + content1 + f"Step 4.3: " + content2 + f"Step 4.4: " + content3
    append_to_line_in_file(txt_file, step, total_content)

def create_river_width_file(embank_gdf, mid_gdf):
    grouped = embank_gdf.groupby('cs_id')
    river_width_list = []

    for idx, group in tqdm(grouped, total=len(grouped),
                       desc="Processing each section, making river width"):
        if len(group) == 2:
            row_0 = group.iloc[0]
            geom_0, cs_id, rv_id, side_0, mid_dist_0= row_0['geometry'], row_0['cs_id'], row_0['rv_id'], row_0['side'], row_0['mid_dist']
            row_1 = group.iloc[1]
            geom_1, side_1, mid_dist_1 = row_1['geometry'], row_1['side'], row_1['mid_dist']

            mid_row = mid_gdf[mid_gdf['cs_id'] == cs_id]
            mid_geom = mid_row['geometry'].iloc[0]
            # mid_geom = mid_row['geometry']
            # mid_coord = (mid_geom.iloc[0].x, mid_geom.iloc[0].y)

            river_width = mid_dist_0 + mid_dist_1
            # river_geom_0 = LineString([mid_coord, geom_0.coords[0]])
            # river_geom_1 = LineString([mid_coord, geom_1.coords[0]])
            river_geom_0 = LineString([mid_geom, geom_0.coords[0]])
            river_geom_1 = LineString([mid_geom, geom_1.coords[0]])

            river_width_list.append({'geometry': river_geom_0,'cs_id':cs_id, 'rv_id':rv_id, 'side':side_0, 'mid_dist': mid_dist_0, 'corner': 0})
            river_width_list.append({'geometry': river_geom_1,'cs_id':cs_id, 'rv_id':rv_id, 'side':side_1, 'mid_dist': mid_dist_1, 'corner': 0})
        elif len(group == 1):
            row = group.iloc[0]
            geom, cs_id, rv_id, side, mid_dist, corner = row['geometry'], row['cs_id'], row['rv_id'], row['side'], row['mid_dist'], row['corner']

            mid_row = mid_gdf[mid_gdf['cs_id'] == cs_id]
            mid_geom = mid_row['geometry'].iloc[0]
            # mid_coord = (mid_geom.iloc[0].x, mid_geom.iloc[0].y)

            # river_geom = LineString([mid_coord, geom.coords[0]])
            river_geom = LineString([mid_geom, geom.coords[0]])

            river_width_list.append({'geometry': river_geom, 'cs_id': cs_id,'rv_id':rv_id, 'side': side, 'mid_dist': mid_dist, 'corner': corner})
        else:
            print(f"Skipping cs_id {idx}: Expected 1 or 2 rows, found {len(group)}")
            continue

    river_width_gdf = gpd.GeoDataFrame(river_width_list)

    return river_width_gdf

def clean_riverwidth_crossing_river(rvwidth_gdf, embank_gdf, rivers_gdf):

    rvwidth_sindex = rvwidth_gdf.sindex
    indices_to_drop = set()

    for index, row in tqdm(rivers_gdf.iterrows(), total=rivers_gdf.shape[0],desc="Processing each river, checking if a riverwidth intersects with it..."):
        line_geom = row['geometry']
        rv_id = row['fid']

        possible_matches_rvwidth_index = list(rvwidth_sindex.intersection(line_geom.bounds))
        possible_matches_rvwidth = rvwidth_gdf.iloc[possible_matches_rvwidth_index]

        intersections_rvwidth = possible_matches_rvwidth[possible_matches_rvwidth.intersects(line_geom)]
        filtered_intersections_rvwidth = intersections_rvwidth[intersections_rvwidth['rv_id'] != rv_id]
        indices_to_drop.update(filtered_intersections_rvwidth.index)


    dropped_rvwidth_rows = rvwidth_gdf.loc[list(indices_to_drop), ['cs_id', 'side']]
    dropped_conditions = set(zip(dropped_rvwidth_rows['cs_id'], dropped_rvwidth_rows['side']))

    # Removing items from rv_width
    rvwidth_gdf = rvwidth_gdf.drop(indices_to_drop)
    # Removing items from embankment
    embank_gdf = embank_gdf[
        ~embank_gdf.apply(lambda row: (row['cs_id'], row['side']) in dropped_conditions, axis=1)]

    # print("Saving files...")
    # rvwidth_gdf.to_file(rvwidth_output, driver="ESRI Shapefile")
    # embank_gdf.to_file(embank_output, driver="ESRI Shapefile")
    content = f"Dropped indices is {len(indices_to_drop)}"

    return rvwidth_gdf, embank_gdf, content

def clean_large_riverwidth_(rvwidth_gdf, embank_gdf):
    # print("Reading files...")
    # rvwidth_gdf = gpd.read_file(riverwidth_in)
    # embank_gdf = gpd.read_file(embank_in)

    print("Making columns...")
    rvwidth_gdf['length'] = rvwidth_gdf['geometry'].length
    rvwidth_gdf['keep'] = False

    print("Group by river...")
    river_group = rvwidth_gdf.groupby('rv_id')

    for idx, river in tqdm(river_group, total=len(river_group), desc="Processing each river, checking for extra wide riverwidths..."):
        avg_length = river['length'].mean()
        std_dev_length = river['length'].std()

        threshold_lower = avg_length - 3 * std_dev_length
        threshold_upper = avg_length + 3 * std_dev_length

        rvwidth_gdf.loc[river.index, 'keep'] = river['length'].between(threshold_lower, threshold_upper)

    print("Filtering geodataframes...")
    filtered_rvwidth_gdf = rvwidth_gdf[rvwidth_gdf['keep']]
    keep_sections = filtered_rvwidth_gdf[['cs_id', 'side']]
    filtered_embank_gdf = embank_gdf.merge(keep_sections, on=['cs_id', 'side'], how='inner')

    content = f"Removed rvwidth length is {len(rvwidth_gdf) - len(filtered_rvwidth_gdf)}, Removed embankments is {len(embank_gdf) - len(filtered_embank_gdf)}"

    return filtered_rvwidth_gdf, filtered_embank_gdf, content


# this is computingdirections
def calculate_line_direction(linestring):
    """Calculate the primary direction of a linestring in degrees from north (0-180)."""
    """Calculate the primary direction of a linestring in degrees (0-360)"""

    coords = np.array(linestring.coords)
    if len(coords) < 2:
        return 0

    start, end = coords[0], coords[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    # angle = math.degrees(math.atan2(dx, dy)) % 180
    angle = math.degrees(math.atan2(dx, dy))
    return angle


def find_intersecting_groups(gdf):
    """
    Create groups of intersecting linestrings using networkx.
    Ensures no group contains multiple items with the same cs_id.
    """
    # Create a graph where nodes are linestring indices
    G = nx.Graph()

    # Create spatial index
    sindex = gdf.sindex

    # First pass: find all intersections
    print("Finding intersections...")
    intersections = []
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Finding intersections"):
        bounds = row.geometry.bounds
        current_cs_id = row['cs_id']

        possible_matches_index = list(sindex.intersection(bounds))
        possible_matches = gdf.iloc[possible_matches_index]

        # Filter out lines with the same cs_id
        possible_matches = possible_matches[possible_matches['cs_id'] != current_cs_id]

        for match_idx, match_row in possible_matches.iterrows():
            if idx != match_idx and row.geometry.intersects(match_row.geometry):
                intersections.append((idx, match_idx))

    # Second pass: build groups ensuring no cs_id conflicts
    print("Building groups...")
    for idx1, idx2 in tqdm(intersections, desc="Building groups"):
        # Check if adding this edge would create a group with duplicate cs_ids
        if not G.has_edge(idx1, idx2):
            # Get connected components (groups) that these nodes belong to
            comp1 = set([idx1]) if idx1 not in G else set(nx.node_connected_component(G, idx1))
            comp2 = set([idx2]) if idx2 not in G else set(nx.node_connected_component(G, idx2))

            # Get cs_ids for potential combined group
            combined_cs_ids = set(gdf.loc[list(comp1.union(comp2)), 'cs_id'])

            # Only add edge if it won't create cs_id conflicts
            if len(combined_cs_ids) == len(comp1.union(comp2)):
                G.add_edge(idx1, idx2)

    # Find connected components (groups of intersecting linestrings)
    groups = list(nx.connected_components(G))

    # Add isolated nodes as single-element groups
    connected_nodes = set().union(*groups) if groups else set()
    isolated_nodes = set(gdf.index) - connected_nodes
    groups.extend([{node} for node in isolated_nodes])

    return groups


def process_group(group_indices, gdf):
    """Process a group of linestrings using median-based deviation."""
    if len(group_indices) == 1:
        return group_indices

    group_indices_list = list(group_indices)
    group_geometries = gdf.loc[group_indices_list]

    # Calculate directions
    directions = {idx: calculate_line_direction(geom) for idx, geom
                  in zip(group_indices_list, group_geometries.geometry)}

    angles = np.array(list(directions.values()))
    median_angle = np.median(angles)

    def angle_diff(a1, a2):
        diff = abs(a1 - a2)
        # return min(diff, 180 - diff) #
        return diff

    # Calculate deviations from median
    deviations = {idx: angle_diff(angle, median_angle)
                  for idx, angle in directions.items()}

    # Calculate standard deviation from median
    median_deviations = np.array(list(deviations.values()))
    std_from_median = np.sqrt(np.mean(median_deviations ** 2))

    if len(group_indices) < 3:
        best_idx = min(deviations.keys(), key=lambda x: deviations[x])
        return {best_idx}
    else:
        keep_indices = {idx for idx, dev in deviations.items()
                        if dev <= std_from_median}
        return keep_indices


def clean_cross_sections_riverwidth_crossing_riverwidth_directions(rvwidth_gdf, embank_gdf, output_groups=None):
    # print("Reading files...")
    # rvwidth_gdf = gpd.read_file(riverwidth_input)
    # embank_gdf = gpd.read_file(embank_input)

    # Find groups of intersecting linestrings
    print("Finding intersecting groups...")
    groups = find_intersecting_groups(rvwidth_gdf)

    # If output_groups is specified, create a visualization of the groups
    if output_groups:
        print("Creating groups visualization...")
        # Create a copy of the input GeoDataFrame
        groups_gdf = rvwidth_gdf.copy()
        # Add group column initialized with -1 (for isolates)
        groups_gdf['group'] = -1
        # Assign group numbers
        for group_idx, group in enumerate(groups):
            groups_gdf.loc[list(group), 'group'] = group_idx
        # Save to file
        groups_gdf.to_file(output_groups, driver="ESRI Shapefile")

    # Process groups and collect indices to keep
    print("Processing groups...")
    indices_to_keep = set()
    for group in tqdm(groups, desc="Processing groups"):
        keep_indices = process_group(group, rvwidth_gdf)
        indices_to_keep.update(keep_indices)

    # Create new filtered GeoDataFrames
    print('Filtering...')
    indices_to_keep_list = list(indices_to_keep)
    indices_to_drop = list(set(rvwidth_gdf.index) - indices_to_keep)

    rvwidth_gdf_cleaned = rvwidth_gdf.loc[indices_to_keep_list]

    # Get the dropped rows information
    dropped_rvwidth = rvwidth_gdf.loc[indices_to_drop]
    dropped_cs_conditions = set(zip(dropped_rvwidth['cs_id'], dropped_rvwidth['side']))

    # Filter embank_gdf
    embank_gdf_cleaned = embank_gdf[
        ~embank_gdf.apply(lambda row: (row['cs_id'], row['side']) in dropped_cs_conditions, axis=1)]

    # print("Saving files...")
    # rvwidth_gdf_cleaned.to_file(rvwidth_output, driver="ESRI Shapefile")
    # embank_gdf_cleaned.to_file(embank_output, driver="ESRI Shapefile")
    content = f"Total dropped: {len(indices_to_drop)}"

    return rvwidth_gdf_cleaned, embank_gdf_cleaned, content



#Final section extraction STEP 5--------------------------------------------------------------------------------------
def section_extraction_final(cs_in, embank_in, mid_in, directory, step, txt_file):
    """
    Our initial cross-sections start from the midpoint and lay both in water and on land. Now the final segments are computed
    as perpendicular lines extending from the embankment point, 100 meters outward. Width is specified in function final_section.
    Args:
        input_cs_shp (str): Path to input sections shapefile
        embank_shp (str): Path to input embankment points shapefile
        output_file_cs (str): Path to output sections shapefile
        mid_shp (str): Path to input midpoint shapefile

    Returns:

    """
    cs_shp = cs_input(directory, cs_in)
    embank_shp = embank_input(directory, embank_in)
    mid_shp = mid_input(directory, mid_in)

    print("Reading files...")
    cs_gdf = gpd.read_file(cs_shp)
    embank_gdf = gpd.read_file(embank_shp)
    midpoint_gdf = gpd.read_file(mid_shp)

    cross_sections = []

    for idx, row in tqdm(cs_gdf.iterrows(), total=cs_gdf.shape[0], desc="Creating new sections..."): # 5 minutes
        cs_id = row['cs_id']
        side = row['side']
        rv_id = row['rv_id']
        corner = row['corner']
        embank_row = embank_gdf[(embank_gdf['cs_id'] == cs_id) & (embank_gdf['side'] == side)]
        mid_row = midpoint_gdf[midpoint_gdf['cs_id'] == cs_id]
        embank_pt = embank_row.geometry
        mid_pt = mid_row.geometry

        if embank_row.empty or mid_row.empty:
            # Happens when we have already deleted the section so it doesnt exist
            continue

        section = final_section(mid_pt, embank_pt)

        if section is None:
            # happens when embankment point is exactly the same as midpoint. We ignore
            continue

        cross_sections.append({
                'geometry': section,
                'rv_id': rv_id,
                'cs_id': cs_id,
                'side': side,
                'corner': corner
            })

    print("Saving files...")
    gdf = gpd.GeoDataFrame(cross_sections)
    gdf.set_crs(epsg=28992, inplace=True)
    cs_out = cs_input(directory, step)
    gdf.to_file(cs_out, driver='ESRI Shapefile')

    content = f"Step {step}: total amount final sections is {len(gdf)}"
    append_to_line_in_file(txt_file, step, content)

    return


def final_section(mid_pt, embank_pt, width=100):
    """
    Gets called by section_extraction_final to compute the final segments. Takes an embankment point and midpoint of the same
    cross-section and computes a LineString between the embankment point and a point 100 meter away perpendicular to the
    river, by extending from the line between midpoint and embankment point.
    Args:
        mid_pt (Point): Midpoint geometry
        embank_pt (Point): Embankment point geometry
        width (int): Width in meters of section, set at 100 meter.

    Returns: Section as LineString

    """
    # Extract the first (and should be only) geometry from the GeoSeries
    if hasattr(mid_pt, 'iloc'):
        mid_pt = mid_pt.iloc[0]
    if hasattr(embank_pt, 'iloc'):
        embank_pt = embank_pt.iloc[0]

    x0, y0 = mid_pt.x, mid_pt.y
    x1, y1 = embank_pt.x, embank_pt.y

    # direction vector
    dx = x1 - x0
    dy = y1 - y0

    # magnitude
    magnitude = (dx**2 + dy**2)**0.5

    if magnitude == 0:
        # mid and embankment are the same point
        return None

    # normalize
    direction_x = dx / magnitude
    direction_y = dy / magnitude

    # extend from embankment point
    extended_x = x1 + direction_x * width
    extended_y = y1 + direction_y * width

    # end pt section
    end_pt = Point(extended_x, extended_y)

    # section
    section = LineString([embank_pt, end_pt])

    return section



#cleaning non-urban STEP 6---------------------------------------------------------------------------------------------
def clean_cross_sections_non_urban(cs_in, embank_in, non_urban_area_shp, directory, step, txt_file):
    cs_shp = cs_input(directory, cs_in)
    embank_shp = embank_input(directory, embank_in)

    print("Reading files...")
    gdf_cs = gpd.read_file(cs_shp)
    total1= len(gdf_cs)
    gdf_non_urban_areas = gpd.read_file(non_urban_area_shp)
    gdf_embank = gpd.read_file(embank_shp)

    print("Creating spatial index for non-urban areas...")
    non_urban_sindex = gdf_non_urban_areas.sindex

    for index, row in tqdm(gdf_cs.iterrows(), total=gdf_cs.shape[0],
                       desc="Processing each section, checking if it lays in a non-urban area"):
        line_geom = row['geometry']
        cs_id = row['cs_id']
        side= row['side']

        possible_matches_index = list(non_urban_sindex.intersection(line_geom.bounds))
        possible_matches = gdf_non_urban_areas.iloc[possible_matches_index]

        intersections = possible_matches[possible_matches.intersects(line_geom)]

        if not intersections.empty:
            gdf_cs = gdf_cs.drop(index)
            embank_row = gdf_embank[(gdf_embank['cs_id'] == cs_id) & (gdf_embank['side'] == side)]
            gdf_embank = gdf_embank.drop(embank_row.index)

    total2 = len(gdf_cs)
    total = total1 - total2
    content = f"Step {step}: Removed sections total is {total}"
    append_to_line_in_file(txt_file, step, content)


    gdf_cs.set_crs(epsg=28992, inplace=True)
    gdf_embank.set_crs(epsg=28992, inplace=True)

    cs_output_file = cs_input(directory, step)
    embank_output_file = embank_input(directory, step)

    gdf_cs.to_file(cs_output_file, driver= "ESRI Shapefile")
    gdf_embank.to_file(embank_output_file, driver="ESRI Shapefile")


#Cleaning embankment line STEP 7 --------------------------------------------------------------------------------------
#To find the embankment line for checking if the segment crosses another embankment line
def order_points_along_river(points, river_line):
    """
    Order points based on their projection along the river line.
    """
    # Project points onto river line and get their distance along the river
    projected_points = [(point, river_line.project(point)) for point in points]

    # Sort points by their distance along the river
    projected_points.sort(key=lambda x: x[1])

    # Return ordered points
    return [p[0] for p in projected_points]


def create_boundary_line(boundary_points, riverline):
    # Order points along the river
    ordered_points = order_points_along_river(boundary_points, riverline)

    # Extract only x and y from POINT or POINT Z objects
    points_2d = [(point.x, point.y) for point in ordered_points]

    # Create LineString if we have enough points
    if len(points_2d) >= 2:
        return LineString(points_2d)
    else:
        # print("Warning: Not enough points for boundary")
        return None


def embankment_line(embank_in, rivers_shp, directory):

    print("Reading files...")
    rivers_gdf = gpd.read_file(rivers_shp)

    embank_shp = embank_input(directory, embank_in)
    embank_gdf = gpd.read_file(embank_shp)

    print("Creating spatial index for river...")
    rivers_sindex = rivers_gdf.sindex
    embankment_lines = []
    line_id = 0

    for index, river in tqdm(rivers_gdf.iterrows(), total=len(rivers_gdf), desc="Processing each river"):
        rv_id = river['fid']
        riverline = river['geometry']

        # Filter embankments for this river
        river_embank = embank_gdf[embank_gdf['rv_id'] == rv_id]

        # Process each side separately
        for side in river_embank['side'].unique():

            # Get points for this side
            side_points = river_embank[river_embank['side'] == side]
            boundary_points = list(side_points['geometry'])

            if boundary_points:
                boundary_line = create_boundary_line(boundary_points, riverline)

                if boundary_line is not None:

                    possible_matches_index = list(rivers_sindex.intersection(boundary_line.bounds))
                    possible_matches = rivers_gdf.iloc[possible_matches_index]

                    # Exclude the current river
                    other_rivers = possible_matches[possible_matches['fid'] != rv_id]

                    # Create a unary union of just the candidate matches
                    other_rivers_union = unary_union(other_rivers.geometry)

                    if boundary_line.intersects(other_rivers_union):
                        intersection_points = boundary_line.intersection(other_rivers_union)

                        if isinstance(intersection_points, Point):
                            intersection_points = MultiPoint([intersection_points])

                        # Split the line at intersection points
                        segments = split_line_at_points(boundary_line, intersection_points)

                        # Add each segment
                        for segment in segments:
                            embankment_lines.append({
                                'geometry': segment,
                                'rv_id': rv_id,
                                'side': side
                            })
                        else:
                            # If no intersections, add the whole line
                            embankment_lines.append({
                                'geometry': boundary_line,
                                'rv_id': rv_id,
                                'side': side
                            })

    lines_gdf = gpd.GeoDataFrame(embankment_lines)
    # lines_gdf.set_crs("EPSG:28992")
    # lines_gdf.to_file(embank_line_shp, driver="ESRI Shapefile")

    return lines_gdf

def embankment_line_old(embank_in, rivers_shp, directory):
    print("Reading files...")
    rivers_gdf = gpd.read_file(rivers_shp)

    embank_shp = embank_input(directory, embank_in)
    embank_gdf = gpd.read_file(embank_shp)

    embankment_lines = []

    for index, river in tqdm(rivers_gdf.iterrows(), total=len(rivers_gdf), desc="Processing each river"):
        rv_id = river['fid']
        riverline = river['geometry']

        # Filter embankments for this river
        river_embank = embank_gdf[embank_gdf['rv_id'] == rv_id]

        # Process each side separately
        for side in river_embank['side'].unique():
            # Get points for this side
            side_points = river_embank[river_embank['side'] == side]
            boundary_points = list(side_points['geometry'])

            if boundary_points:
                boundary_line = create_boundary_line(boundary_points, riverline)

                if boundary_line is not None:
                    embankment_lines.append({
                        'geometry': boundary_line,
                        'rv_id': rv_id,
                        'side': side
                    })

    lines_gdf = gpd.GeoDataFrame(embankment_lines)
    # lines_gdf.set_crs("EPSG:28992")
    # lines_gdf.to_file(embank_line_shp, driver="ESRI Shapefile")

    return lines_gdf


def split_line_at_points(line, points):
    """
    Split a LineString at given points and return all valid segments
    """
    if isinstance(points, Point):
        # points = MultiPoint([points])
        points = [points]
    elif isinstance(points, MultiPoint):
        points = [pt for pt in points.geoms]

    # Convert to list of coordinates
    coords = list(line.coords)
    new_coords = []
    segments = []

    # Add original vertices
    for coord in coords:
        new_coords.append(coord)
        # Check if any point is near this vertex
        point = Point(coord)
        for split_point in points:
            if point.distance(split_point) < 1e-8:  # Small threshold
                # If we have enough coords for a line, create segment
                if len(new_coords) >= 2:
                    segments.append(LineString(new_coords))
                new_coords = [coord]  # Start new segment

    # Add final segment if it exists
    if len(new_coords) >= 2:
        segments.append(LineString(new_coords))

    return segments


#clean segments that cross another embankment
def clean_riverwidth_and_segment_crossing_embankmentline(rivers_shp, rv_in, embank_in, cs_in, embankline_in_shp, embankline_out_shp, directory, step, txt_file):
    riverwidth_shp = riverwidth_input(directory, rv_in)
    cs_shp = cs_input(directory, cs_in)
    embank_shp = embank_input(directory, embank_in)

    print("Reading files...")
    rvwidth_gdf = gpd.read_file(riverwidth_shp)
    embankline_gdf = gpd.read_file(embankline_in_shp)
    cs_gdf = gpd.read_file(cs_shp)
    embankpts_gdf = gpd.read_file(embank_shp)

    rvwidth_sindex = rvwidth_gdf.sindex
    cs_sindex = cs_gdf.sindex
    indices_to_drop_rv = set()
    indices_to_drop_cs = set()

    for idx, embankline in tqdm(embankline_gdf.iterrows(), total= embankline_gdf.shape[0], desc="Processing each embankment line..."):

        line_geom =embankline['geometry']
        rv_id_embankline = embankline['rv_id']

        # If a riverwidth crosses an embankment line, we want to remove it
        possible_matches_rvwidth_index = list(rvwidth_sindex.intersection(line_geom.bounds))
        possible_matches_rvwidth = rvwidth_gdf.iloc[possible_matches_rvwidth_index]
        intersections_rvwidth = possible_matches_rvwidth[possible_matches_rvwidth.intersects(line_geom)]

        if not intersections_rvwidth.empty:
            for idx, intersection_rv in intersections_rvwidth.iterrows():
                rv_id = intersection_rv['rv_id']
                if rv_id != rv_id_embankline:
                    # indices_to_drop_rv.update(intersection_rv.index)
                    indices_to_drop_rv.add(intersection_rv.name)
                else:
                    continue

        # If a section crosses an embankment line, we want to remove it
        possible_matches_cs_index = list(cs_sindex.intersection(line_geom.bounds))
        possible_matches_cs = cs_gdf.iloc[possible_matches_cs_index]
        intersections_cs = possible_matches_cs[possible_matches_cs.intersects(line_geom)]


        if not intersections_cs.empty:
            for idy, intersection_cs in intersections_cs.iterrows():
                rv_id = intersection_cs['rv_id']
                if rv_id != rv_id_embankline:
                    # indices_to_drop_cs.update(intersection_cs.index)
                    indices_to_drop_cs.add(intersection_cs.name)
                else:
                    continue


    print("Filtering geodataframes...")
    #We filter out rvwidth-embankline intersections from the rvwidth file and rvwidth-embankline and segment-embankline intersections from cs_shp

    rvwidth_gdf_cleaned = rvwidth_gdf.drop(indices_to_drop_rv)

    # Get the dropped rows information
    dropped_rvwidth = rvwidth_gdf.loc[list(indices_to_drop_rv)]
    dropped_rvwidth_conditions = set(zip(dropped_rvwidth['cs_id'], dropped_rvwidth['side']))

    #combine dropping information of intersections riverwidth-embankment line and segment-embankment line
    combined_indices_to_drop_cs = indices_to_drop_cs | {idx for idx, row in cs_gdf.iterrows() if
                                                  (row['cs_id'], row['side']) in dropped_rvwidth_conditions}
    cs_gdf_cleaned = cs_gdf.drop(list(combined_indices_to_drop_cs))

    #fixing embankmentline and points file
    indices_to_drop_embankpts = {idx for idx, row in embankpts_gdf.iterrows() if
                                                  (row['cs_id'], row['side']) in combined_indices_to_drop_cs}
    embank_pts_cleaned = embankpts_gdf.drop(list(indices_to_drop_embankpts))

    content = f"Step {step}: Indices dropped from rv; cs & embank is {len(indices_to_drop_rv)}; {len(combined_indices_to_drop_cs)}"
    append_to_line_in_file(txt_file, step, content)

    print("Saving files...")
    embank_out = embank_input(directory, step)
    embank_pts_cleaned.to_file(embank_out, driver="ESRI Shapefile")

    embank_line_gdf = embankment_line_old(step, rivers_shp, directory)
    embank_line_gdf.to_file(embankline_out_shp, driver= "ESRI Shapefile")

    cs_out = cs_input(directory, step)
    cs_gdf_cleaned.to_file(cs_out, driver="ESRI Shapefile")

    rvwidth_out = riverwidth_input(directory, step)
    rvwidth_gdf_cleaned.to_file(rvwidth_out, driver="ESRI Shapefile")


#points STEP 8---------------------------------------------------------------------------------------------------------
def create_cross_section_points(cross_sections_shapefile, cs_width, output_gpkg, output_csv):
    """
    Creates points along cross-sections and calculates horizontal distances.

    Parameters:
    cross_sections_shapefile: Path to shapefile containing cross-section lines
    n_points: Number of points to create along each cross-section
    output_shapefile: Path where the resulting points shapefile will be saved

    Returns:
    GeoDataFrame containing points with horizontal distances
    """
    # Read cross-sections from shapefile
    cross_sections = gpd.read_file(cross_sections_shapefile)
    print(f"Loaded {len(cross_sections)} cross-sections from {cross_sections_shapefile}")

    all_points = []
    attributes = []

    n_points= 2 * cs_width
    unique_id_counter = 0

    for ind, row in tqdm(cross_sections.iterrows(), total=cross_sections.shape[0], desc="Processing cross-sections..."):
        cs_id = row['cs_id']
        side = row['side']
        rv_id = row['rv_id']

        # Extract start and end coordinates from the LineString geometry
        start_coords = list(row.geometry.coords)[0]
        end_coords = list(row.geometry.coords)[1]

        # Create points along the cross-section
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

        # Create Point objects and calculate distances
        for i, (x, y) in enumerate(zip(lon, lat)):
            point_geom = Point(x, y)
            h_distance = Point(start_coords).distance(point_geom)
            all_points.append({
                'geometry': point_geom,
                'id_uniq':unique_id_counter
            })

            attributes.append({
                'id_uniq': unique_id_counter,
                'cs_id': cs_id,
                'rv_id': rv_id,
                'side': side,
                'h_dist': h_distance
            })

            unique_id_counter += 1

    # Create GeoDataFrame
    print("Creating geodataframes...")
    gdf = gpd.GeoDataFrame(all_points)
    gdf.set_crs(epsg=28992, inplace=True)
    attributes_df = pd.DataFrame(attributes)

    # Save to shapefile
    print("Saving to files...")
    gdf.to_file(output_gpkg, driver='GPKG')
    attributes_df.to_csv(output_csv, index=False)
    print(f"Cross-section points saved to: {output_gpkg} and attributes to: {output_csv}")

    return gdf

def split_points_by_geojson_extents(json_file_path, points_file_path, output_dir):

    print("Reading extents")
    extent_gdf = gpd.read_file(json_file_path)
    # print("Reading points file")
    log_step("Reading points file")
    # points_gdf = gpd.read_file(points_file_path)
    points_gdf = gpd.read_file(points_file_path, engine="pyogrio") #pyogrio should be faster than fiona
    log_step("Finished reading points file")

    os.makedirs(output_dir, exist_ok=True)
    tiles = []

    for _, extent_row in tqdm(extent_gdf.iterrows(), total=extent_gdf.shape[0], desc="Processing each extend..."):
        # tile_name = extent_row['properties']['name']
        tile_name = extent_row['kaartbladNr']
        tile_geom = extent_row['geometry']

        # tile_geom = extent_row['geometry']['coordinates'][0]
        # Clip points within the extent
        clipped_points = points_gdf[points_gdf.intersects(tile_geom)]

        # Skip empty subsets
        if clipped_points.empty:
            # print(f"No points found within tile: {tile_name}. Skipping...")
            continue

        tiles.append({'geometry': tile_geom, 'name': tile_name, 'done': 0})

        # Save to a new shapefile
        output_path = os.path.join(output_dir, f"{tile_name}.gpkg")
        clipped_points.to_file(output_path)

    tiles_gdf = gpd.GeoDataFrame(tiles)
    # tiles_gdf.to_file("output/CS/mid/tiles_processing_visibility.shp", driver= "ESRI Shapefile")

        # print(f"Saved {len(clipped_points)} points to {output_path}")



#add height of embankmen to midpoint file
#filenames-------------------------------------------------------------------------------------------------------------
def cs_input(directory, step):
    cs_dir = f"{directory}/CS/lines/cs_{step}"
    os.makedirs(cs_dir, exist_ok=True)
    cs_input = f"{directory}/CS/lines/cs_{step}/cs_{step}.shp"
    return cs_input

def mid_input(directory, step):
    mid_dir = f"{directory}/CS/mid/mid_{step}"
    os.makedirs(mid_dir, exist_ok=True)

    mid_input = f"{directory}/CS/mid/mid_{step}/mid_{step}.shp"
    return mid_input

def embank_input(directory, step):
    embank_dir = f"{directory}/CS/embank/embank_{step}"
    os.makedirs(embank_dir, exist_ok=True)
    embank_input = f"{directory}/CS/embank/embank_{step}/embank_{step}.shp"
    return embank_input

def riverwidth_input(directory, step):
    riverwidth_dir = f"{directory}/CS/riverwidth/riverwidth_{step}"
    os.makedirs(riverwidth_dir, exist_ok=True)
    riverwidth_input = f"{directory}/CS/riverwidth/riverwidth_{step}/riverwidth_{step}.shp"
    return riverwidth_input


def log_step(step):
    """
    Logs the given step with the current timestamp.

    Args:
        step (str): A description of the step to log.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Step: {step}")

if __name__ == '__main__':
    """
    First the river data is retrieved from OSM and processed in QGIS
    1. Fetch all waterways tagged as river and canal
    2. Process in QGIS to obtain rivers in urban areas
    """
    # All waterways defined as rivers and canals in the Netherlands we're fetched from OSM
    # gdf_popdens = gpd.read_file("input/urban_areas/1000/popdenshigehrthan1000raster.shp")
    # rivers_total_shp = "input/RIVERS/OSM_rivers.shp"
    # fetch_waterways_in_areas(gdf_popdens, rivers_total_shp)

    """
    Here all the file paths are defined for input and output
    """
    # This shapefile was clipped in QGIS to only contain the rivers in urban areas
    # This is the shapefile containing the areas with a population density higher than 1000 in the Netherlands
    grid_shp = "input/urban_areas/1000/popdenshigehrthan1000raster.shp"
    # This is the shapefile defining non-urban areas in the Netherlands (opposite of the urban areas file)
    non_urban_areas = "input/urban_areas/non_urban_areas_total_gridded.shp"

    # this file contains all the rivers that fall into urban areas and contains no duplicates
    # rivers_nodupl_shp = "input/RIVERS/OSM_rivers_clipped_filtered.shp"
    # This file is all the rivers that fall into urban areas
    # rivers_urban_shp = "input/RIVERS/OSM_rivers_clipped.shp"
    # this file contains all those rivers that fall into urban areas, contains no duplicates, and are not in waterplains, and lay in waterdelen
    # rivers_dissolved_shp = "input/RIVERS/OSM_rivers_clipped_filtered_nowatervlaktes_noNonWaterdeel_DISSOLVED.shp"
    # rivers_shp = "input/RIVERS/OSM_rivers_clipped_filtered_nowatervlaktes_noNonWaterdeel.shp"
    rivers_shp = "input/RIVERS/OSM_rivers_clipped_filtered_nowatervlaktes_noNonWaterdeel_dissolved_split_uniqueID.shp"

    # These are the bgt bridge ('overbruggingsdeel') polygons in the urban areas
    bridges_shp = "input/DATA/BGT/bgt_overbrugginsdeel_clipped_extractedbylocation.shp"



    #WATERDELEN
    # This file contains the waterdelen after preparing the data
    # waterdelen_pre_1 = "input/DATA/BGT_rivers/waterdelen_popdens_1000_objectEindisNULL.shp"
    # waterdelen_pre_2 = "input/DATA/BGT_rivers/waterdelen_popdens_1000_objectEindisNULL_valid.shp"
    # waterdelen_pre_3 = "input/DATA/BGT_rivers/waterdelen_popdens_1000_objectEindisNULL_valid_dissolved.shp"
    # waterdelen_pre_4 = "input/DATA/BGT_rivers/waterdelen_popdens_1000_objectEindisNULL_valid_dissolved_from_multipart_to_singlepart.shp"
    # waterdelen_pre_5 = "input/DATA/BGT_rivers/waterdelen_popdens_1000_objectEindisNULL_valid_dissolved_from_multipart_to_singlepart_extractedbylocationRivers.shp"
    waterdelen_shp= "input/DATA/BGT_rivers/waterdelen_popdens_1000_objectEindisNULL_valid_dissolved_from_multipart_to_singlepart_extractedbylocationRivers_holesRemoved.shp"



    # output directory
    directory = "output"
    os.makedirs(directory, exist_ok=True)
    # Results of functions
    txt_file = "output/steps_info.txt"

    """
    
    RUN SCRIPT
    1. Create the initial cross-sections and midpoints of our segments.
    2. Clean the initial sections by removing sections that are bridges 
    3. Embankment points are found for each section
    4. Riverwidth file is created and some points are cleaned out
        4.1 remove embankment where riverwidth is much bigger or smaller than average of river riverwidth
        4.2 if a riverwidth crosses another riverwidth, certain conditions apply
            Preferably, if the group size is two, I want to create a new embankment point and midpoint, and replace it with this which has the mean angle ----NOT IMPLEMENTED YET----
        4.3 Remove riverwidths that cross another riverline 
    5. New sections are created 100m outwards from the embankment points
    6. Cleaning segments: Remove the sections that lay in non-urban areas
    7. Cleaning segments: Remove segments intersecting with an embankment line     
    8. Create points and split into multiple files
    9. Prepare midpoint file for visibility analysis
    
    """

    step = 1
    print(f"Step {step}")
    # cross_section_extraction(rivers_shp, 25, 600, directory, step, txt_file)
    # gives cs_shp, mid_shp


    step = 2
    print(f"Step {step}")
    # cleaning_initial_sections(1, 1, waterdelen_shp, bridges_shp, directory, step, txt_file)
    # gives cs_shp, mid_shp


    step = 3
    print(f"Step {step}")
    # find_embankment_faster(2, 2, waterdelen_shp, directory, txt_file, step) #10-16-25-30 minutes
    #gives cs_shp, embank_shp


    step = 4
    print(f"Step {step}")
    # create_and_clean_riverwidth(3, 2, rivers_shp, directory, step , txt_file)
    #gives riveriwdth_shp, embank_shp


    step  = 5
    print(f"Step {step}")
    # section_extraction_final(3, 4, 2, directory, step, txt_file) # 4-5 minutes
    #gives cs_shp


    print("Step 6")
    step = 6
    # clean_cross_sections_non_urban(5, 4, non_urban_areas, directory, step, txt_file) # 14 minutes
    # gives cs_shp, embank_shp


    print("Step 7")
    step = 7

    # embank_line_gdf = embankment_line_old(6, rivers_shp, directory)
    embankline_in_shp = "output/CS/embank/embank_line/embank_line_0.shp"
    # embank_line_gdf.to_file(embankline_in_shp, driver="ESRI Shapefile")

    embankline_out_shp = "output/CS/embank/embank_line/embank_line_1.shp"
    # clean_riverwidth_and_segment_crossing_embankmentline(rivers_shp, 4, 6, 6, embankline_in_shp, embankline_out_shp, directory, step, txt_file)
    # gives cs_shp, embank_shp, rvwidth_shp


    print("Step 8")
    points_geometry_gpkg = "output/PTS/pts.gpkg"
    points_attributes_csv = "output/PTS/attributes.csv"
    cs_in = cs_input(directory, 7)

    # create_cross_section_points(cs_in, 100, points_geometry_gpkg, points_attributes_csv)

    json_file_path = "input/DATA/AHN/kaartbladindex_AHN_DSM.json"
    # points_shp = "output/PTS/pts_dtm.shp"
    split_output_folder = "output/PTS/split"
    os.makedirs(split_output_folder, exist_ok=True)

    split_points_by_geojson_extents(json_file_path, points_geometry_gpkg, split_output_folder)


    print("Step 9")
    step = 9
    #First, add dtm data to embankment points --> from embank_7 to embank_dtm
    # add_width_to_mid(7, 2, directory, step) # todo: add height value to midpoint
    # gives mid_shp

    # split_midpoints = "output/CS/mid/split"
    # mid_in = mid_input(directory, 9)
    # split_points_by_geojson_extents(json_file_path,mid_in, split_midpoints )
