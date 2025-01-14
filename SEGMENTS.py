import os

from shapely import MultiLineString, MultiPoint, GeometryCollection
from tqdm import tqdm
import overpy
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
import geopandas as gpd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# River data retrieval
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

#cs preprocess
def cross_section_extraction(river_file, interval, width, output_file_cs, output_river_mid):
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
    #this takes 11 seconds
    for index, row in tqdm(projected_gdf.iterrows(), total=projected_gdf.shape[0], desc="Processing each river"):
        riverline = row['geometry']
        # fid is an unique identifier OSM gives when retrieving the data. We use it here as the river id
        river_fid = row['fid']

        # print("index, riverline length: ", index, riverline.length)
        # distance = 0
        total_length += riverline.length

        for distance_along_line in np.arange(0, riverline.length, interval):
            side_0, side_1, point_on_line = get_perpendicular_cross_section(riverline, distance_along_line, width)

            cross_sections.append({
                'geometry': side_0,
                'rv_id': river_fid,
                'cs_id': i,
                'side': 0
            })
            cross_sections.append({
                'geometry': side_1,
                'rv_id': river_fid,
                'cs_id': i,
                'side': 1
            })

            river_points.append({'geometry': point_on_line,
                                 'rv_id': river_fid,
                                'cs_id': i})
            i += 1
            # distance = distance_along_line

        # starting_point = interval - (riverline.length - distance)

    print("Saving files...")
    # Save cross-sections to a Shapefile (.shp)
    gdf = gpd.GeoDataFrame(cross_sections)
    gdf.set_crs(epsg=28992, inplace=True)
    gdf.to_file(output_file_cs, driver='ESRI Shapefile')

    # Save river midpoints to shapefile (.shp)
    river_points_gdf = gpd.GeoDataFrame(river_points)
    river_points_gdf.set_crs(epsg=28992, inplace=True)
    river_points_gdf.to_file(output_river_mid, driver='ESRI Shapefile')
    print(f"Total length river {total_length} and sections amount is {i}")

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


def clean_bridges(cs_clean_shp, cs_clean_nobridge_shp, mid_shp, overbruggingsdeel_shp):
    print("Reading files...")
    gdf_cs = gpd.read_file(cs_clean_shp)
    gdf_mid = gpd.read_file(mid_shp)
    gdf_bridges = gpd.read_file(overbruggingsdeel_shp)

    # Ensure crs
    gdf_cs = gdf_cs.to_crs(epsg=28992)
    gdf_mid = gdf_mid.to_crs(epsg=28992)
    gdf_bridges = gdf_bridges.to_crs(epsg=28992)

    bridge_points = []
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
                pt_geom = point['geometry']
                pt_id = point['cs_id']
                # print(f"point geometry is {point['geometry']} and id of river is is {point['rv_id']}")
                bridge_points.append({'geometry': pt_geom, 'cs_id': pt_id})

                # Midpoint is marked as bridge
                gdf_mid.loc[gdf_mid['cs_id']== pt_id, 'bridge'] = 1



    print("Start removing bridges from the cs shapefile...")
    # Remove rows in gdf_cs where 'cs_id' matches
    bridge_cs_ids = gdf_mid[gdf_mid['bridge'] == 1]['cs_id'].unique()
    gdf_cs_cleaned = gdf_cs[~gdf_cs['cs_id'].isin(bridge_cs_ids)]

    # Save the cleaned GeoDataFrame
    print("Saving cleaned section file...")
    gdf_cs_cleaned.to_file(cs_clean_nobridge_shp, driver="ESRI Shapefile")
    print("Saving midpoints file...")
    gdf_mid.to_file(mid_shp, driver= "ESRI Shapefile")
    print(f"Amount of midpoints identified as bridges is {gdf_mid[gdf_mid['bridge'] == 1].shape[0]}") # 1271

# def remove_midpoints_not_in_water(waterdelen_shp,mid_in, mid_out, tolerance=0.01):
#     # set tolerance as 0.01 = 1cm. Could probably be even smaller but this works.
#     print("Reading files...")
#     water_gdf = gpd.read_file(waterdelen_shp)
#     mid_gdf = gpd.read_file(mid_in)
#
#     # Pre-compute water boundaries with tolerance forprocessing speed (i used to compute boundary buffer repeatedly)
#     print("Pre-computing water boundaries...")
#     water_gdf['boundary_with_tolerance'] = water_gdf.geometry.boundary.buffer(tolerance)
#
#     # Spatial join (faster than spatial index lookup that I did before)
#     print("Performing spatial join...")
#     # First join points with water bodies
#     joined = gpd.sjoin(mid_gdf, water_gdf, predicate='within')
#
#     # Set default all points keep
#     kept_points = set() #set lookup is much faster than list lookup)
#     print("Processing points in water...")
#     # this takes
#     for idx in tqdm(joined.index.unique()):
#         point = mid_gdf.loc[idx]
#         water_matches = water_gdf.loc[joined.loc[joined.index == idx, 'index_right']]
#
#         for _, water in water_matches.iterrows():
#             if point.geometry.intersects(water['boundary_with_tolerance']):
#                 kept_points.add(point['cs_id'])
#                 break
#
#
#
#     # Filter cross-sections using bulk operations
#     print("Filtering data...")
#     mid_to_keep = mid_gdf.apply(lambda row: (row['cs_id']) in kept_points, axis=1)
#
#     filtered_mid = mid_gdf[mid_to_keep]
#
#     print(f"Deleted points: {len(mid_gdf) - len(filtered_mid) }")
#
#     print("Saving to file...")
#     filtered_mid.to_file(mid_out)
#
#     return


# Find embankment
# def add_grid_id_to_cs_and_mid(cs_shp, grid_shp, mid_shp):
#     # make sure grid_shp is NOT the buffered grid
#     print("Reading files...")
#     cs_gdf = gpd.read_file(cs_shp)
#     grid_gdf = gpd.read_file(grid_shp)
#     mid_gdf = gpd.read_file(mid_shp)
#
#     if cs_gdf.crs != grid_gdf.crs:
#         grid_gdf = grid_gdf.to_crs(cs_gdf.crs)
#     print("Joining dataframes..")
#     joined_cs_gdf = gpd.sjoin(cs_gdf, grid_gdf[['gid', 'geometry']], how='left', predicate='intersects')
#     joined_mid_gdf = gpd.sjoin(mid_gdf, grid_gdf[['gid', 'geometry']], how='left', predicate='intersects')
#     # print(joined_gdf[['gid']].head())
#
#     print("Saving files...")
#     joined_cs_gdf.to_file(cs_shp, driver="ESRI Shapefile")
#     joined_mid_gdf.to_file(mid_shp, driver="ESRI Shapefile")

def remove_midpoints_not_in_water(waterdelen_shp, mid_in, mid_out, tolerance=0.01):
    """
    Sometimes a midpoint on the OSM river falls on land/structure instead of in 'waterdelen'. We remove these as no embankment will be able to be found.
    Args:
        waterdelen_shp (str): Path to the water polygons shapefile
        mid_in (str): Path to input midpoint shapefile
        mid_out (str): Path to output midpoint shapefile
        tolerance (float): Tolerance value for intersection computation, set at 1cm

    Returns:

    """
    print("Reading files...")
    water_gdf = gpd.read_file(waterdelen_shp)
    mid_gdf = gpd.read_file(mid_in)

    # Pre-compute water boundaries with tolerance
    print("Pre-computing water boundaries...")
    water_gdf['boundary_with_tolerance'] = water_gdf.geometry.boundary.buffer(tolerance)

    # Build spatial index for midpoints
    print("Building spatial index...")
    mid_sindex = mid_gdf.sindex

    # Set to track points to keep
    removed_points = set()
    print("Processing water polygons...")

    # Iterate over each water polygon and find points within its buffered boundary
    # this took 2:52 minutes
    for _, water in tqdm(water_gdf.iterrows(), total=water_gdf.shape[0]):
        # Get points within the bounding box of the buffered water boundary
        possible_matches_index = list(mid_sindex.intersection(water['boundary_with_tolerance'].bounds))
        possible_matches = mid_gdf.iloc[possible_matches_index]

        # Check for actual intersection with the buffered boundary
        for idx, point in possible_matches.iterrows():
            if point.geometry.intersects(water['boundary_with_tolerance']):
                removed_points.add(point['cs_id'])

    # Filter cross-sections using bulk operations
    print("Filtering data...")
    mid_to_keep = ~mid_gdf['cs_id'].isin(removed_points)
    filtered_mid = mid_gdf[mid_to_keep]

    print(f"Deleted points: {len(mid_gdf) - len(filtered_mid)}") # is 952

    print("Saving to file...")
    filtered_mid.to_file(mid_out)

    return

#embankment point computation
def find_embankment(cs_shp_in, cs_shp_out, mid_shp, waterdelen_shp, embank_shp):
    """
    Function to find the embankment point of each segment.
    Args:
        cs_shp_in (str): Path to input sections shapefile
        cs_shp_out(str): Path to output sections shapefile, removing sections that do not have an embankment point
        mid_shp (str): Path to input midpoint shapefile
        waterdelen_shp (str): Path to the water polygons shapefile
        embank_shp (str): Path to the output file for embankment points

    Returns:

    """
    print("Reading files...")
    gdf_cs = gpd.read_file(cs_shp_in)
    gdf_mid = gpd.read_file(mid_shp)
    gdf_water = gpd.read_file(waterdelen_shp)

    # Create spatial index for water features
    water_sindex = gdf_water.sindex

    to_drop = set()
    embankment_results = {}  # Key: (cs_id, side), Value: (intersection_point, distance)

    # Pre-compute water boundaries
    print("Computing water boundaries...")
    # water_boundaries = [geom.boundary for geom in gdf_water.geometry]

    water_boundaries = gpd.GeoDataFrame(
        geometry=[geom.boundary for geom in gdf_water.geometry],
        crs=gdf_water.crs
    )

    # Create sindex on these boundaries
    water_boundary_sindex = water_boundaries.sindex


    # Iterate through cross sections
    for _, cs_row in tqdm(gdf_cs.iterrows(), total=len(gdf_cs), desc="Processing cross sections"): #takes 26 minutes
        cs_id = cs_row['cs_id']
        side = cs_row['side']
        section_geom = cs_row['geometry']
        unique_id = (cs_id, side)

        # Get corresponding midpoint
        # midpoint_geom = gdf_mid[gdf_mid['cs_id'] == cs_id]['geometry']
        mid_mask = gdf_mid['cs_id'] == cs_id
        if not any(mid_mask):
            to_drop.add(unique_id)
            continue
        midpoint_geom = gdf_mid[mid_mask].geometry.iloc[0]

        # Find potential water features that intersect with the section's bounding box
        possible_matches_idx = list(water_boundary_sindex.intersection(section_geom.bounds))
        if not possible_matches_idx:
            to_drop.add(unique_id)
            continue

        # Find intersections with water features boundaries
        intersection_points = []
        for idx in possible_matches_idx:
            try:
                # print(f"Index being used: {idx}")
                # print(f"Water boundaries index: {water_boundaries.index}")
                water_boundary = water_boundaries.iloc[idx].geometry
                # print(f"1. Got water boundary type: {type(water_boundary)}")
                # print(f"2. Section geom type: {type(section_geom)}")

                intersects = section_geom.intersects(water_boundary)
                # print(f"3. Intersects type: {type(intersects)}")
                if section_geom.intersects(water_boundary):
                    intersection = section_geom.intersection(water_boundary)

                    if intersection.is_empty:
                        continue

                    # Convert intersection to point(s)
                    if isinstance(intersection, LineString):
                        intersection_points.append(Point(intersection.coords[0]))
                    elif isinstance(intersection, Point):
                        intersection_points.append(intersection)
                    elif isinstance(intersection, MultiPoint):
                        intersection_points.extend([Point(p.coords[0]) for p in intersection.geoms])
                    elif isinstance(intersection, MultiLineString):
                        intersection_points.extend([Point(line.coords[0]) for line in intersection.geoms])
                    elif isinstance(intersection, GeometryCollection):
                        for geom in intersection.geoms:
                            if isinstance(geom, Point):
                                intersection_points.append(geom)
                            elif isinstance(geom, LineString):
                                intersection_points.append(Point(geom.coords[0]))


            except Exception as e:
                print(f"Error processing intersection for cs_id {cs_id}, side {side}: {e}")
                continue

        if not intersection_points:
            to_drop.add(unique_id)
            continue

        # Find closest intersection point
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
            'mid_dist': distance
        })

    # Removing cs that have no embankment point
    mask = [(row['cs_id'], row['side']) not in to_drop for _, row in gdf_cs.iterrows()]
    gdf_cs = gdf_cs[mask]

    print("Saving files...")
    gdf_cs.to_file(cs_shp_out)

    gdf_embank = gpd.GeoDataFrame(embankments, geometry='embank')
    gdf_embank.set_crs(epsg=28992, inplace=True)
    gdf_embank.to_file(embank_shp)


def find_embankment_faster(cs_shp, cs_shp_cleaned_embank, mid_shp, dissolved_waterdelen_shp, embank_shp):
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

    print("Processing water polygons...")
    for idx, water_row in tqdm(gdf_water.iterrows(), total=len(gdf_water), desc="Processing water features"):
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

    print("Finding closest intersections...")
    embankment_results = {}  # Key: (cs_id, side), Value: (intersection_point, distance)

    # Process all intersections to find the closest point for each cross-section
    for unique_id, intersection_points in tqdm(all_intersections.items(), desc="Finding closest points"):
        cs_id, side = unique_id
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
            'mid_dist': distance
        })

    # Keep only cross-sections with embankment points
    mask = [(row['cs_id'], row['side']) in embankment_results for _, row in gdf_cs.iterrows()]
    gdf_cs = gdf_cs[mask]

    print("Saving files...")
    gdf_cs.to_file(cs_shp_cleaned_embank)

    gdf_embank = gpd.GeoDataFrame(embankments, geometry='embank')
    gdf_embank.set_crs(epsg=28992, inplace=True)
    gdf_embank.to_file(embank_shp)

# def find_embankment(cs_shp, cs_shp_cleaned_embank, mid_shp, dissolved_waterdelen_shp,  embank_shp):
#     print("Reading files...")
#     gdf_cs = gpd.read_file(cs_shp)
#     gdf_mid = gpd.read_file(mid_shp)
#     gdf_water = gpd.read_file(dissolved_waterdelen_shp) #these are seperated into buffered grid cells
#
#     to_drop = set()
#     embankment_results = {}  # Key: cs_id, Value: (intersection_point, distance, grid_id)
#
#     grouped_mid = gdf_mid.groupby('gid')
#
#     for gid, group_mid in tqdm(grouped_mid, desc="Processing grid cells"):
#         # Get the water features for this grid cell
#         grid_water = gdf_water[gdf_water['gid'] == gid]
#
#         if grid_water.empty:
#             continue
#
#         # Process each section in this grid
#         for index, midpoint in group_mid.iterrows():
#             midpoint_geom = midpoint['geometry']
#             cs_id = midpoint['cs_id']
#
#             sides = [0, 1]
#             for side in sides:
#                 # Create unique identifier tuple
#                 unique_id = (cs_id, side)
#                 cs_row = gdf_cs[(gdf_cs['cs_id'] == cs_id) & (gdf_cs['side'] == side)]
#                 if cs_row.empty:
#                     continue
#
#                 section_geom = cs_row['geometry'].iloc[0]
#
#                 # Find intersections with water features boundaries in this grid
#                 intersection_points = []
#                 for _, water_feature in grid_water.iterrows():
#                     try:
#                         # Get the boundary of the water polygon
#                         water_boundary = water_feature.geometry.boundary
#
#                         # Check intersection with boundary
#                         if section_geom.intersects(water_boundary):
#                             intersection = section_geom.intersection(water_boundary)
#
#                             # Convert intersection to point(s)
#                             if isinstance(intersection, LineString):
#                                 intersection_point = Point(intersection.coords[0])
#                                 intersection_points.append(intersection_point)
#                             elif isinstance(intersection, Point):
#                                 intersection_points.append(intersection)
#                             elif isinstance(intersection, MultiPoint):
#                                 # Add all points from multipoint
#                                 intersection_points.extend([Point(p.coords[0]) for p in intersection.geoms])
#                             elif isinstance(intersection, MultiLineString):
#                                 # Add first point of each linestring
#                                 intersection_points.extend([Point(line.coords[0]) for line in intersection.geoms])
#                             elif isinstance(intersection, GeometryCollection):
#                                 # Handle each geometry in the collection
#                                 for geom in intersection.geoms:
#                                     if isinstance(geom, Point):
#                                         intersection_points.append(geom)
#                                     elif isinstance(geom, LineString):
#                                         intersection_points.append(Point(geom.coords[0]))
#                             else:
#                                 print(f"Unexpected intersection type: {type(intersection)}")
#                                 continue
#
#                     except Exception as e:
#                         print(f"Error processing intersection for cs_id {cs_id}, side {side}: {e}")
#
#                 if not intersection_points:
#                     to_drop.add(unique_id)
#                     continue
#
#                 # Find closest intersection point
#                 closest_intersection = min(intersection_points, key=lambda x: midpoint_geom.distance(x))
#                 mid_intersection_distance = midpoint_geom.distance(closest_intersection)
#                 embankment_results[unique_id] = (closest_intersection, mid_intersection_distance, gid)
#
#
#     print("Making the embankment file...")
#     embankments = []
#     for (cs_id, side), (intersection_point, distance, gid) in embankment_results.items():
#         # Get original section data (take first occurrence matching both cs_id and side)
#         section_data = gdf_cs[(gdf_cs['cs_id'] == cs_id) & (gdf_cs['side'] == side)].iloc[0]
#         embankments.append({
#             'cs_id': cs_id,
#             'rv_id': section_data['rv_id'],
#             'side': side,
#             'embank': intersection_point,
#             'mid_dist': distance,
#             'gid': gid
#         })
#     # Removing cs that have no embankment point
#     mask = [(row['cs_id'], row['side']) not in to_drop for _, row in gdf_cs.iterrows()]
#     gdf_cs = gdf_cs[mask]
#
#     print("Saving files...")
#     gdf_cs.to_file(cs_shp_cleaned_embank)
#
#     gdf_embank = gpd.GeoDataFrame(embankments, geometry='embank')
#     gdf_embank.set_crs(epsg=28992, inplace=True)
#     gdf_embank.to_file(embank_shp, driver="ESRI Shapefile")



# Finale sections

def remove_embankments_in_water(waterdelen_shp, input_embank_shp, output_embank_shp, input_cs_shp, output_cs_shp,  tolerance=0.01):
    # set tolerance as 0.01 = 1cm. Could probably be even smaller but this works.
    # THIS DELETED NOTHING ON 130125
    print("Reading files...")
    embank_gdf = gpd.read_file(input_embank_shp)
    water_gdf = gpd.read_file(waterdelen_shp)
    cs_gdf = gpd.read_file(input_cs_shp)

    # Pre-compute water boundaries with tolerance forprocessing speed (i used to compute boundary buffer repeatedly)
    print("Pre-computing water boundaries...")
    water_gdf['boundary_with_tolerance'] = water_gdf.geometry.boundary.buffer(tolerance)

    # Spatial join (faster than spatial index lookup that I did before)
    print("Performing spatial join...")
    # First join points with water bodies
    joined = gpd.sjoin(embank_gdf, water_gdf, predicate='within')

    # Set default all points keep
    removed_points = set() #set lookup is much faster than list lookup)
    print("Processing points in water...")
    for idx in tqdm(joined.index.unique()):
        point = embank_gdf.loc[idx]
        water_matches = water_gdf.loc[joined.loc[joined.index == idx, 'index_right']]

        for _, water in water_matches.iterrows():
            if not point.geometry.intersects(water['boundary_with_tolerance']):
                removed_points.add((point['cs_id'], point['side']))
                break



    # Filter cross-sections using bulk operations
    print("Filtering data...")
    cs_to_keep = ~cs_gdf.apply(lambda row: (row['cs_id'], row['side']) in removed_points, axis=1)
    embank_to_keep = ~embank_gdf.apply(lambda row: (row['cs_id'], row['side']) in removed_points, axis=1)

    filtered_cs = cs_gdf[cs_to_keep]
    filtered_embank = embank_gdf[embank_to_keep]

    print(f"Deleted points: {len(embank_gdf) - len(filtered_embank) }")
    print(f"Deleted segments cross-sections: {len(cs_gdf)- len(filtered_cs)}")

    print("Saving to file...")
    filtered_embank.to_file(output_embank_shp)
    filtered_cs.to_file(output_cs_shp)

    return


#Final section extraction
def section_extraction_final(input_cs_shp, embank_shp, output_file_cs, mid_shp):
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
    print("Reading files...")
    cs_gdf = gpd.read_file(input_cs_shp)
    embank_gdf = gpd.read_file(embank_shp)
    midpoint_gdf = gpd.read_file(mid_shp)

    cross_sections = []

    for idx, row in tqdm(cs_gdf.iterrows(), total=cs_gdf.shape[0], desc="Creating new sections..."):
        cs_id = row['cs_id']
        side = row['side']
        rv_id = row['rv_id']
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
                'side': side
            })

    print("Saving files...")
    gdf = gpd.GeoDataFrame(cross_sections)
    gdf.set_crs(epsg=28992, inplace=True)
    gdf.to_file(output_file_cs, driver='ESRI Shapefile')


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



#cleaning
def clean_cross_sections_non_urban(cs_shp, non_urban_area_shp, cs_output_file, embank_input, embank_output):
    print("Reading files...")
    gdf_cs = gpd.read_file(cs_shp)
    gdf_non_urban_areas = gpd.read_file(non_urban_area_shp)
    gdf_embank = gpd.read_file(embank_input)

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

    gdf_cs.set_crs(epsg=28992, inplace=True)
    gdf_embank.set_crs(epsg=28992, inplace=True)
    gdf_cs.to_file(cs_output_file, driver= "ESRI Shapefile")
    gdf_embank.to_file(embank_output, driver="ESRI Shapefile")


def create_river_width_file(embank_shp, mid_shp,  river_width_output):
    print("Reading files...")
    embank_gdf = gpd.read_file(embank_shp)
    mid_gdf = gpd.read_file(mid_shp)

    grouped = embank_gdf.groupby('cs_id')
    river_width_list = []

    for idx, group in tqdm(grouped, total=len(grouped),
                       desc="Processing each section, making river width"):
        if len(group) == 2:
            row_0 = group.iloc[0]
            geom_0, cs_id, rv_id, side_0, mid_dist_0 = row_0['geometry'], row_0['cs_id'], row_0['rv_id'], row_0['side'], row_0['mid_dist']
            row_1 = group.iloc[1]
            geom_1, side_1, mid_dist_1 = row_1['geometry'], row_1['side'], row_1['mid_dist']

            mid_row = mid_gdf[mid_gdf['cs_id'] == cs_id]
            mid_geom = mid_row['geometry'].iloc[0]

            river_width = mid_dist_0 + mid_dist_1
            river_geom_0 = LineString([mid_geom, geom_0.coords[0]])
            river_geom_1 = LineString([mid_geom, geom_1.coords[0]])

            river_width_list.append({'geometry': river_geom_0,'cs_id':cs_id, 'rv_id':rv_id, 'side':side_0, 'mid_dist': mid_dist_0})
            river_width_list.append({'geometry': river_geom_1,'cs_id':cs_id, 'rv_id':rv_id, 'side':side_1, 'mid_dist': mid_dist_1})
        elif len(group == 1):
            row = group.iloc[0]
            geom, cs_id, rv_id, side, mid_dist = row['geometry'], row['cs_id'], row['rv_id'], row['side'], row['mid_dist']

            mid_row = mid_gdf[mid_gdf['cs_id'] == cs_id]
            mid_geom = mid_row['geometry'].iloc[0]

            river_geom = LineString([mid_geom, geom.coords[0]])

            river_width_list.append({'geometry': river_geom, 'cs_id': cs_id,'rv_id':rv_id, 'side': side, 'mid_dist': mid_dist})
        else:
            print(f"Skipping cs_id {idx}: Expected 1 or 2 rows, found {len(group)}")
            continue

    river_width_gdf = gpd.GeoDataFrame(river_width_list)
    river_width_gdf.set_crs(epsg=28992, inplace=True)
    river_width_gdf.to_file(river_width_output)


def clean_cross_sections_crossing_rivers(rivers_shp, rv_width, rv_width_out, cs_shp_input, cs_shp_output, embank_in, embank_out):
    print("Reading files...")
    rivers_gdf = gpd.read_file(rivers_shp)
    rvwidth_gdf = gpd.read_file(rv_width)
    cs_gdf = gpd.read_file(cs_shp_input)
    embank_gdf = gpd.read_file(embank_in)

    print("Creating spatial index for sections and riverwidths...")
    cs_sindex = cs_gdf.sindex
    rvwidth_sindex = rvwidth_gdf.sindex
    indices_to_drop = set()
    indices_to_drop_rv_width = set()

    for index, row in tqdm(rivers_gdf.iterrows(), total=rivers_gdf.shape[0],
                           desc="Processing each river, checking if a cs or rvwidth intersects with it"):
        line_geom = row['geometry']
        rv_id = row['fid']

        possible_matches_index = list(cs_sindex.intersection(line_geom.bounds))
        possible_matches = cs_gdf.iloc[possible_matches_index]

        intersections = possible_matches[possible_matches.intersects(line_geom)]
        indices_to_drop.update(intersections.index)

        possible_matches_rvwidth_index = list(rvwidth_sindex.intersection(line_geom.bounds))
        possible_matches_rvwidth = rvwidth_gdf.iloc[possible_matches_rvwidth_index]

        intersections_rvwidth = possible_matches_rvwidth[possible_matches_rvwidth.intersects(line_geom)]
        filtered_intersections_rvwidth = intersections_rvwidth[intersections_rvwidth['rv_id'] != rv_id]
        indices_to_drop_rv_width.update(filtered_intersections_rvwidth.index)

    # Collecting identifiers of dropped river widths
    dropped_rvwidth_rows = rvwidth_gdf.loc[list(indices_to_drop_rv_width), ['cs_id', 'side']]
    dropped_cs_conditions = set(zip(dropped_rvwidth_rows['cs_id'], dropped_rvwidth_rows['side']))
    #combine all that need to be removed
    combined_indices_to_drop = indices_to_drop | {idx for idx, row in cs_gdf.iterrows() if
                                                  (row['cs_id'], row['side']) in dropped_cs_conditions}

    # Remove those indices from cs_gdf
    cs_gdf_cleaned = cs_gdf.drop(list(combined_indices_to_drop))
    # Removing items from rv_width
    rvwidth_gdf = rvwidth_gdf.drop(indices_to_drop_rv_width)
    #Removing items from embankment
    embank_gdf = embank_gdf[
        ~embank_gdf.apply(lambda row: (row['cs_id'], row['side']) in dropped_cs_conditions, axis=1)]

    print("Saving files...")
    cs_gdf_cleaned.to_file(cs_shp_output, driver="ESRI Shapefile")
    rvwidth_gdf.to_file(rv_width_out, driver="ESRI Shapefile")
    embank_gdf.to_file(embank_out, driver="ESRI Shapefile")

def clean_cross_sections_riverwidth_crossing_riverwidth(riverwidth_input, rvwidth_output, cs_shp_input, cs_shp_output, embank_input, embank_output):
    #todo: for some reason it doesnt delete all the riverwidths and segments

    print("Reading files...")
    rvwidth_gdf = gpd.read_file(riverwidth_input)
    cs_gdf = gpd.read_file(cs_shp_input)
    embank_gdf = gpd.read_file(embank_input)

    rvwidth_sindex = rvwidth_gdf.sindex

    indices_to_drop = set()

    for index, riverwidth in tqdm(rvwidth_gdf.iterrows(), total=rvwidth_gdf.shape[0], desc="Processing each riverwidth"):
        if index in indices_to_drop:
            continue

        line_geom = riverwidth['geometry']
        cs_id = riverwidth['cs_id']
        # side = riverwidth['side']

        possible_matches_index = list(rvwidth_sindex.intersection(line_geom.bounds))
        possible_matches = rvwidth_gdf.iloc[possible_matches_index]
        # dont check for current cross section
        mask = (possible_matches.index != index) & (possible_matches['cs_id'] != cs_id)
        possible_matches = possible_matches[mask]

        intersections = possible_matches[possible_matches.intersects(line_geom)]

        if not intersections.empty:
            indices_to_drop.update(intersections.index)

    # Removing items from rv_width
    rvwidth_gdf_cleaned = rvwidth_gdf.drop(indices_to_drop)

    # Get the dropped rows information
    dropped_rvwidth = rvwidth_gdf.loc[list(indices_to_drop)]
    dropped_cs_conditions = set(zip(dropped_rvwidth['cs_id'], dropped_rvwidth['side']))

    # Filter cs_gdf and embank_gdf
    cs_gdf_cleaned = cs_gdf[~cs_gdf.apply(lambda row: (row['cs_id'], row['side']) in dropped_cs_conditions, axis=1)]
    embank_gdf_cleaned = embank_gdf[
        ~embank_gdf.apply(lambda row: (row['cs_id'], row['side']) in dropped_cs_conditions, axis=1)]


    print("Saving files...")
    cs_gdf_cleaned.to_file(cs_shp_output, driver="ESRI Shapefile")
    rvwidth_gdf_cleaned.to_file(rvwidth_output, driver="ESRI Shapefile")
    embank_gdf_cleaned.to_file(embank_output, driver="ESRI Shapefile")


def clean_midpoints_not_in_water(mid_shp, mid_out, cs_shp, cs_out, embank_shp, embank_out, rvwidth_shp, rvwidth_out, water_shp):
    #NOTDONE do it via 'gid'
    print("Reading files...")
    mid_gdf = gpd.read_file(mid_shp)
    water_gdf = gpd.read_file(water_shp)
    rvwidth_gdf = gpd.read_file(rvwidth_shp)
    cs_gdf = gpd.read_file(cs_shp)
    embank_gdf = gpd.read_file(embank_shp)

    mid_sindex = mid_gdf.sindex

    for index, row in tqdm(water_gdf.iterrows(), total=water_gdf.shape[0],
                           desc="Processing each water.."):
        water_geom = row['geometry']


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


def embankment_line(embank_shp, embank_line_shp, rivers_shp):
    print("Reading files...")
    embank_gdf = gpd.read_file(embank_shp)
    rivers_gdf = gpd.read_file(rivers_shp)

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
    lines_gdf.set_crs("EPSG:28992")
    lines_gdf.to_file(embank_line_shp, driver="ESRI Shapefile")

#clean segments that cross another embankment
def clean_riverwidth_crossing_embankmentline(rivers_shp, riverwidth_shp, riverwidth_out, embankline_in, embankline_out, cs_shp, cs_out, embank_in, embank_out):
    print("Reading files...")
    rvwidth_gdf = gpd.read_file(riverwidth_shp)
    embankline_gdf = gpd.read_file(embankline_in)
    cs_gdf = gpd.read_file(cs_shp)
    embankpts_gdf = gpd.read_file(embank_in)

    rvwidth_sindex = rvwidth_gdf.sindex
    cs_sindex = cs_gdf.sindex
    embankline_sindex = embankline_gdf.sindex
    indices_to_drop_rv = set()
    indices_to_drop_cs = set()

    for idx, embankline in tqdm(embankline_gdf.iterrows(), total= embankline_gdf.shape[0], desc="Processing each embankment line..."):

        line_geom =embankline['geometry']
        rv_id_embankline = embankline['rv_id']

        possible_matches_rvwidth_index = list(rvwidth_sindex.intersection(line_geom.bounds))
        possible_matches_rvwidth = rvwidth_gdf.iloc[possible_matches_rvwidth_index]

        intersections_rvwidth = possible_matches_rvwidth[possible_matches_rvwidth.intersects(line_geom)]

        possible_matches_cs_index = list(cs_sindex.intersection(line_geom.bounds))
        possible_matches_cs = cs_gdf.iloc[possible_matches_cs_index]

        intersections_cs = possible_matches_cs[possible_matches_cs.intersects(line_geom)]

        if not intersections_rvwidth.empty:
            for idx, intersection_rv in intersections_rvwidth.iterrows():
                rv_id = intersection_rv['rv_id']
                if rv_id != rv_id_embankline:
                    # indices_to_drop_rv.update(intersection_rv.index)
                    indices_to_drop_rv.add(intersection_rv.name)
                else:
                    continue

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
    #combine dropping information of intersections riverwidth-embankement line and segment-embankment line
    combined_indices_to_drop_cs = indices_to_drop_cs | {idx for idx, row in cs_gdf.iterrows() if
                                                  (row['cs_id'], row['side']) in dropped_rvwidth_conditions}
    cs_gdf_cleaned = cs_gdf.drop(list(combined_indices_to_drop_cs))

    #fixing embankmentline and points file
    indices_to_drop_embankpts = {idx for idx, row in embankpts_gdf.iterrows() if
                                                  (row['cs_id'], row['side']) in combined_indices_to_drop_cs}
    embank_pts_cleaned = embankpts_gdf.drop(list(indices_to_drop_embankpts))
    embank_pts_cleaned.to_file(embank_out, driver="ESRI Shapefile")
    embankment_line(embank_out, embankline_out, rivers_shp)


    print("Saving files...")
    cs_gdf_cleaned.to_file(cs_out, driver="ESRI Shapefile")
    rvwidth_gdf_cleaned.to_file(riverwidth_out, driver="ESRI Shapefile")

#points
def create_cross_section_points(cross_sections_shapefile, cs_width, output_shapefile):
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
    n_points= 2 * cs_width

    for ind, row in cross_sections.iterrows():
        print('Processing cross-section', ind)

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
                'id': ind,
                'h_distance': h_distance
            })

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(all_points)
    gdf.set_crs(epsg=28992, inplace=True)

    # Save to shapefile
    gdf.to_file(output_shapefile, driver='ESRI Shapefile')
    print(f"Base cross-section points saved to: {output_shapefile}")

    return gdf


if __name__ == '__main__':
    """
    First the river data is retrieved from OSM and processed in QGIS
    1. Fetch all waterways tagged as river and canal
    2. Clip by mask in QGIS to obtain rivers in urban areas
    """
    # All waterways defined as rivers and canals in the Netherlands we're fetched from OSM
    rivers_total_shp = "input/RIVERS/OSM_rivers.shp"
    # fetch_waterways_in_areas(gdf_popdens, rivers_total_shp)

    # This shapefile was clipped in QGIS to only contain the rivers in urban areas
    # This is the shapefile containing the areas with a population density higher than 1000 in the Netherlands
    grid_shp = "input/urban_areas/1000/popdenshigehrthan1000raster.shp"
    # This file is all the rivers that fall into urban areas
    rivers_urban_shp = "input/RIVERS/OSM_rivers_clipped.shp"
    # this file contains all the rivers that fall into urban areas and contains no duplicates
    rivers_nodupl_shp = "input/RIVERS/OSM_rivers_clipped_filtered.shp"
    # this file contains all those rivers that fall into urban areas, contains no duplicates, and are not in waterplains
    rivers_shp = "input/RIVERS/OSM_rivers_clipped_filtered_nowatervlaktes.shp"


    """
    Here all the file paths are defined for input and output
    """
    # These are the needed INPUT files
    # CLEANUP: remove bridges and remove non-urban areas
    # These are the bgt bridge ('overbruggingsdeel') polygons in the urban areas
    bridges_shp = "input/DATA/BGT/bgt_overbrugginsdeel_clipped_extractedbylocation.shp"
    bridges_midpoints_shp = "input/DATA/BGT/bridges_mid.shp"
    # This is the shapefile defining non-urban areas in the Netherlands (opposite of the urban areas file)
    non_urban_areas = "input/urban_areas/non_urban_areas_total_gridded.shp"
    # This shapefile contains the waterplain polygons per grid cell (its not dissoolved?)
    # dissolved_waterdelen_noholes_split = "input/DATA/BGT/waterdeel_FINAL_buffered.shp"
    # This file contains the waterdelen after preparing the data
    waterdelen_shp = "input/DATA/BGT/waterdeel_FINAL_frommultiparttosinglepart_extractedbylocation.shp"

    #These are the output files for the cross sections
    # LINES
    cross_section_lines_directory = "output/CS/lines"
    os.makedirs(cross_section_lines_directory, exist_ok=True)
    cs_0 = "output/CS/lines/cs_lines_pre_0.shp"
    cs_1 = "output/CS/lines/cs_lines_pre_1.shp"
    cs_2 = "output/CS/lines/cs_lines_pre_2.shp"
    cs_3 = "output/CS/lines/cs_lines_pre_3.shp"
    cs_4 = "output/CS/lines/cs_lines_pre_4.shp"
    cs_5 = "output/CS/lines/cs_lines_pre_5.shp"
    cs_6 = "output/CS/lines/cs_lines_pre_6.shp"
    cs_7 = "output/CS/lines/cs_lines_pre_7.shp"
    cs_8 = "output/CS/lines/cs_lines_pre_8.shp"
    cs_TEST = "output/CS/lines/cs_lines_TEST.shp"

    # MIDPOINTS
    cross_section_mid_directory = "output/CS/mid"
    os.makedirs(cross_section_mid_directory, exist_ok=True)
    mid_0 = "output/CS/mid/cs_mid_0.shp"
    mid_1 = "output/CS/mid/cs_mid_1.shp"
    # EMBANKMENTS
    embank_folder = "output/CS/embank"
    os.makedirs(embank_folder, exist_ok=True)
    embank_0 = "output/CS/embank/embank_0.shp"
    embank_1 = "output/CS/embank/embank_1.shp"
    embank_2 = "output/CS/embank/embank_2.shp"
    embank_3 = "output/CS/embank/embank_3.shp"
    embank_4 = "output/CS/embank/embank_4.shp"
    embank_5 = "output/CS/embank/embank_5.shp"
    embank_TEST = "output/CS/embank/embank_TEST.shp"

    embank_line_0 = "output/CS/embank/embank_line_0.shp"
    embank_line_1 = "output/CS/embank/embank_line_1.shp"
    #river width output
    rv_width_0 = "output/CS/embank/river_width_0.shp"
    rv_width_1 = "output/CS/embank/river_width_1.shp"
    rv_width_2 = "output/CS/embank/river_width_2.shp"
    rv_width_3 = "output/CS/embank/river_width_3.shp"


    # POINTS
    points_shp = "output/PTS/pts_shp"

    """
    RUN SCRIPT
    1. Create the initial cross-sections and midpoints of our segments.
    2. Clean by removing sections that are bridges.
    3. A grid ID is added to the cross-sections to speed up intersection computations with waterplain polygon
    
    4. Embankment points are found for each section
    5. New sections are created 100m outwards from the embankment points
    6. River width is added to the midpoint file as distance between embankment points 
    7. Clean the sections by removing the ones that lay in non-urban areas.
    """

    # Create initial cross sections
    # cross_section_extraction(rivers_shp, 25, 600, cs_0, mid_0 )

    # Remove the sections that lay on bridges or on land
    # clean_bridges(cs_0, cs_1, mid_0, bridges_shp)
    # remove_midpoints_not_in_water(waterdelen_shp, mid_0, mid_1)

    #This causes some extra rows, where one line can have two rows cause it lays in two grid cells
    #DONT RUN I DONT USE THIS ANYMORE
    # add_grid_id_to_cs_and_mid(cs_1, grid_shp, mid_1)

    # Find the embankment points by intersecting with water plains
    #todo: i have to change such that it doesnt use gid but just a spatial index. But this can work too...

    #embankment finding goes wrong after i changed the waterdelen file? probably has to do with gid?
    # find_embankment(cs_1, cs_2, mid_1, waterdelen_shp, embank_0)
    find_embankment_faster(cs_1, cs_TEST, mid_1, waterdelen_shp, embank_TEST)


    # Make new sections
    # section_extraction_final(cs_2, embank_0, cs_3, mid_1)

    # Remove embankment points that lay in water (why would this happen if I use a spatial index now?)
    # remove_embankments_in_water(waterdelen_shp, embank_0, embank_1, cs_3, cs_4)


    # Remove the sections that lay in non-urban areas
    # clean_cross_sections_non_urban(cs_4, non_urban_areas, cs_5, embank_1, embank_2)

    # create riverwidth file
    # create_river_width_file(embank_2, mid_1, rv_width_0)
    # remove sections where riverwidth intersects with a river
    # remove sections intersecting with river
    # clean_cross_sections_crossing_rivers(rivers_shp, rv_width_0, rv_width_1, cs_5, cs_6, embank_2, embank_3)


    #if a rivrewidth crosses another riverwidth, remove both
    # todo: this keeps 1 of them for some reason
    # clean_cross_sections_riverwidth_crossing_riverwidth(rv_width_1, rv_width_2, cs_6,
    #                                                         cs_7, embank_3, embank_4)


    # embankment_line(embank_4, embank_line_0, rivers_shp)

    # if a section crosses another embankment line, remove section
    # todo: this does not do what i want it to do. either all widths are removed or none?
    # clean_riverwidth_crossing_embankmentline(rivers_shp, rv_width_2, rv_width_3, embank_line_0, embank_line_1, cs_7, cs_8, embank_4, embank_5)

    # todo: if a segment is mostly water, remove
    # intersect with water, compute length of intersection, length / 100 <= 30

    # create_cross_section_points(cs_7, 100, points_shp)
# todo: add riverwidth to midpoint shapefile. If only one embankment, then we need to find the opposit edge of the waterdeel somehow
