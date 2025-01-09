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
    Fetches all rivers and canals within specified polygon areas using OSM Overpass API

    Parameters:
    areas_gdf (GeoDataFrame): GeoDataFrame containing polygon geometries of areas of interest
    output_file (str): Path where to save the output shapefile

    Returns:
    GeoDataFrame: Contains all waterways as LineStrings
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

#cs preprocess
def cross_section_extraction(river_file, interval, width, output_file_cs, output_river_mid):
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
    print(f"Amount of midpoints identified as bridges is {gdf_mid[gdf_mid['bridge'] == 1].shape[0]}")

# Find embankment
def add_grid_id_to_cs(cs_shp, grid_shp, cs_shp_added_gridvalue):
    cs_gdf = gpd.read_file(cs_shp)
    grid_gdf = gpd.read_file(grid_shp)
    if cs_gdf.crs != grid_gdf.crs:
        grid_gdf = grid_gdf.to_crs(cs_gdf.crs)

    joined_gdf = gpd.sjoin(cs_gdf, grid_gdf[['gid', 'geometry']], how='left', predicate='intersects')
    # print(joined_gdf[['gid']].head())

    joined_gdf.to_file(cs_shp_added_gridvalue, driver="ESRI Shapefile")

def find_embankment(cs_shp, cs_shp_cleaned_embank, mid_shp, dissolved_waterdelen_shp, grid_shp,  embank_shp):
    # todo: add in a check for if the intersection also lays on the original grid, then we do not want to add it as an embankment
    print("Reading files...")
    gdf_cs = gpd.read_file(cs_shp)
    gdf_mid = gpd.read_file(mid_shp)
    gdf_water = gpd.read_file(dissolved_waterdelen_shp)
    gdf_grid = gpd.read_file(grid_shp)

    to_drop = set()
    embankment_results = {}  # Key: cs_id, Value: (intersection_point, distance, grid_id)

    grouped_cs = gdf_cs.groupby('gid')

    for gid, group_cs in tqdm(grouped_cs, desc="Processing grid cells"):
        # Get the water features for this grid cell
        grid_water = gdf_water[gdf_water['gid'] == gid]

        if grid_water.empty:
            continue

        # Process each section in this grid
        for index, section in group_cs.iterrows():
            section_geom = section['geometry']
            cs_id = section['cs_id']
            rv_id = section['rv_id']
            side = section['side']
            # Create unique identifier tuple
            unique_id = (cs_id, side)

            # Find intersections with water features boundaries in this grid
            intersection_points = []
            for _, water_feature in grid_water.iterrows():
                try:
                    # Get the boundary of the water polygon
                    water_boundary = water_feature.geometry.boundary

                    # Check intersection with boundary
                    if section_geom.intersects(water_boundary):
                        intersection = section_geom.intersection(water_boundary)

                        # Convert intersection to point(s)
                        if isinstance(intersection, LineString):
                            intersection_point = Point(intersection.coords[0])
                            intersection_points.append(intersection_point)
                        elif isinstance(intersection, Point):
                            intersection_points.append(intersection)
                        elif isinstance(intersection, MultiPoint):
                            # Add all points from multipoint
                            intersection_points.extend([Point(p.coords[0]) for p in intersection.geoms])
                        elif isinstance(intersection, MultiLineString):
                            # Add first point of each linestring
                            intersection_points.extend([Point(line.coords[0]) for line in intersection.geoms])
                        elif isinstance(intersection, GeometryCollection):
                            # Handle each geometry in the collection
                            for geom in intersection.geoms:
                                if isinstance(geom, Point):
                                    intersection_points.append(geom)
                                elif isinstance(geom, LineString):
                                    intersection_points.append(Point(geom.coords[0]))
                        else:
                            print(f"Unexpected intersection type: {type(intersection)}")
                            continue

                except Exception as e:
                    print(f"Error processing intersection for cs_id {cs_id}, side {side}: {e}")

            if not intersection_points:
                # Only mark for dropping if we haven't found an intersection for this unique_id yet
                if unique_id not in embankment_results:
                    to_drop.add(index)
                continue

            # Get midpoint for distance calculation
            mid_row = gdf_mid[gdf_mid['cs_id'] == cs_id]
            mid_geom = mid_row.geometry.iloc[0]

            # Find closest intersection point
            closest_intersection = min(intersection_points, key=lambda x: mid_geom.distance(x))
            mid_intersection_distance = mid_geom.distance(closest_intersection)

            # Update result if this is the first finding or if it's closer than previous finding
            if unique_id not in embankment_results or mid_intersection_distance < embankment_results[unique_id][1]:
                embankment_results[unique_id] = (closest_intersection, mid_intersection_distance, gid)
                # Remove from to_drop if it was there
                to_drop.discard(index)

    print("Making the embankment file...")
    embankments = []
    for (cs_id, side), (intersection_point, distance, gid) in embankment_results.items():
        # Get original section data (take first occurrence matching both cs_id and side)
        section_data = gdf_cs[(gdf_cs['cs_id'] == cs_id) & (gdf_cs['side'] == side)].iloc[0]
        embankments.append({
            'cs_id': cs_id,
            'rv_id': section_data['rv_id'],
            'side': side,
            'embank': intersection_point,
            'mid_dist': distance,
            'gid': gid
        })
    #todo: it doesnt do the drop correctly, only when there is a single section in a cell?
    gdf_cs.drop(list(to_drop), inplace=True)

    print("Saving files...")
    gdf_cs.to_file(cs_shp_cleaned_embank)

    gdf_embank = gpd.GeoDataFrame(embankments, geometry='embank')
    gdf_embank.set_crs(epsg=28992, inplace=True)
    gdf_embank.to_file(embank_shp, driver="ESRI Shapefile")


# Final sections NOT DONE
def section_extraction_final(input_cs_shp, embank_shp, output_file_cs, mid_shp):
    print("Reading files...")
    cs_gdf = gpd.read_file(input_cs_shp)
    embank_gdf = gpd.read_file(embank_shp)
    midpoint_gdf = gpd.read_file(mid_shp)

    cross_sections = []

    for idx, row in tqdm(cs_gdf.iterrows(), total=cs_gdf.shape[0], desc="Creating new sections sections"):
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

def clean_cross_sections(cs_shp, non_urban_area_shp, output_file):
    gdf_cs = gpd.read_file(cs_shp)
    gdf_non_urban_areas = gpd.read_file(non_urban_area_shp)

    for index, row in tqdm(gdf_cs.iterrows(), total=gdf_cs.shape[0],
                       desc="Processing each section, checking if it lays in a non-urban area"):
        line_geom = row['geometry']
        intersections = gdf_non_urban_areas[gdf_non_urban_areas.intersects(line_geom)]
        if not intersections.empty:
            gdf_cs = gdf_cs.drop(index)

    gdf_cs.set_crs(epsg=28992, inplace=True)
    gdf_cs.to_file(output_file, driver= "ESRI Shapefile")


def remove_embankments_in_water(waterdelen_shp, input_embank_shp, output_embank_shp, input_cs_shp, output_cs_shp,  tolerance=0.01):
    # print("Reading files...")
    # embank_gdf = gpd.read_file(input_embank_shp)
    # water_gdf = gpd.read_file(waterdelen_shp)
    # cs_gdf = gpd.read_file(input_cs_shp)
    #
    # print("Building spatial index for water...")
    # # embank_index = embank_gdf.sindex
    # water_index = water_gdf.sindex
    #
    # points_to_keep = []
    # removed_points = []
    #
    # for idx, point in tqdm(embank_gdf.iterrows(), total=len(embank_gdf),
    #                        desc="Processing each point"):
    #
    #     bounds = point.geometry.buffer(tolerance).bounds
    #     possible_matches_idx = list(water_index.intersection(bounds))
    #
    #     if not possible_matches_idx:
    #         # If no potential matches, keep the point
    #         points_to_keep.append(True)
    #         continue
    #
    #     # Get the actual polygons that might intersect
    #     possible_matches = water_gdf.iloc[possible_matches_idx]
    #
    #     # Check if point is within any of the potential polygons
    #     keep_point = True
    #     for _, poly in possible_matches.iterrows():
    #         # Create boundary with tolerance
    #         boundary_with_tolerance = poly.geometry.boundary.buffer(tolerance)
    #
    #         # Keep point if:
    #         # 1. It's not completely within the polygon, OR
    #         # 2. It's within the tolerance distance of the boundary
    #         if (point.geometry.within(poly.geometry) and
    #                 not point.geometry.intersects(boundary_with_tolerance)):
    #             keep_point = False
    #             removed_points.append((point['cs_id'], point['side']))
    #             break
    #
    #     points_to_keep.append(keep_point)
    #
    #     # Filter the original GeoDataFrame
    # filtered_embank = embank_gdf[points_to_keep]
    #
    # print(f"Original points: {len(embank_gdf)}")
    # print(f"Filtered points: {len(filtered_embank)}")
    #
    # print("Filtering cross-sections...")
    # cs_to_keep = ~cs_gdf.apply(lambda row: (row['cs_id'], row['side']) in removed_points, axis=1)
    # filtered_cs = cs_gdf[cs_to_keep]
    # set tolerance as 0.01 = 1cm. Could probably be even smaller but this works.
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
    rivers_shp = "input/RIVERS/OSM_rivers_clipped_filtered.shp"


    """
    Here all the file paths are defined for input and output
    """
    # These are the needed INPUT files
    # CLEANUP: remove bridges and remove non-urban areas
    # These are the bgt bridge ('overbruggingsdeel') polygons in the urban areas
    bridges_shp = "input/DATA/BGT/bgt_overbrugginsdeel_clipped.shp"
    bridges_midpoints_shp = "input/DATA/BGT/bridges_mid.shp"
    # This is the shapefile defining non-urban areas in the Netherlands (opposite of the urban areas file)
    non_urban_areas = "input/urban_areas/non_urban_areas.shp"
    # This shapefile contains the waterplain polygons per grid cell
    dissolved_waterdelen_noholes_split = "input/DATA/BGT/waterdeel_FINAL_buffered.shp"

    #These are the output files for the cross sections
    # LINES
    cross_section_lines_directory = "output/CS/lines"
    os.makedirs(cross_section_lines_directory, exist_ok=True)
    # cross_sections_lines_shp = "output/CS/lines/cs_lines_pre_0.shp"
    cs_0 = "output/CS/lines/cs_lines_pre_0.shp"
    # cross_sections_lines_nobridge_shp = "output/CS/lines/cs_lines_pre_1.shp"
    cs_1 = "output/CS/lines/cs_lines_pre_1.shp"
    # cross_sections_lines_nobridge_grid_shp = "output/CS/lines/cs_lines_pre_2.shp"
    cs_2 = "output/CS/lines/cs_lines_pre_2.shp"
    # cross_sections_lines_nobridge_grid_embank_shp = "output/CS/lines/cs_lines_pre_3.shp"
    cs_3 = "output/CS/lines/cs_lines_pre_3.shp"
    # cross_sections_final_with_non_urban = "output/CS/lines/cs_lines_pre_4.shp"
    cs_4 = "output/CS/lines/cs_lines_pre_4.shp"
    # cross_sections_drop_waterdelen = "output/CS/lines/cs_lines_pre_5.shp"
    cs_5 = "output/CS/lines/cs_lines_pre_5.shp"
    # cross_sections_final_drop_non_urban = "output/CS/lines/cs_lines_final.shp"
    cs_6 = "output/CS/lines/cs_lines_final_6.shp"
    # MIDPOINTS
    cross_section_mid_directory = "output/CS/mid"
    os.makedirs(cross_section_mid_directory, exist_ok=True)
    cross_section_mid_shp = "output/CS/mid/cs_mid.shp"
    # EMBANKMENTS
    embank_folder = "output/CS/embank"
    os.makedirs(embank_folder, exist_ok=True)
    embank_shp = "output/CS/embank/embank.shp"
    embank_cleaned_shp = "output/CS/embank/embank_cleaned0.shp"
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
    # cross_section_extraction(rivers_shp, 50, 450, cross_sections_lines_shp, cross_section_mid_shp )


    # Remove the sections that lay on bridges
    # clean_bridges(cross_sections_lines_shp, cross_sections_lines_nobridge_shp, cross_section_mid_shp, bridges_shp)

    #This causes some extra rows, where one line can have two rows cause it lays in two grid cells
    # add_grid_id_to_cs( cross_sections_lines_nobridge_shp, grid_shp, cross_sections_lines_nobridge_grid_shp)

    # Find the embankment points by intersecting with water plains
    # find_embankment(cross_sections_lines_nobridge_grid_shp, cross_sections_lines_nobridge_grid_embank_shp, cross_section_mid_shp, dissolved_waterdelen_noholes_split,grid_shp, embank_shp)

    # Make new sections
    # section_extraction_final(cross_sections_lines_nobridge_grid_embank_shp, embank_shp, cross_sections_final_with_non_urban, cross_section_mid_shp)

    # todo: change order of final section extraction and embankment removal as we dont need to compute sections at those embankments that lay in water
    remove_embankments_in_water(dissolved_waterdelen_noholes_split, embank_shp, embank_cleaned_shp, cs_4, cs_5)

    # Remove the sections that lay in non-urban areas
    # clean_cross_sections(embank_shp, cross_sections_final_with_non_urban, non_urban_areas, cross_sections_final_drop_non_urban)

    # todo: remove sections of which the embankment point lays in waterdeel, or if the embankment point lays on the edge of grid.

    # create_cross_section_points(cross_section_lines_shp, 100, points_shp)
# todo: add riverwidth to midpoint shapefile. If only one embankment, then we need to find the opposit edge of the waterdeel somehow
