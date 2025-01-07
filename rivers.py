import os
from tqdm import tqdm
import overpy
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
import geopandas as gpd
import numpy as np

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

#cs
def cross_section_extraction(river_file, interval, width, output_file_cs, output_river_mid):

    river = gpd.read_file(river_file)
    projected_gdf = river.to_crs(epsg=28992)
    cross_sections = []
    river_points = []
    total_length = 0

    # starting_point = 0
    i = 0

    for index, row in tqdm(projected_gdf.iterrows(), total=projected_gdf.shape[0], desc="Processing each river"):
        riverline = row['geometry']

        # print("index, riverline length: ", index, riverline.length)
        # distance = 0
        total_length += riverline.length


        for distance_along_line in np.arange(0, riverline.length, interval):
            side_0, side_1, point_on_line = get_perpendicular_cross_section(riverline, distance_along_line, width)

            cross_sections.append({
                'geometry': side_0,
                'rv_id': index,
                'cs_id': i,
                'side': 0
            })
            cross_sections.append({
                'geometry': side_1,
                'rv_id': index,
                'cs_id': i,
                'side': 1
            })

            river_points.append({'geometry': point_on_line,
                                 'rv_id': index,
                                'cs_id': i})
            i += 1
            # distance = distance_along_line

        # starting_point = interval - (riverline.length - distance)

    # Save cross-sections to a Shapefile (.shp)
    gdf = gpd.GeoDataFrame(cross_sections)
    gdf.set_crs(epsg=28992, inplace=True)
    gdf.to_file(output_file_cs, driver='ESRI Shapefile')

    # Save river midpoints to shapefile (.shp)
    # river_points_gdf = gpd.GeoDataFrame(river_points)
    # river_points_gdf.set_crs(epsg=28992, inplace=True)
    # river_points_gdf.to_file(output_river_mid, driver='ESRI Shapefile')
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


def clean_bridges(cs_clean_shp, cs_clean_nobridge_shp, mid_shp, overbruggingsdeel_shp, bridge_points_output_shp):
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

    for index, row in tqdm(gdf_bridges.iterrows(), total=gdf_bridges.shape[0],
                           desc="Processing each bridge"):
        bridge_geom = row['geometry']
        intersections = gdf_mid[gdf_mid.intersects(bridge_geom)]

        if not intersections.empty:
            # print(f"Intersection found! for bridge {index}")

            for idx, point in intersections.iterrows():
                pt_geom = point['geometry']
                pt_id = point['cs_id']
                # print(f"point geometry is {point['geometry']} and id of river is is {point['rv_id']}")
                bridge_points.append({'geometry': pt_geom, 'cs_id': pt_id})

                # Midpoint is marked as bridge
                gdf_mid.loc[gdf_mid['cs_id']== pt_id, 'bridge'] = 1

                # Remove sections
                # rows_to_drop = gdf_cs[gdf_cs['cs_id'] == pt_id].index
                # if not rows_to_drop.empty:
                #     gdf_cs = gdf_cs.drop(rows_to_drop)
                # gdf_cs.drop(gdf_cs[(gdf_cs['cs_id'] == pt_id)].index, inplace=True)

    # bridge_midpoints = gpd.GeoDataFrame(bridge_points)
    # bridge_midpoints.set_crs(epsg=28992, inplace=True)
    # bridge_midpoints.to_file(bridge_points_output_shp, driver="ESRI Shapefile")

    print("Start removing bridges from the cs shapefile...")
    # Remove rows in gdf_cs where 'cs_id' matches
    bridge_cs_ids = gdf_mid[gdf_mid['bridge'] == 1]['cs_id'].unique()
    print(f"CS IDs to remove: {bridge_cs_ids}")


    gdf_cs_cleaned = gdf_cs[~gdf_cs['cs_id'].isin(bridge_cs_ids)]

    # Save the cleaned GeoDataFrame
    gdf_cs_cleaned.to_file(cs_clean_nobridge_shp, driver="ESRI Shapefile")
    print(f"Cleaned from bridges GeoDataFrame saved to {cs_clean_nobridge_shp}")

    gdf_mid.to_file(mid_shp, driver= "ESRI Shapefile")
    print(f"bridges amount of midpoints is {gdf_mid[gdf_mid['bridge'] == 1].shape[0]}")


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






# Output file for rivers
rivers_total_shp = "input/RIVERS/OSM_rivers.shp"
# fetch_waterways_in_areas(gdf_popdens, rivers_total_shp)
# --> Clip in QGIS

# This file is all the rivers that fall into urban areas
rivers_shp = "input/RIVERS/OSM_rivers_clipped.shp"
#This is the shapefile containing the areas with a population density higher than 1000 in the Netherlands
# pop_dens_shp = "input/RIVERS/popdenshigehrthan1000raster.shp"

#These are the files needed to create the cross sections
cross_section_lines_directory = "output/CS/lines"
os.makedirs(cross_section_lines_directory, exist_ok=True)
cross_section_lines_shp = "output/CS/lines/cs_lines.shp"
cross_sections_lines_cleaned_shp = "output/CS/lines/cs_lines_cleaned.shp"
cross_sections_lines_cleaned_nobridge_shp = "output/CS/lines/cs_lines_nobridge_cleaned.shp"

cross_section_mid_directory = "output/CS/mid"
os.makedirs(cross_section_mid_directory, exist_ok=True)
cross_section_mid_shp = "output/CS/mid/cs_mid.shp"

# CLEANUP: remove bridges and remove non-urban areas
# These are the bgt bridge ('overbruggingsdeel') polygons in the urban areas
bridges_shp = "input/DATA/BGT/bgt_overbrugginsdeel_clipped.shp"
bridges_midpoints_shp = "input/DATA/BGT/bridges_mid.shp"
# This is the shapefile defining non-urban areas in the Netherlands (opposite of the urban areas file)
non_urban_areas = "input/urban_areas/non_urban_areas.shp"

# Output file for the points
points_shp = "output/PTS/pts_shp"

# remove_bridge_rows(cross_section_mid_shp, cross_sections_lines_cleaned_shp, cross_sections_lines_cleaned_nobridge_shp)

"""
First create the initial cross-sections and midpoints of our segments.
Then clean the sections by removing the ones that lay in non-urban areas.
Clean by removing sections that are bridges.
todo: I make a new file now for bridge points but i can better add it as a new column to midpoint as bridge: 0, 1
"""
cross_section_extraction(rivers_shp, 50, 450, cross_section_lines_shp, cross_section_mid_shp )
# clean_cross_sections(cross_sections_lines_shp, non_urban_areas, cross_sections_lines_cleaned_shp)
# clean_bridges(cross_sections_lines_cleaned_shp, cross_sections_lines_cleaned_nobridge_shp, cross_section_mid_shp, bridges_shp, bridges_midpoints_shp)



# todo: find embankment, make new sections

# create_cross_section_points(cross_section_lines_shp, 100, points_shp)
