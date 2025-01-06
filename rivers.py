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

    for index, row in projected_gdf.iterrows():
        riverline = row['geometry']

        print("index, riverline length: ", index, riverline.length)
        # distance = 0
        total_length += riverline.length

        i = 0
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
                                 'rv_id': index})
            i += 1
            # distance = distance_along_line

        # starting_point = interval - (riverline.length - distance)

    # Save cross-sections to a Shapefile (.shp)
    gdf = gpd.GeoDataFrame(cross_sections)
    gdf.set_crs(epsg=28992, inplace=True)
    gdf.to_file(output_file_cs, driver='ESRI Shapefile')

    # Save river midpoints to shapefile (.shp)
    river_points_gdf = gpd.GeoDataFrame(river_points)
    river_points_gdf.set_crs(epsg=28992, inplace=True)
    river_points_gdf.to_file(output_river_mid, driver='ESRI Shapefile')
    print(f"total length river {total_length}")
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


def clean_cross_sections(cs_shp, non_urban_area_shp, output_file):
    gdf_cs = gpd.read_file(cs_shp)
    gdf_non_urban_areas = gpd.read_file(non_urban_area_shp)
    for index, row in tqdm(gdf_cs.iterrows(), total=gdf_cs.shape[0],
                       desc="Processing each section, checking if it lays in a non-urban area"):
        line_geom = row['geometry']
        intersections = gdf_non_urban_areas[gdf_non_urban_areas.intersects(line_geom)]
        if not intersections.empty:
            gdf_cs = gdf_cs.drop(index)

    # for _, row in tqdm(gdf_non_urban_areas.iterrows(),
    #                    total=gdf_non_urban_areas.shape[0],
    #                    desc="Processing non-urban areas"):
    #
    #     polygon_geom = row['geometry']
    #     intersections = gdf_cs[gdf_cs.intersects(polygon_geom)]
    #
    #     if not intersections.empty:
    #         for index, line in intersections.iterrows():
    #             gdf_cs = gdf_cs.drop(gdf_cs[(gdf_cs['rv_id'] == line['rv_id']) &
    #                                  (gdf_cs['cs_id'] == line['cs_id']) &
    #                                  (gdf_cs['side'] == line['side'])].index)

    gdf_cs.set_crs(epsg=28992, inplace=True)
    gdf_cs.to_file(output_file, driver= "ESRI Shapefile")


def clean_bridges(cs_shp, mid_shp, overbruggingsdeel_shp):
    gdf_cs = gpd.read_file(cs_shp)
    gdf_mid = gpd.read_file(mid_shp)
    gdf_bridges = gpd.read_file(cs_shp)




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

#This is the shapefile containing the areas with a population density higher than 1000 in the Netherlands
pop_dens_shp = "input/RIVERS/popdenshigehrthan1000raster.shp"
gdf_popdens = gpd.read_file(pop_dens_shp)
#Output file for rivers
rivers_total_shp = "input/RIVERS/OSM_rivers.shp"
# fetch_waterways_in_areas(gdf_popdens, rivers_total_shp)

rivers_shp = "input/RIVERS/OSM_rivers_clipped.shp"
river_gdf = gpd.read_file(rivers_shp)

cross_section_lines_directory = "output/CS/lines"
os.makedirs(cross_section_lines_directory, exist_ok=True)
cross_section_lines_shp = "output/CS/lines/cs_lines.shp"
cross_sections_lines_cleaned_shp = "output/CS/lines/cs_lines_cleaned.shp"
cross_section_mid_directory = "output/CS/mid"
os.makedirs(cross_section_mid_directory, exist_ok=True)
cross_section_mid_shp = "output/CS/mid/cs_mid.shp"

points_shp = "output/PTS/pts_shp"
non_urban_areas = "input/urban_areas/non_urban_areas.shp"

# cross_section_extraction(rivers_shp, 50, 450, cross_section_lines_shp, cross_section_mid_shp )
clean_cross_sections(cross_sections_lines_cleaned_shp, non_urban_areas, cross_sections_lines_cleaned_shp)
# create_cross_section_points(cross_section_lines_shp, 600, points_shp)
