"""
Delineates the river space based on first line of buildings. Uses cross-section intersection with building data.
Only 2D points for boundary line
TODO: As the river bends, cross-sections miss so also perform it from a buffer line.
TODO: smoothen line?
TODO: Instead of taking intersection with buffer, take last point on cross-section
"""
import geopandas as gpd
import numpy as np
from shapely import MultiLineString
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import multiprocessing as mp


def get_max_height(geometry):
    if isinstance(geometry, Polygon):
        return np.max([coord[2] for coord in geometry.exterior.coords])
    elif isinstance(geometry, MultiPolygon):
        return np.max([np.max([coord[2] for coord in poly.exterior.coords]) for poly in geometry.geoms])
    else:
        raise ValueError(f"Unexpected geometry type: {type(geometry)}")


def get_intersection(cross_section, buildings):
    # Get intersecting buildings directly using spatial operation
    intersecting_buildings = buildings[buildings.intersects(cross_section)]

    if len(intersecting_buildings) == 0:
        return [], None

    intersections = []
    for idx, building in intersecting_buildings.iterrows():
        try:
            intersection = cross_section.intersection(building.geometry)

            # Convert various geometry types to Point
            if isinstance(intersection, LineString):
                intersection_point = Point(intersection.coords[0])
            elif isinstance(intersection, MultiPolygon):
                # Take the centroid of the first polygon
                intersection_point = Point(list(intersection.geoms)[0].centroid)
            elif isinstance(intersection, Polygon):
                intersection_point = Point(intersection.centroid)
            elif isinstance(intersection, Point):
                intersection_point = intersection
            elif isinstance(intersection, MultiLineString):
                # Get the first LineString in the MultiLineString
                first_line = intersection.geoms[0]
                # Get the first coordinate of this LineString
                first_coord = first_line.coords[0]
                # Create a Point from the first coordinate
                intersection_point = Point(first_coord)
            else:
                print(f"Unexpected intersection type: {type(intersection)}")
                continue

            max_height = get_max_height(building.geometry)
            intersections.append([intersection_point, max_height, building['identificatie']])

        except Exception as e:
            print(f"Error processing building {idx}: {e}")

    if intersections:
        # Use the actual cross_section line for projection
        line_for_projection = cross_section
        closest_intersection = min(intersections,
                                 key=lambda x: line_for_projection.project(x[0]))  # x[0] is already a Point
        return intersections, closest_intersection[0]
    return [], None

def split_cross_section(cross_section, river):
    """Split cross-section into left and right parts based on river centerline."""
    intersection_point = cross_section.intersection(river)
    if intersection_point.is_empty:
        print("intersection point is empty")
        return None, None

    coords = list(cross_section.coords)
    split_point = list(intersection_point.coords)[0]
    left_line = LineString([split_point, coords[0]])
    right_line = LineString([split_point, coords[-1]])
    return left_line, right_line


def get_buffer_intersection(line, buffer):
    buffer_intersection = line.intersection(buffer.boundary)
    if buffer_intersection.is_empty:
        return None
    if isinstance(buffer_intersection, Point):
        return buffer_intersection
    elif isinstance(buffer_intersection, LineString):
        points = list(buffer_intersection.coords)
        if points:
            return Point(points[0])  # Take the first intersection point
    return None


def process_cross_section(args):
    line, buildings, river, buffer_distance = args

    # Split cross-section into left and right parts
    left_line, right_line = split_cross_section(line, river)
    if left_line is None or right_line is None:
        print('there are no left and right lines!')
        return None

    boundary_points = []

    # Process left side
    left_intersections, left_point = get_intersection(left_line, buildings)
    if left_point is None:
        left_buffer = river.buffer(buffer_distance)
        left_buffer_point = get_buffer_intersection(left_line, left_buffer)
        if left_buffer_point is not None:
            boundary_points.append(left_buffer_point)
    else:
        boundary_points.append(left_point)

    # Process right side
    right_intersections, right_point = get_intersection(right_line, buildings)
    if right_point is None:
        right_buffer = river.buffer(buffer_distance)
        right_buffer_point = get_buffer_intersection(right_line, right_buffer)
        if right_buffer_point is not None:
            boundary_points.append(right_buffer_point)
    else:
        boundary_points.append(right_point)

    return boundary_points


def process_cross_sections(cross_sections, buildings, river, buffer_distance):
    with mp.Pool() as pool:
        args = [(line.geometry, buildings, river.geometry.iloc[0], buffer_distance)
                for _, line in cross_sections.iterrows()]
        boundary_points_lists = pool.map(process_cross_section, args)

    # Flatten the list of boundary points
    boundary_points = [point for sublist in boundary_points_lists if sublist is not None
                      for point in sublist if point is not None]
    return boundary_points


def order_points_along_river(points, river_line):
    """
    Order points based on their projection along the river line.
    Separates points into left and right banks based on their position relative to the river.
    """
    # Project points onto river line
    projected_points = [(point, river_line.interpolate(river_line.project(point))) for point in points]

    # Determine which side of the river each point is on
    left_points = []
    right_points = []

    for point, projected in projected_points:
        # Create a vector from projected point to actual point
        vector = np.array([point.x - projected.x, point.y - projected.y])

        # Create a vector along the river at this point
        river_point = river_line.interpolate(river_line.project(point))
        river_point_ahead = river_line.interpolate(river_line.project(point) + 1)
        river_vector = np.array([river_point_ahead.x - river_point.x,
                                 river_point_ahead.y - river_point.y])

        # Cross product to determine side (positive = left, negative = right)
        cross_product = np.cross(river_vector, vector)

        if cross_product > 0:
            left_points.append((river_line.project(point), point))
        else:
            right_points.append((river_line.project(point), point))

    # Sort points by their distance along the river
    left_points.sort(key=lambda x: x[0])
    right_points.sort(key=lambda x: x[0])

    return [p[1] for p in left_points], [p[1] for p in right_points]


def create_boundary_line(boundary_points, riverline):
    """
    Create two boundary lines (left and right bank) from the boundary points.
    Returns a list of two LineStrings representing the left and right river space boundaries.
    """
    river = riverline.geometry.iloc[0]

    # Separate and order points for left and right banks
    left_points, right_points = order_points_along_river(boundary_points, river)

    # Extract only x and y from POINT or POINT Z objects
    left_points = [(point.x, point.y) for point in left_points]  # Get only 2D coordinates
    right_points = [(point.x, point.y) for point in right_points]  # Get only 2D coordinates

    # Create LineStrings for both banks if we have enough points
    boundary_lines = []

    if len(left_points) >= 2:
        left_line = LineString(left_points)
        boundary_lines.append(left_line)
    else:
        print("Warning: Not enough points for left bank boundary")

    if len(right_points) >= 2:
        right_line = LineString(right_points)
        boundary_lines.append(right_line)
    else:
        print("Warning: Not enough points for right bank boundary")

    return boundary_lines

def boundary_line_polygon(boundary_line, river, output_file):
    """
    Create a polygon by connecting the entire river line and the boundary line of the building.

    Parameters:
    - boundary_line: The building's boundary side (as a LineString).
    - river_line: The river's geometry (LineString).

    Returns:
    - Polygon object representing the area between the boundary and the river.
    TODO: this needs to take river edge instead i think
    """
    # Get the points of the boundary and river lines
    boundary_points = list(boundary_line.coords)
    river_geom = river.geometry.iloc[0]
    river_points = list(river_geom.coords)

    # Combine the boundary points and the river points (in reverse order to close the polygon correctly)
    polygon_points = boundary_points + river_points[::-1]
    # polygon_points = boundary_points + river_points
    polygon = Polygon(polygon_points)

    gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=28992)
    gdf.to_file(output_file)

    # Create and return the closed polygon
    return polygon


# Define your file paths and parameters
buffer_distance = 100
buildings_file = 'thesis_output/3DBAG_combined_tiles/combined_3DBAG.gpkg'
river_file = "input/river/longest_river.shp"
cross_sections_file = 'output/cross_sections/cross_sections_int05.shp'
output_file = "output/riverspace_delineation/boundary_points.shp"

if __name__ == '__main__':
    print("Loading data...")
    buildings = gpd.read_file(buildings_file, layer="pand")
    river = gpd.read_file(river_file)
    cross_sections = gpd.read_file(cross_sections_file)

    print("Processing cross-sections...")
    boundary_points = process_cross_sections(cross_sections, buildings, river, buffer_distance)

    print(f"Found {len(boundary_points)} boundary points")

    print("Creating boundary lines...")
    boundary_lines = create_boundary_line(boundary_points, river)
    print(f"Created {len(boundary_lines)} boundary lines")

    # # Save boundary points for debugging (optional)
    # points_gdf = gpd.GeoDataFrame(geometry=boundary_points, crs="EPSG:28992")
    # points_gdf.to_file("output/riverspace_delineation/debug_boundary_points.shp")
    #
    # print("Saving output...")
    # gdf_output = gpd.GeoDataFrame(geometry=boundary_lines, crs="EPSG:28992")
    # gdf_output.to_file(output_file)

    print("Creating polygons")
    boundary_line_polygon(boundary_lines[0], river, 'output/riverspace_delineation/left_polygon')
    boundary_line_polygon(boundary_lines[1], river, 'output/riverspace_delineation/right_polygon')

    print("Process completed.")