"""
Now i shoot rays. I want to change it to selecting all buildings in a polygon (buffer) and then computing the river space from this.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from rtree import index
import multiprocessing as mp


def create_spatial_index(buildings):
    idx = index.Index()
    for i, building in enumerate(buildings.geometry):
        idx.insert(i, building.bounds)
    return idx


def get_max_height(geometry):
    if isinstance(geometry, Polygon):
        return np.max([coord[2] for coord in geometry.exterior.coords])
    elif isinstance(geometry, MultiPolygon):
        return np.max([np.max([coord[2] for coord in poly.exterior.coords]) for poly in geometry.geoms])
    else:
        raise ValueError(f"Unexpected geometry type: {type(geometry)}")


def get_intersection(cross_section, buildings, spatial_index):
    potential_matches_idx = list(spatial_index.intersection(cross_section.bounds))
    potential_matches = buildings.iloc[potential_matches_idx]
    print('potential matches: ', potential_matches)

    intersections = []
    for _, building in potential_matches.iterrows():
        if cross_section.intersects(building.geometry):
            intersection = cross_section.intersection(building.geometry)
            print('intersection ', intersection)
            max_height = get_max_height(building.geometry)
            intersections.append([intersection, max_height, building['identificatie']])

    if intersections:
        closest_intersection = min(intersections, key=lambda x: cross_section.project(x[0]))
        print("closest intersection : ", closest_intersection)
        return intersections, Point(closest_intersection[0].coords[0])
    return [], None


def process_cross_section(args):
    line, buildings, spatial_index, river, buffer_distance = args
    intersections, closest_building_point = get_intersection(line, buildings, spatial_index)
    print('intersections ', intersections)
    print('closest building point ', closest_building_point)

    if closest_building_point is not None:
        print('closest building point is not none')
        return closest_building_point
    else:
        buffer = river.buffer(buffer_distance)
        buffer_intersection = line.intersection(buffer.boundary)
        if buffer_intersection.is_empty:
            print("buffer intersection is empty")
            return None
        if isinstance(buffer_intersection, Point):
            return buffer_intersection
        elif isinstance(buffer_intersection, LineString):
            points = list(buffer_intersection.coords)
            if points:
                return Point(min(points, key=lambda p: Point(p).distance(
                    river.interpolate(river.project(Point(line.coords[0]))))))
        else:
            print('i dont know what happens')
    return None


def process_cross_sections(cross_sections, buildings, river, buffer_distance):
    spatial_index = create_spatial_index(buildings)
    # pool.map uses python's multiprocessing to parallelize the processing of cross-sections
    with mp.Pool() as pool:
        args = [(line.geometry, buildings, spatial_index, river.geometry.iloc[0], buffer_distance) for _, line in
                cross_sections.iterrows()]
        boundary_points = pool.map(process_cross_section, args)

    return [point for point in boundary_points if point is not None]


def order_points_along_river(points, river_line):
    projected_points = [river_line.interpolate(river_line.project(point)) for point in points]
    distances = [river_line.project(point) for point in projected_points]
    return [point for _, point in sorted(zip(distances, points), key=lambda x: x[0])]


def create_boundary_line(boundary_points, riverline):
    river = riverline.geometry.iloc[0]
    ordered_points = order_points_along_river(boundary_points, river)
    return ordered_points


def main():
    buffer_distance = 100
    buildings_file = '../3DBAG_combined_tiles/combined_3DBAG.gpkg'
    river_file = "../river_shapefiles/longest_river.shp"
    cross_sections_file = '../cross_sections/cs_1_interval0_5/cs_1_interval0_5.shp'
    output_file = "boundary_points_buildings/boundary_points_05_1.shp"

    print("Loading data...")
    buildings = gpd.read_file(buildings_file)
    river = gpd.read_file(river_file)
    cross_sections = gpd.read_file(cross_sections_file)

    print("Creating spatial index...")
    spatial_index = create_spatial_index(buildings)

    print("Processing cross-sections...")
    boundary_points = process_cross_sections(cross_sections, buildings, river, buffer_distance)

    print("Creating boundary line...")
    boundary_line = create_boundary_line(boundary_points, river)

    print("Saving output...")
    gdf_output = gpd.GeoDataFrame(geometry=[boundary_line], crs="EPSG:28992")
    gdf_output.to_file(output_file)

    print("Process completed.")


if __name__ == "__main__":
    main()