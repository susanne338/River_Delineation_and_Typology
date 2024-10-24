"""
Extracts OSM river data by specifying the name. Either polygon or linestring
input: name of river or waterway and outputfile name
output: returns geodataframe AND could save as shapefile

TO DO:
Fix where it is saving to
The function depends on defining rivrs which may not be complete
"""
import osmnx as ox
import matplotlib.pyplot as plt
import overpy
import geopandas as gpd
from shapely.geometry import LineString, Polygon

def fetch_river_overpass(river_name, output_file):
    """
    :param river_name:
    :param output_file:
    :return: geodataframe of river eihter as geometry=lines or geometry = polygons AND saves as shapefile
    """
    api = overpy.Overpass()
    # These are all possibilities but I would get multiple rivers if I didn't specify that it was a canal. I need to alter the function so that I can specify more? or even just take some id instead of river name
    # Overpass query
    # query = f"""
    # [out:json];
    # (
    #   way["waterway"="river"]["name"="{river_name}"];
    #   way["waterway"="canal"]["name"="{river_name}"];
    #   relation["waterway"="river"]["name"="{river_name}"];
    #   way["water"="river"]["name"="{river_name}"];
    #   relation["water"="river"]["name"="{river_name}"];
    #   way["natural"="water"]["name"="{river_name}"];
    #   relation["natural"="water"]["name"="{river_name}"];
    # );
    # out body;
    # """
    query = f"""
        [out:json];
        (
          way["waterway"="canal"]["name"="{river_name}"];
        );
        out body;
        """

    result = api.query(query)

    lines = []
    polygons = []

    for way in result.ways:
        # Resolve missing nodes (fetch missing node data if needed)
        way.get_nodes(resolve_missing=True)
        coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
        lines.append(LineString(coords))

    for rel in result.relations:
        for member in rel.members:
            if member.geometry:
                coords = [(float(geom.lon), float(geom.lat)) for geom in member.geometry]
                polygons.append(Polygon(coords))

    # Save to shapefile
    if lines:
        gdf = gpd.GeoDataFrame(geometry=lines,  crs="EPSG:4326")
        gdf = gdf.to_crs("EPSG:28992")
        gdf.to_file(output_file, driver='ESRI Shapefile')
        print(f"Shapefile saved with line geometries: {output_file}")
        return gdf
    elif polygons:
        gdf = gpd.GeoDataFrame(geometry=polygons,  crs="EPSG:4326")
        gdf = gdf.to_crs("EPSG:28992")
        gdf.to_file(output_file, driver='ESRI Shapefile')
        print(f"Shapefile saved with polygon geometries: {output_file}")
        return gdf
    else:
        print(f"No data found for {river_name}")


# Example usage
# fetch_river_overpass("Kanaal door Walcheren", "river_shapefile/KanaalDoorWalcheren.shp")