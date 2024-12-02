"""
Fetches river from OSM and selects subrivers. Requires to take a look in QGIS what subrivers you want to use

input: Name and type of river
output: River shapefiles
"""

import overpy
from shapely.geometry import LineString, Polygon
import geopandas as gpd

#subrivers---------------------------------------------------------------------------------------------------
subrivers_dommel_eindhoven = [10,9,8,50,7]
subrivers_dommel_gestel = [52]
subrivers_maas_roermond = [10, 37, 6]
subrivers_maas_venlo = [7, 28, 27, 34]
subrivers_maas_cuijk = [64, 9, 18]
subrivers_maas_maastricht = [2, 5]
subrivers_ar_amsterdam = [0]
subrivers_ar_utrecht = [27, 25]
subrivers_ijssel_zwolle_deventer = [0]
subrivers_lek = [2]
subrivers_vliet = [0, 5, 4, 11, 10, 6, 3]
subrivers_schie = [0]
subrivers_winschoterdiep = [0, 1, 2]
subrivers_harinx = [1, 6, 4, 11, 12, 2]
subrivers_nijmegen = [12,13,23]
subrivers_zaltbommel = [14]

# OSM RIVER-------------------------------------------------------------------------------------------------------------
# to fetch, change tags in function
def fetch_river_overpass(river_name ,type, output_file):
    """
    :param river_name: String of river name
    :param output_file: Shapefile path to write river to
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
          way["waterway"={type}]["name"="{river_name}"];
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


river = 'dommel'
# os.makedirs(f"input/river/{river}", exist_ok=True)
# river_shp = f"input/river/{river}/{river}.shp"
# fetch_river_overpass('Dommel', 'river', river_shp)

def select_subrivers(river_shp, subrivers_list, river_output_shp):
    river = gpd.read_file(river_shp)
    selected_rivers = river[river['FID'].isin(subrivers_list)]
    selected_rivers.to_file(river_output_shp)


# city = 'eindhoven'
city = 'gestel'

# main_river = f'input/river/{river}/{river}.shp'
# river_folder = f'input/river/{river}/{city}'
# os.makedirs(river_folder, exist_ok=True)
# river_file = f'input/river/{river}/{city}/{city}.shp'
# select_subrivers(main_river, subrivers_dommel_gestel, river_file)
# select_subrivers(river, subrivers_maas_maastricht, river_maastricht)