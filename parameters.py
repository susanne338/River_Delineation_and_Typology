"""
execute river retrieval, preprocess and visibility before

6. Adds parameters values to points: Adds elev_dsm, imperviousness, landuse, visibility, and flood to points shapefile
7. Splits sections, adds column 'river' and finds building intersections.


"""
import os
import fiona
import pyproj
import rasterio
import pandas as pd
from rasterio.transform import rowcol
from shapely import LineString, Point, MultiPolygon, MultiLineString, Polygon
from tqdm import tqdm
from rasterio.windows import from_bounds
import geopandas as gpd
import numpy as np
# from data_retrieval import run_data_retrieval
# from visibility import combine_viewsheds

def add_elevation_from_tiles(shapefile_path, tiles_folder, elevation_column_name):
    """
    Adds elevation data from tiles to existing shapefile and updates it.

    Parameters:
    shapefile_path: Path to the existing shapefile with points
    tiles_folder: Folder containing elevation tiles (.tif files)
    elevation_column_name: Name of the column to store elevation values

    Returns:
    GeoDataFrame with added elevation column and count of missing points
    """
    # Read existing shapefile
    points_gdf = gpd.read_file(shapefile_path)
    print(f"Loaded {len(points_gdf)} points from {shapefile_path}")

    # Initialize elevation column with NaN
    points_gdf[elevation_column_name] = np.nan

    # Get list of tif files
    tif_files = [f for f in os.listdir(tiles_folder) if f.endswith(".tif")]
    print(f"List of tif file {tif_files}")
    missing_points = 0
    nodata_points = 0

    for idy, tif_file in tqdm(enumerate(tif_files), total=len(tif_files), desc="Processing points"):
        print(f"name of tif file {tif_file}")
        tif_path = os.path.join(tiles_folder, tif_file)
        elevation_found = False
        with rasterio.open(tif_path) as src:
            tif_bounds = src.bounds
            minx, miny, maxx, maxy = tif_bounds
            for idx, row in points_gdf.iterrows():
                if row.geometry is not None:
                    point = row.geometry

                    if minx <= point.x <= maxx and miny <= point.y <= maxy:
                        try:
                            py, px = src.index(point.x, point.y)
                            window = rasterio.windows.Window(px - 1, py - 1, 2, 2)
                            data = src.read(1, window=window)
                            center_value = data[1, 1]
                            if center_value == src.nodata:
                                is_nodata = True
                            elif data.size > 0:
                                points_gdf.at[idx, elevation_column_name] = float(center_value)
                                elevation_found = True
                                continue

                        except (IndexError, ValueError):
                            print(f"there was an error processing this point in the tile")
                            continue

                        if not elevation_found:
                            missing_points += 1
                            if is_nodata:
                                nodata_points += 1
                                print(f"NoData value found for point {idx} at ({point.x}, {point.y})")
                            else:
                                print(
                                    f"No elevation data found for point {idx} at ({point.x}, {point.y}) - point may be between tiles")

    # Save the updated GeoDataFrame
    points_gdf.to_file(shapefile_path, driver='ESRI Shapefile')
    print(f"\nUpdated shapefile with {elevation_column_name} at: {shapefile_path}")
    print(f"Total missing points: {missing_points}")

    return points_gdf

def process_landuse(points_gdf, gml_folder):
    """
    Add landuse information to points GeoDataFrame from gpkg files from BGT data.

    Args:
        points: geodataframe of points
        gml_folder: Folder containing .gml files

    Returns:
        GeoDataFrame: Updated points with landuse information
    """

    # Initialize landuse column
    points_gdf = points_gdf.copy(deep = True)
    points_gdf = points_gdf.to_crs(epsg=28992)
    points_gdf['landuse'] = [[] for _ in range(len(points_gdf))]

    # counter for points that get a landuse
    points_with_landuse = set()

    bbox = points_gdf.total_bounds
    # print('I have found the bbox of points: ', bbox)
    # print('total number of points: ', len(points_gdf))

    # Loop through gmlfiles
    for i, gml_file in enumerate(os.listdir(gml_folder)):
        if not gml_file.endswith('.gml'):
            continue

        gml_path = os.path.join(gml_folder, gml_file)
        print(f'\nProcessing file: {gml_file}')
        original_crs = 'epsg:28992'
        # with fiona.open(gml_path) as src:
        #     print(f"Fiona driver: {src.driver}")
        #     print(f"Fiona CRS: {src.crs}") # No crs for some reason
        #     print(f"Fiona schema: {src.schema}")
        #     print(f"Fiona bounds: {src.bounds}")

        for layer_name in fiona.listlayers(gml_path):
            try:
                print(f'Processing layer: {layer_name} in file: {gml_file}')
                landuse_gdf = gpd.read_file(gml_path, layer=layer_name)

                if len(landuse_gdf) == 0:
                    print(f'Layer {layer_name} is empty')
                    continue

                landuse_filtered = landuse_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                # print('I filtered the landuse to bbox')

                if landuse_filtered.crs is None:
                    landuse_filtered.set_crs(original_crs, inplace=True)

                if len(landuse_filtered) == 0:
                    # print(f'No features in bbox for layer {layer_name}')
                    continue

                # Ensure CRS matches
                if landuse_filtered.crs != points_gdf.crs:
                    landuse_filtered = landuse_filtered.to_crs(points_gdf.crs)
                    # points_gdf = points_gdf.to_crs(landuse_filtered.crs)
                    # print('crs landuse ', landuse_filtered.crs)
                # print('points bbox now is ', points_gdf.total_bounds)
                # print('landuse filtered columns are ', landuse_filtered.columns)
                # print("Filtered landuse count:", len(landuse_filtered))
                    # Check for valid geometries
                invalid_geoms = ~landuse_filtered.geometry.is_valid
                if invalid_geoms.any():
                    print(f'Found {invalid_geoms.sum()} invalid geometries. Attempting to fix...')
                    landuse_filtered.geometry = landuse_filtered.geometry.buffer(0)
                # Process each point
                # I get a warning here that says I should work with a copy but I don't want to I want to write to this specific one no?
                joined_gdf = gpd.sjoin(points_gdf, landuse_filtered, how="left", predicate="intersects")
                matches_in_layer = len(joined_gdf[~joined_gdf.index_right.isna()])
                print(f'Points with matches in this layer: {matches_in_layer}')
                # It's this line: that points_gdf is being altered instead of a copy of it
                # points_gdf.loc[joined_gdf.index, 'landuse'] = layer_name
                # For each point that intersects with a landuse polygon, append the layer_name
                for idx in joined_gdf.index[~joined_gdf.index_right.isna()]:
                    if layer_name not in points_gdf.at[idx, 'landuse']:
                        points_gdf.at[idx, 'landuse'].append(layer_name)
                        points_with_landuse.add(idx)

            except Exception as e:
                print(f'Error processing layer {layer_name}: {str(e)}')
                continue
                # Process each point for CBS data
                # joined_gdf = gpd.sjoin(points_gdf, landuse_filtered, how="left", predicate="intersects")
                # points_gdf['landuse'] = joined_gdf['description']

    # Convert landuse lists to strings for easier handling
    points_gdf['landuse'] = points_gdf['landuse'].apply(lambda x: '; '.join(x) if x else None)
    # Final statistics
    total_points = len(points_gdf)
    points_with_values = len(points_with_landuse)
    points_without_values = total_points - points_with_values
    # Create a summary of found landuse types
    landuse_summary = points_gdf['landuse'].value_counts()
    print("\nLanduse distribution:")
    print(landuse_summary)

    if points_without_values > 0:
        missing_points = points_gdf[points_gdf['landuse'].isna()].head()
        print('\nSample of points without landuse:')
        print('Coordinates:')
        print(missing_points.geometry.to_string())

    return points_gdf


def add_landuse_to_shapefile(shapefile_path, gml_folder):
    """
    Reads a shapefile, adds landuse information from gml files, and saves the updated shapefile.

    Parameters:
        shapefile_path: Path to the shapefile containing all cross-section points
        gml_folder: path to folder of gml files
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    print('The shapefile has been read! its columns are: ', gdf.columns)

    # Add landuse information
    gdf = process_landuse(gdf, gml_folder)
    print("landuse is added to the geodataframe!")

    # Save updated shapefile
    gdf.to_file(shapefile_path)
    print(f"Updated shapefile saved to: {shapefile_path}")


def extract_unique_landuses(shapefile_path, output_file):
    """
    Extract unique landuse types from a shapefile,
    and assigning unique indices to individual landuse types.
    Args:
        shapefile_path: Path to shapefile of points that contains landuse information
        output_file: path to .csv file where we want to store the unique landuses list

    Returns:

    """

    points_gdf = gpd.read_file(shapefile_path)
    individual_landuses = []

    # Process each landuse entry
    for landuse in points_gdf['landuse'].dropna():
        # Split by semicolon and strip whitespace
        parts = [part.strip() for part in str(landuse).split(';')]
        individual_landuses.extend(parts)

    # Get unique values and sort them
    unique_landuses = sorted(list(set(individual_landuses)))

    # Create DataFrame with index
    landuses_df = pd.DataFrame({
        'landuse': unique_landuses,
        'index': range(len(unique_landuses))
    })

    # Save to CSV
    landuses_df.to_csv(output_file, index=False)
    print(f"Unique landuses saved to: {output_file}")

    return landuses_df


# IMPERVIOUSNESS AND VISIBILITY
def normalize_crs(crs):
    """
    Normalize CRS to EPSG:28992 if it's any variant of Amersfoort RD New
    (I was having problems with some 'local' crs)
    """
    if crs:
        # Check for various forms of Amersfoort RD New
        if any(marker in str(crs).upper() for marker in ['AMERSFOORT', 'RD NEW', '28992']):
            return pyproj.CRS.from_epsg(28992)
    return crs


def load_raster_data(raster_path):
    """
    Load raster data and return necessary components for processing.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # Read the first band
        transform = src.transform
        bounds = src.bounds
        crs = normalize_crs(src.crs)
        print(f"Raster transform: {transform}")
        print(f"Raster shape: {data.shape}")
    return data, transform, bounds, crs


def check_raster_value(location, raster_data, transform, bounds):
    """
    Check raster value at a given point location.
    """
    if location is None or location.is_empty or location.geom_type != 'Point':
        print(f"Invalid location: {location}")
        return np.nan

    x, y = location.x, location.y

    # Check if point is within raster bounds with a small buffer
    buffer = 1.0  # 1 meter buffer
    if not (bounds.left - buffer <= x <= bounds.right + buffer and
            bounds.bottom - buffer <= y <= bounds.top + buffer):
        print(f"Point ({x}, {y}) is outside raster bounds: {bounds}")
        return np.nan

    try:
        # Convert coordinates to pixel indices using rasterio's rowcol function
        row, col = rowcol(transform, x, y)

        # Convert to integers
        row, col = int(row), int(col)

        # Debug information
        # print(f"Point coordinates: ({x}, {y})")
        # print(f"Pixel coordinates: (row={row}, col={col})")
        # print(f"Raster shape: {raster_data.shape}")

        # Ensure indices are within array bounds
        if 0 <= row < raster_data.shape[0] and 0 <= col < raster_data.shape[1]:
            value = raster_data[row, col]
            # print(f"Sampled value: {value}")
            return value
        else:
            print(f"Computed pixel coordinates ({row}, {col}) are outside raster dimensions {raster_data.shape}")
            return np.nan

    except Exception as e:
        print(f"Error processing point ({x}, {y}): {str(e)}")
        return np.nan


def compute_raster_value(row, raster_data, transform, bounds):
    """
    Compute raster value for a GeoDataFrame row.
    """
    location = row['geometry']
    return check_raster_value(location, raster_data, transform, bounds)


def add_raster_column(shapefile_path, raster_path, column_name, overwrite=True):
    """
    Add raster values as a new column to a shapefile.
    """

    raster_data, transform, bounds, raster_crs = load_raster_data(raster_path)
    gdf = gpd.read_file(shapefile_path)

    # Normalize the shapefile CRS
    gdf.crs = normalize_crs(gdf.crs)

    # Verify points overlap with raster
    points_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    if not (bounds.left <= points_bounds[2] and points_bounds[0] <= bounds.right and
            bounds.bottom <= points_bounds[3] and points_bounds[1] <= bounds.top):
        print("WARNING: Points extent does not overlap with raster extent!")

    # Check if column exists
    if column_name in gdf.columns and not overwrite:
        raise ValueError(f"Column {column_name} already exists and overwrite=False")

    # Process points in smaller chunks
    chunk_size = 1000
    num_chunks = len(gdf) // chunk_size + (1 if len(gdf) % chunk_size else 0)

    results = []
    for i in tqdm(range(num_chunks), desc=f"Processing chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(gdf))
        chunk = gdf.iloc[start_idx:end_idx]

        chunk_results = chunk.apply(
            compute_raster_value,
            axis=1,
            raster_data=raster_data,
            transform=transform,
            bounds=bounds
        )
        results.extend(chunk_results)

    gdf[column_name] = results
    gdf.to_file(shapefile_path)
    print(f"\nUpdated shapefile saved to: {shapefile_path}")

    # Print summary statistics
    # valid_values = gdf[column_name].dropna()
    # print("\nSummary statistics for sampled values:")
    # print(f"Total points: {len(gdf)}")
    # print(f"Valid values: {len(valid_values)}")
    # print(f"Invalid/out of bounds: {len(gdf) - len(valid_values)}")
    # if len(valid_values) > 0:
    #     print(f"Min value: {valid_values.min()}")
    #     print(f"Max value: {valid_values.max()}")
    #     print(f"Mean value: {valid_values.mean():.2f}")


def split_cross_sections(midpoints_shp, cross_sections_shp, halves_shp):
    """
    Splits the cross-section geometries in half and adds the left and right linestrings
    to a new shapefile.
    halves shapefile is stored in same place as where midpoint shapefile is

    Args:
        midpoints_shp (str): File path to the midpoints shapefile.
        cross_sections_shp (str): File path to the cross-sections shapefile.

    Returns:
        None
    """
    midpoints = gpd.read_file(midpoints_shp)
    cross_sections = gpd.read_file(cross_sections_shp)

    # Create lists to store the data for the output GeoDataFrame
    # ids = []
    # sides = []
    geometries = []

    for idx, row in cross_sections.iterrows():
        cross_section = row.geometry
        if cross_section is not None:
            midpoint = midpoints[midpoints['FID'] == row['id']].iloc[0].geometry
            coords = list(cross_section.coords)
            left_line = LineString([midpoint, coords[0]])
            right_line = LineString([midpoint, coords[-1]])

            # Append the left and right linestrings to the lists
            # ids.extend([row['FID'], row['FID']])
            # sides.extend([0, 1])
            # geometries.extend([left_line, right_line])
            print(f"i append {row['id']}, 0 and {left_line}")
            geometries.append([row['id'], 0, left_line])
            geometries.append([row['id'], 1, right_line])

    # Create the output GeoDataFrame
    # output_df = gpd.GeoDataFrame({'id': ids, 'side': sides, 'geometry': geometries}, crs=midpoints.crs)
    output_df = gpd.GeoDataFrame(geometries, columns=['id', 'side', 'geometry'])
    output_df.set_crs("EPSG:28992", inplace=True)
    # Save the output GeoDataFrame to a new shapefile

    # output_df.to_file(midpoints_shp.replace('.shp', '_halves.shp'))
    output_df.to_file(halves_shp, driver='ESRI Shapefile')


def split_points_identify_river(points_shp, embankment_shp, cross_section_shp, midpoint_shp):
    """
    Splits the points file in the two half by creating columns 'side' and 'split_h_di'
    Splitpoint is determined as half of max h_distance
    Adds column 'river' with 0 being not river and 1 being river


    side: 1 for right side, 0 for left side
    split_h_di give the distance along line from midpoint

    Args:
        points_shp: shapefile of points with parameters

    Returns:

    """
    gdf = gpd.read_file(points_shp)
    gdf_embankment = gpd.read_file(embankment_shp)
    grouped = gdf.groupby('id')
    invalid_ids = []

    # Initializing river column
    gdf['river'] = 0

    for idx, section in tqdm(grouped, desc="Processing sections for splitting points into halves and identifying river"):
        id = section.iloc[0]['id']

        embankment_row_left = gdf_embankment[(gdf_embankment['id'] == id) & (gdf_embankment['side'] == 0) ]
        embankment_row_right = gdf_embankment[(gdf_embankment['id'] == id) & (gdf_embankment['side'] == 1)]

        left_embankment = embankment_row_left['h_dist'].values[0] if not embankment_row_left.empty else None
        right_embankment = embankment_row_right['h_dist'].values[0] if not embankment_row_right.empty else None

        if left_embankment is None or right_embankment is None:
            print(f"Warning: Could not find embankment for section {id}")
            continue

        section_sorted = section.sort_values('h_distance')
        section_sorted.loc[(section_sorted['h_distance'] >= left_embankment) &
                           (section_sorted['h_distance'] <= right_embankment), 'river'] = 1


        # Check neighboring points of the middle point
        mid_index = len(section_sorted) // 2
        expand_up = True
        expand_down = True
        range_end = len(section_sorted) / 2

        for offset in range(1, len(section_sorted)):
            # Check above the middle point
            upper_index = mid_index + offset
            lower_index = mid_index - offset

            if section_sorted.iloc[mid_index]['landuse'] != 'Waterdeel':
                # then this becomes an invalid section. Then we need to remove all the data and represent it as invalid
                invalid_ids.append(id)
                break

            # Check upper side
            if expand_up and upper_index < len(section_sorted):
                if section_sorted.iloc[upper_index]['river'] == 1:
                    continue # or break?
                if section_sorted.iloc[upper_index]['landuse'] == 'Waterdeel':
                    section_sorted.loc[upper_index, 'river'] = 1
                else:
                    expand_up = False

            # Check lower side
            if expand_down and lower_index >= 0:
                if section_sorted.iloc[lower_index]['river'] == 1:
                    continue # or break?
                if section_sorted.iloc[lower_index]['landuse'] == 'Waterdeel':
                    section_sorted.loc[lower_index, 'river'] = 1
                else:
                    expand_down = False

            # Stop if both directions can no longer expand
            if not expand_up and not expand_down:
                break

        gdf.loc[section_sorted.index, 'river'] = section_sorted['river']


        #Split points. Adds 'side' column
        max_h_distance = section['h_distance'].max()
        splitpoint = max_h_distance / 2

        # boolean to determine side. True is converted by astype(int) to 1, and False to 0 integers.
        section['side'] = (section['h_distance'] >= splitpoint).astype(int)

        # Create the new h_distance column as the absolute difference from the splitpoint
        section['split_h_di'] = (section['h_distance'] - splitpoint).abs()

        # Update the main GeoDataFrame with the modified 'section'
        gdf.loc[gdf['id'] == idx, ['side', 'split_h_di']] = section[['side', 'split_h_di']]

    # Remove invalid segments from all the files
    print(f"About to remove invalid points: {invalid_ids}")
    gdf = gdf[~gdf['id'].isin(invalid_ids)]
    gdf_embankment = gdf_embankment[~gdf_embankment['id'].isin(invalid_ids)]

    cs_gdf = gpd.read_file(cross_section_shp)
    cs_gdf = cs_gdf[~cs_gdf['id'].isin(invalid_ids)]

    mid_gdf = gpd.read_file(midpoint_shp)
    columns_to_nullify = [col for col in mid_gdf.columns if col not in ['geometry', 'FID']]
    mid_gdf.loc[mid_gdf['FID'].isin(invalid_ids), columns_to_nullify] = None


    #Save everything to files (overwrite)
    gdf.to_file(points_shp)
    gdf_embankment.to_file(embankment_shp)
    cs_gdf.to_file(cross_section_shp)
    mid_gdf.to_file(midpoint_shp)

    return


def building_parameters(halves_shp, building_gpkg, intersection_output_file, closest_intersection_output_file):
    """
    Compute parameters related to building intersections with the river space.
    This version handles cross-sections that have already been split into left and right parts, with the input file in a custom format.

    Parameters:
    cs_halvex_shp (str): Path to the Shapefile containing the pre-split cross-sections.
    building_gpkg (str): Path to the GeoPackage containing the building data.
    river_shp (str): Path to the Shapefile containing the river geometry.

    Returns:
    GeoDataFrame: The input cs_halvex_shp GeoDataFrame with additional columns:
        buil1 (Point): The first intersection point
        height1 (float): The maximum building height of the first intersection
        buil_int (MultiLineString): The geometry of all intersection points
    """
    # Load data
    halves = gpd.read_file(halves_shp)
    # buildings = gpd.read_file(building_gpkg, layer="pand")
    buildings = gpd.read_file(building_gpkg, layer="lod12_2d")
    # river = gpd.read_file(river_shp)

    # Prepare additional columns in the cs_halvex GeoDataFrame
    # cs_halvex['build1'] = None
    # cs_halvex['height1'] = None

    intersections_list = []
    closest_intersections_list = []

    for _, row in halves.iterrows():
        line = row.geometry
        side = row['side']
        id = row['id']

        # point is [intersection_point, max_height, building['identificatie']]
        point, total = get_intersection(line, buildings)

        if point is not None:
            # cs_halvex.loc[_, 'build1'] = point
            # cs_halvex.loc[_, 'height1'] = point.z
            closest_intersections_list.append([id, side, point[0], point[1]])
            for intersection in total:
                intersections_list.append([id,side,intersection])


    # cs_halvex.to_file(cs_halvex_shp)
    gdf = gpd.GeoDataFrame(intersections_list, columns=["id","side","geometry"])
    gdf.set_crs("EPSG:28992", inplace=True)
    gdf.to_file(intersection_output_file, driver="ESRI Shapefile")

    gdf_clos = gpd.GeoDataFrame(closest_intersections_list, columns=['id', 'side', 'geometry', 'height'])
    gdf_clos.set_crs("EPSG:28992", inplace=True)
    gdf_clos.to_file(closest_intersection_output_file, driver="ESRI Shapefile")

    return

def get_intersection(line, buildings):
    # Get intersecting buildings directly using spatial operation
    intersecting_buildings = buildings[buildings.intersects(line)]

    if len(intersecting_buildings) == 0:
        return None, []

    intersections = []
    total_intersections = []
    for idx, building in intersecting_buildings.iterrows():
        try:
            intersection = line.intersection(building.geometry)
            total_intersections.append(intersection)

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

            # max_height = get_max_height(building.geometry)
            max_height = building['b3_h_max']
            intersections.append([intersection_point, max_height, building['identificatie']])

        except Exception as e:
            print(f"Error processing building {idx}: {e}")

    if intersections:
        # Use the actual line for projection
        closest_intersection = min(intersections, key=lambda x: line.project(x[0]))
        # print(f"intersections {intersections}, {closest_intersection[0]}, {total_intersections}")
        return  closest_intersection, total_intersections
    return  None, []

def get_max_height(geometry):
    if isinstance(geometry, Polygon):
        return np.max([coord[2] for coord in geometry.exterior.coords])
    elif isinstance(geometry, MultiPolygon):
        return np.max([np.max([coord[2] for coord in poly.exterior.coords]) for poly in geometry.geoms])
    else:
        raise ValueError(f"Unexpected geometry type: {type(geometry)}")


def run_parameters(river, city):
    """
    Executes all parameter extraction functions: landuse, flood depth, dsm elevation, imperviousness, visibility.
    Adds them to the points shapefile.
    Creates the unique landuses .csv file.

    Args:
        river: String name of river
        city: String name of city

    Returns: -

    """
    # run_data_retrieval(river, city)

    points_shp = f"output/cross_sections/{river}/{city}/final/points.shp"
    cross_sections_shp = f"output/cross_sections/{river}/{city}/final/cross_sections.shp"
    halves_shp = f"output/cross_sections/{river}/{city}/final/halves.shp"
    embankment_shp = f"output/embankment/{river}/{city}/embankments.shp"
    midpoints_shp = f"output/river/{river}/{city}/{city}_mid.shp"

    viewshed_file = f'output/visibility/{river}/{city}/combined_viewshed.tif'
    tiles_folder_dsm = f"input/AHN/{river}/{city}/DSM"
    bgt_folder = f"input/BGT/{river}/{city}"
    unique_landuses_output = f"output/unique_landuses/{city}.csv"
    os.makedirs('output/unique_landuses', exist_ok=True)
    flood_folder = "input/flood/middelgrote_kans"
    imperv_raster = f"input/imperviousness/MERGED_reproj_28992.tif"

    # PARAMETERS
    add_landuse_to_shapefile(points_shp, bgt_folder)
    extract_unique_landuses(points_shp, unique_landuses_output)

    add_elevation_from_tiles(points_shp,flood_folder , 'flood_dept')
    add_elevation_from_tiles(points_shp, tiles_folder_dsm, 'elev_dsm')

    add_raster_column(shapefile_path=points_shp, raster_path=imperv_raster, column_name='imperv')

    # VISIBILITY
    viewshed_dir= f'output/visibility/{river}/{city}/viewsheds'
    visibility_dir = f'output/visibility/{river}/{city}'
    output_file = os.path.join(visibility_dir, 'combined_viewshed.tif')

    # from visibility import combine_viewsheds
    # combine_viewsheds(viewshed_dir, output_file)
    add_raster_column(shapefile_path=points_shp, raster_path=viewshed_file, column_name='visible')

    # SPLIT
    split_points_identify_river(points_shp, embankment_shp, cross_sections_shp, midpoints_shp)
    split_cross_sections(midpoints_shp, cross_sections_shp, halves_shp)

    # BUILDINGS
    intersections = f"output/buildings/{river}/{city}/building_intersections"
    os.makedirs(intersections, exist_ok=True)
    closest_intersections = f"output/buildings/{river}/{city}/closest_building_intersections"
    os.makedirs(closest_intersections, exist_ok=True)
    buildings = f'input/3DBAG/{river}/{city}/{city}/combined_tiles/combined.gpkg'
    building_parameters(halves_shp, buildings, intersections, closest_intersections)


if __name__ == "__main__":
    run_parameters('dommel', 'eindhoven')