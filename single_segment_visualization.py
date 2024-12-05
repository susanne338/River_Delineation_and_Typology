"""
getsel section 12
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

def select_points_by_id(gdf, section_id):
    """
    Select points with a given section ID.

    Args:
        gdf (GeoDataFrame): Geopandas DataFrame containing the shapefile data.
        section_id (int): The ID of the section to filter by.

    Returns:
        GeoDataFrame: Filtered GeoDataFrame with points matching the section ID.
    """
    return gdf[gdf['id'] == section_id]

def add_plot_index(points):
    """
    Add a sequential index for better spacing on the x-axis in plots.

    Args:
        points (GeoDataFrame): Points filtered by section ID.

    Returns:
        GeoDataFrame: GeoDataFrame with a new 'plot_index' column.
    """
    points = points.copy()
    points['plot_index'] = np.arange(len(points))
    return points

def plot_elevation(points):
    """
    Plot 'elev_dsm' and 'elev_dtm' with correct aspect ratio.

    Args:
        points (GeoDataFrame): Points filtered by section ID.
    """
    plt.figure()
    plt.plot(points['plot_index'], points['elev_dsm'], label='DSM Elevation', color='blue')
    plt.plot(points['plot_index'], points['elev_dtm'], label='DTM Elevation', color='green')
    plt.xlabel('Point Index')
    plt.ylabel('Elevation')
    plt.title('Elevation Plot')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_landuse(points):
    """
    Plot 'elev_dtm' with colors based on 'landuse'.

    Args:
        points (GeoDataFrame): Points filtered by section ID.
    """
    unique_landuses = points['landuse'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_landuses)))
    landuse_color_map = {landuse: color for landuse, color in zip(unique_landuses, colors)}

    plt.figure()
    for landuse, color in landuse_color_map.items():
        subset = points[points['landuse'] == landuse]
        plt.scatter(subset['plot_index'], subset['elev_dtm'], label=landuse, color=color)
    plt.xlabel('Point Index')
    plt.ylabel('Elevation (DTM)')
    plt.title('Landuse vs Elevation (DTM)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_flood_depth(points):
    """
    Plot 'flood_depth' and 'elev_dtm', excluding 'flood_depth' values of -9999.

    Args:
        points (GeoDataFrame): Points filtered by section ID.
    """
    # Exclude points where 'flood_depth' is -9999
    valid_points = points[points['flood_dept'] != -9999]

    plt.figure()
    plt.scatter(valid_points['plot_index'], valid_points['flood_dept'], label='Flood Depth', color='red', marker='o')
    plt.scatter(points['plot_index'], points['elev_dtm'], label='DTM Elevation', color='green', marker='x')
    plt.xlabel('Point Index')
    plt.ylabel('Value')
    plt.title('Flood Depth and Elevation (DTM)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
def plot_visibility(points):
    """
    Plot visibility colored by 'visible' status.

    Args:
        points (GeoDataFrame): Points filtered by section ID.
    """
    plt.figure()
    visible = points[points['visible'] == 1]
    not_visible = points[points['visible'] != 1]

    plt.scatter(visible['plot_index'], visible['elev_dsm'], color='blue', label='Visible')
    plt.scatter(not_visible['plot_index'], not_visible['elev_dsm'], color='orange', label='Not Visible')
    plt.xlabel('Point Index')
    plt.ylabel('Elevation (DSM)')
    plt.title('Visibility Plot')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def main():
    # Load the shapefile
    shapefile_path = "output/cross_sections/maas/maastricht/final/points.shp/points.shp"
    gdf = gpd.read_file(shapefile_path)

    # Specify the section ID to filter
    section_id = 28  # Replace with your desired section ID

    # Filter points by section ID
    selected_points = select_points_by_id(gdf, section_id)

    # Add a plot index for better spacing
    selected_points = add_plot_index(selected_points)

    # Generate plots
    plot_elevation(selected_points)
    plot_landuse(selected_points)
    plot_flood_depth(selected_points)
    plot_visibility(selected_points)

if __name__ == "__main__":
    main()
