import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_spider_charts(stats_path, results_path, centers_path, output_path):
    """
    Create separate spider charts for general metrics and landuse metrics
    """
    # Load data
    stats = pd.read_csv(stats_path)
    results = pd.read_csv(results_path)
    centers = pd.read_csv(centers_path)

    # Separate features into general and landuse
    features = stats['feature'].unique()
    landuse_features = [f for f in features if f.startswith('landuse_')]
    general_features = [f for f in features if not f.startswith('landuse_')]

    # Get unique clusters
    n_clusters = len(stats['cluster'].unique())

    def prepare_spider_data(features, angles, values):
        """Helper function to prepare data for spider chart"""
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        return values, angles

    def create_spider_subplot(ax, values, angles, feature_names, title, color):
        """Helper function to create a spider subplot"""
        # Calculate appropriate scale for this set of values
        max_val = np.max(values)
        min_val = np.min(values)

        # Create rounded scale with some padding
        scale_max = np.ceil(max_val * 1.2 * 10) / 10
        scale_min = np.floor(min_val * 0.8 * 10) / 10

        values, angles = prepare_spider_data(feature_names, angles, values)

        ax.plot(angles, values, 'o-', linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_title(title, pad=15)

        # Set chart properties
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, size=8)

        # Set ylim based on the calculated scale
        ax.set_ylim(scale_min, scale_max)

        # Add gridlines at specific intervals
        n_circles = 5
        intervals = np.linspace(scale_min, scale_max, n_circles)
        ax.set_rticks(intervals)
        ax.set_rlabel_position(0)

        # Add circular gridlines
        for interval in intervals:
            circle = plt.Circle((0, 0), interval, transform=ax.transData._b,
                                fill=False, color='gray', alpha=0.1)
            ax.add_artist(circle)

        return scale_min, scale_max

    # Color palette for clusters
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    # Create charts for each cluster
    for cluster in range(n_clusters):
        cluster_stats = stats[stats['cluster'] == cluster]

        # Create figure with two subplots
        fig = plt.figure(figsize=(20, 10))

        # General metrics spider chart
        general_means = cluster_stats[cluster_stats['feature'].isin(general_features)]['mean'].values
        general_angles = [n / float(len(general_features)) * 2 * np.pi for n in range(len(general_features))]

        ax1 = fig.add_subplot(121, projection='polar')
        scale_min_gen, scale_max_gen = create_spider_subplot(
            ax1, general_means, general_angles, general_features,
            f'Cluster {cluster}: General Metrics', colors[cluster]
        )

        # Landuse metrics spider chart
        landuse_means = cluster_stats[cluster_stats['feature'].isin(landuse_features)]['mean'].values
        landuse_angles = [n / float(len(landuse_features)) * 2 * np.pi for n in range(len(landuse_features))]

        ax2 = fig.add_subplot(122, projection='polar')
        scale_min_land, scale_max_land = create_spider_subplot(
            ax2, landuse_means, landuse_angles, landuse_features,
            f'Cluster {cluster}: Land Use Metrics', colors[cluster]
        )

        plt.suptitle(f'Characteristics of Cluster {cluster}', size=16, y=1.05)
        plt.tight_layout()
        plt.savefig(f'{output_path}/cluster_{cluster}_spider.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Print representative point information
        representative = cluster_stats.iloc[0]
        print(f"\nCluster {cluster} Representative Point:")
        print(f"ID: {representative['representative_id']}")
        print(f"Side: {representative['representative_side']}")
        print("\nGeneral metrics:")
        for feature in general_features:
            feat_stats = cluster_stats[cluster_stats['feature'] == feature].iloc[0]
            print(f"  {feature}: {feat_stats['mean']:.2f} ± {feat_stats['std']:.2f}")
        print("\nLand use metrics:")
        for feature in landuse_features:
            feat_stats = cluster_stats[cluster_stats['feature'] == feature].iloc[0]
            print(f"  {feature}: {feat_stats['mean']:.2f} ± {feat_stats['std']:.2f}")


def plot_cluster_comparison(stats_path, output_path):
    """
    Create comparison plots showing all clusters together, separate for general and landuse metrics
    """
    stats = pd.read_csv(stats_path)
    features = stats['feature'].unique()

    # Separate features
    landuse_features = [f for f in features if f.startswith('landuse_')]
    general_features = [f for f in features if not f.startswith('landuse_')]

    # Color palette for clusters
    n_clusters = len(stats['cluster'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    # Create separate comparison plots for general and landuse metrics
    for feature_set, title in [(general_features, 'General Metrics'),
                               (landuse_features, 'Land Use Metrics')]:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='polar')

        # Calculate angles for this feature set
        angles = [n / float(len(feature_set)) * 2 * np.pi for n in range(len(feature_set))]
        angles = np.concatenate((angles, [angles[0]]))

        # Find scale for this set of features
        max_val = max(stats[stats['feature'].isin(feature_set)]['mean'])
        min_val = min(stats[stats['feature'].isin(feature_set)]['mean'])
        scale_max = np.ceil(max_val * 1.2 * 10) / 10
        scale_min = np.floor(min_val * 0.8 * 10) / 10

        # Plot each cluster
        for cluster in stats['cluster'].unique():
            cluster_stats = stats[stats['cluster'] == cluster]
            cluster_stats = cluster_stats[cluster_stats['feature'].isin(feature_set)]
            values = cluster_stats['mean'].values
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}',
                    color=colors[cluster])

        # Set chart properties
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_set, size=8)
        ax.set_ylim(scale_min, scale_max)

        # Add gridlines
        n_circles = 5
        intervals = np.linspace(scale_min, scale_max, n_circles)
        ax.set_rticks(intervals)
        ax.set_rlabel_position(0)

        # Add circular gridlines
        for interval in intervals:
            circle = plt.Circle((0, 0), interval, transform=ax.transData._b,
                                fill=False, color='gray', alpha=0.1)
            ax.add_artist(circle)

        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.title(f'Comparison of Cluster {title}', size=16, pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_path}/cluster_comparison_{title.lower().replace(" ", "_")}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()


# Example usage:
if __name__ == "__main__":
    # Set your paths
    input_dir = "results"
    output_dir = "spider_charts"

    # Create visualizations
    create_spider_charts(
        f"{input_dir}/cluster_statistics.csv",
        f"{input_dir}/clustering_results.csv",
        f"{input_dir}/cluster_centers.csv",
        output_dir
    )

    # Create comparison plot
    plot_cluster_comparison(
        f"{input_dir}/cluster_statistics.csv",
        output_dir
    )