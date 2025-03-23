import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import warnings
from typing import Optional, Union, List, Tuple
import numpy as np
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

class CoordinatePlotter:
    """
    A flexible plotter class to visualize geographical coordinates on a map with subplot support.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'Latitude' and 'Longitude' columns.
        uid (Optional[int]): User ID for filtering the data frame. Default is None.
        year (Optional[int]): Year for filtering the data frame. Default is None.
        num_of_samples (Optional[int]): Limit the number of samples plotted. Default is None (no limit).
        time_columns (Optional[List[str]]): Columns used for sorting data by time. Default is None.
        margin (float): Margin for map boundary in degrees. Default is 2.
        figsize (tuple): Figure size for plotting. Default is (10, 6).

    Example:
        plotter = CoordinatePlotter(df=my_dataframe)
        plotter.plot_multiple(subplot_layout=(1, 2), plot_configs=[
            {'title': 'Plot 1', 'point_color': 'blue'},
            {'title': 'Plot 2', 'point_color': 'green'}
        ])
    """

    def __init__(self,
                 df: pd.DataFrame,
                 uid: Optional[int] = None,
                 year: Optional[int] = None,
                 num_of_samples: Optional[int] = None,
                 time_columns: Optional[Union[str, List[str]]] = None,
                 margin: float = 2,
                 figsize: tuple = (10, 6)):

        self.df = df
        self.uid = uid
        self.year = year
        self.num_of_samples = num_of_samples
        self.time_columns = [time_columns] if isinstance(time_columns, str) else time_columns
        self.margin = margin
        self.figsize = figsize

        self._ensure_correct_types()
        self.filtered_df = self._filter_and_sort_data()

        if self.filtered_df.empty:
            raise ValueError("Filtered DataFrame is empty. Check your filtering parameters (uid, year, etc.) and ensure the data meets your criteria.")

        self.min_lat, self.max_lat, self.min_lon, self.max_lon = self._calculate_bounds()

    def _ensure_correct_types(self):
        # [Unchanged from original]
        for col in ['Latitude', 'Longitude']:
            if col in self.df.columns and not pd.api.types.is_numeric_dtype(self.df[col]):
                warnings.warn(f"Column '{col}' is not numeric. Attempting to convert.")
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        if self.uid is not None and 'Uid' in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df['Uid']):
                warnings.warn("Column 'Uid' is not numeric. Attempting to convert.")
                self.df['Uid'] = pd.to_numeric(self.df['Uid'], errors='coerce')

        if self.time_columns:
            for col in self.time_columns:
                if col in self.df.columns and not (pd.api.types.is_numeric_dtype(self.df[col]) or pd.api.types.is_datetime64_any_dtype(self.df[col])):
                    warnings.warn(f"Column '{col}' is not numeric or datetime. Attempting to convert.")
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    except:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        for col in ['timestamp', 'datetime']:
            if col in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                warnings.warn(f"Column '{col}' is not datetime. Attempting to convert.")
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

    def _filter_and_sort_data(self) -> pd.DataFrame:
        # [Unchanged from original]
        filtered_df = self.df.copy()

        if self.uid is not None:
            filtered_df = filtered_df[filtered_df['Uid'] == self.uid]

        if self.year is not None:
            filtered_df = filtered_df[filtered_df['Year'] == self.year]

        if self.time_columns:
            filtered_df = filtered_df.sort_values(by=self.time_columns)
        elif 'timestamp' in filtered_df.columns or 'datetime' in filtered_df.columns:
            sort_col = 'timestamp' if 'timestamp' in filtered_df.columns else 'datetime'
            filtered_df = filtered_df.sort_values(by=sort_col)

        filtered_df = filtered_df.drop_duplicates(subset=['Latitude', 'Longitude'])

        if self.num_of_samples:
            filtered_df = filtered_df.iloc[:self.num_of_samples]

        return filtered_df

    def _calculate_bounds(self) -> tuple:
        # [Unchanged from original]
        min_lat = self.filtered_df['Latitude'].min() - self.margin
        max_lat = self.filtered_df['Latitude'].max() + self.margin
        min_lon = self.filtered_df['Longitude'].min() - self.margin
        max_lon = self.filtered_df['Longitude'].max() + self.margin
        return min_lat, max_lat, min_lon, max_lon

    def _plot_single(self, ax: plt.Axes, **kwargs):
        """Helper method to plot on a single Axes object."""
        # Extract parameters with defaults
        title = kwargs.get('title', "Coordinate Plotter")
        xlabel = kwargs.get('xlabel', "Longitude")
        ylabel = kwargs.get('ylabel', "Latitude")
        legend_loc = kwargs.get('legend_loc', "lower left")
        point_color = kwargs.get('point_color', 'red')
        point_edge_color = kwargs.get('point_edge_color', 'darkred')
        point_size = kwargs.get('point_size', 30)
        land_color = kwargs.get('land_color', 'beige')
        water_color = kwargs.get('water_color', 'lightblue')
        coastline_width = kwargs.get('coastline_width', 0.1)
        country_width = kwargs.get('country_width', 0.2)
        overlay_density_background = kwargs.get('overlay_density_background', False)
        density_levels = kwargs.get('density_levels', 20)
        density_cmap = kwargs.get('density_cmap', 'viridis')
        density_based_alpha = kwargs.get('density_based_alpha', False)
        min_alpha = kwargs.get('min_alpha', 0.3)
        max_alpha = kwargs.get('max_alpha', 1.0)
        bw_method = kwargs.get('bw_method', None)

        # Create Basemap instance
        m = Basemap(projection='cyl',
                    llcrnrlat=self.min_lat, urcrnrlat=self.max_lat,
                    llcrnrlon=self.min_lon, urcrnrlon=self.max_lon,
                    resolution='i', ax=ax)

        m.drawcoastlines(linewidth=coastline_width)
        m.drawcountries(linewidth=country_width)
        m.drawmapboundary(fill_color=water_color)
        m.fillcontinents(color=land_color, lake_color=water_color)

        # Get coordinate arrays
        lons = self.filtered_df['Longitude'].values
        lats = self.filtered_df['Latitude'].values
        x, y = m(lons, lats)

        # Overlay density background if enabled
        if overlay_density_background:
            coords = np.vstack([lons, lats])
            kde = gaussian_kde(coords, bw_method=bw_method)
            X, Y = np.mgrid[self.min_lon:self.max_lon:200j, self.min_lat:self.max_lat:200j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kde(positions), X.shape)
            cs = ax.contourf(X, Y, Z, levels=density_levels, cmap=density_cmap, alpha=0.6)
            plt.colorbar(cs, ax=ax, label="Density")

        # Handle density-based alpha if enabled
        if density_based_alpha:
            coords = np.vstack([lons, lats])
            kde = gaussian_kde(coords, bw_method=bw_method)
            densities = kde(coords)
            dens_norm = (densities - densities.min()) / (densities.max() - densities.min())
            alphas = min_alpha + (max_alpha - min_alpha) * dens_norm
            base_rgba = mcolors.to_rgba(point_color)
            point_colors = [(base_rgba[0], base_rgba[1], base_rgba[2], a) for a in alphas]
            scatter_kwargs = {'c': point_colors, 's': point_size, 'marker': 'o', 'edgecolors': point_edge_color}
        else:
            scatter_kwargs = {'color': point_color, 's': point_size, 'marker': 'o', 'edgecolors': point_edge_color}

        # Plot the scatter points
        ax.scatter(x, y, **scatter_kwargs, label="Locations")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc=legend_loc)

    def plot(self, **kwargs):
        """Plots a single map."""
        plt.rcParams['font.family'] = "Times New Roman"
        fig, ax = plt.subplots(figsize=self.figsize)
        self._plot_single(ax, **kwargs)
        plt.show()

    def plot_multiple(self,
                      subplot_layout: Tuple[int, int],
                      plot_configs: List[dict],
                      tight_layout: bool = True):
        """
        Plots multiple maps in a single figure with specified subplot layout.

        Parameters:
            subplot_layout (Tuple[int, int]): Tuple of (rows, cols) for subplot grid.
            plot_configs (List[dict]): List of dictionaries with plot parameters for each subplot.
            tight_layout (bool): If True, applies tight layout to prevent overlap.

        Example:
            plotter.plot_multiple(subplot_layout=(1, 2), plot_configs=[
                {'title': 'Plot 1', 'point_color': 'blue'},
                {'title': 'Plot 2', 'point_color': 'green'}
            ])
        """
        if len(plot_configs) > subplot_layout[0] * subplot_layout[1]:
            raise ValueError("Number of plot configurations exceeds the subplot grid size.")

        plt.rcParams['font.family'] = "Times New Roman"
        fig, axes = plt.subplots(*subplot_layout, figsize=self.figsize)
        
        # Handle single subplot case
        if subplot_layout[0] * subplot_layout[1] == 1:
            axes = np.array([axes])

        # Flatten axes array for iteration
        axes_flat = axes.flat if subplot_layout[0] * subplot_layout[1] > 1 else [axes]

        # Plot each configuration
        for i, config in enumerate(plot_configs):
            self._plot_single(axes_flat[i], **config)

        # Hide unused subplots
        for j in range(len(plot_configs), len(axes_flat)):
            axes_flat[j].set_visible(False)

        if tight_layout:
            plt.tight_layout()
        plt.show()
