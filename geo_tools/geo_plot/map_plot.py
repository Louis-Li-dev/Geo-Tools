import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import warnings
from typing import Optional, Union, List
import numpy as np
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
class CoordinatePlotter:
    """
    A flexible plotter class to visualize geographical coordinates on a map.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'Latitude' and 'Longitude' columns.
        uid (Optional[int]): User ID for filtering the data frame. Default is None.
        year (Optional[int]): Year for filtering the data frame. Default is None.
        num_of_samples (Optional[int]): Limit the number of samples plotted. Default is None (no limit).
        time_columns (Optional[List[str]]): Columns used for sorting data by time. Default is None.
        margin (float): Margin for map boundary in degrees. Default is 2.
        figsize (tuple): Figure size for plotting. Default is (10, 6).

    Example:
        plotter = CoordinatePlotter(df=my_dataframe, uid=3, year=2009, num_of_samples=100, 
                                    time_columns=['Month', 'Day', 'Hour', 'Minute', 'Second'])
        plotter.plot()
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
        """
        Ensures 'Latitude', 'Longitude', and optionally 'Uid', 'Year', and time-related columns are numeric or datetime.
        Warns the user and attempts conversion if incorrect types are found.
        """
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
        """
        Filters the DataFrame based on provided UID and year criteria, then sorts by time columns if provided.

        Returns:
            pd.DataFrame: A filtered and sorted DataFrame ready for plotting.
        """
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
        """
        Calculates latitude and longitude bounds for the map plot.

        Returns:
            tuple: Boundaries (min_lat, max_lat, min_lon, max_lon).
        """
        min_lat = self.filtered_df['Latitude'].min() - self.margin
        max_lat = self.filtered_df['Latitude'].max() + self.margin
        min_lon = self.filtered_df['Longitude'].min() - self.margin
        max_lon = self.filtered_df['Longitude'].max() + self.margin
        return min_lat, max_lat, min_lon, max_lon

    def plot(self,
             title: str = "Coordinate Plotter",
             xlabel: str = "Longitude",
             ylabel: str = "Latitude",
             legend_loc: str = "lower left",
             point_color: str = 'red',
             point_edge_color: str = 'darkred',
             point_size: int = 30,
             land_color: str = 'beige',
             water_color: str = 'lightblue',
             coastline_width: float = 0.1,
             country_width: float = 0.2,
             overlay_density_background: bool = False,  # New option for KDE background
             density_levels: int = 20,
             density_cmap: str = 'viridis',
             density_based_alpha: bool = False,         # New option for per-point alpha based on density
             min_alpha: float = 0.3,
             max_alpha: float = 1.0):
        """
        Plots geographical coordinates on a map.

        Additional parameters:
            overlay_density_background (bool): If True, overlays a Gaussian KDE background.
            density_levels (int): Number of contour levels for the density plot.
            density_cmap (str): Colormap used for the density plot.
            density_based_alpha (bool): If True, vary the alpha of each point based on its local density.
            min_alpha (float): Minimum alpha value for the lowest density.
            max_alpha (float): Maximum alpha value for the highest density.
        """
        plt.rcParams['font.family'] = "Times New Roman"
        fig, ax = plt.subplots(figsize=self.figsize)

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

        # If density background overlay is enabled, compute and plot KDE contours.
        if overlay_density_background:
            # Compute KDE in longitude-latitude space
            coords = np.vstack([lons, lats])
            kde = gaussian_kde(coords)
            # Define grid boundaries based on data (with some margin)
            X, Y = np.mgrid[self.min_lon:self.max_lon:200j, self.min_lat:self.max_lat:200j]

            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kde(positions), X.shape)
            # Plot density contours in the background.
            # Note: the contour is plotted in lon/lat space and aligns with the map.
            cs = ax.contourf(X, Y, Z, levels=density_levels, cmap=density_cmap, alpha=0.6)
            plt.colorbar(cs, ax=ax, label="Density")

        # If density-based alpha is enabled, compute density per point and adjust alpha.
        if density_based_alpha:
            coords = np.vstack([lons, lats])
            kde = gaussian_kde(coords)
            densities = kde(coords)
            # Normalize densities to [0,1]
            dens_norm = (densities - densities.min()) / (densities.max() - densities.min())
            # Map normalized density to alpha range
            alphas = min_alpha + (max_alpha - min_alpha) * dens_norm
            # Create a list of RGBA colors for each point based on the provided point_color.
            base_rgba = mcolors.to_rgba(point_color)
            point_colors = [(base_rgba[0], base_rgba[1], base_rgba[2], a) for a in alphas]
            scatter_kwargs = {'c': point_colors, 's': point_size, 'marker': 'o', 'edgecolors': point_edge_color}
        else:
            # Use a uniform color and alpha for all points.
            scatter_kwargs = {'color': point_color, 's': point_size, 'marker': 'o', 'edgecolors': point_edge_color}

        # Plot the scatter points on the map.
        ax.scatter(x, y, **scatter_kwargs, label="Locations")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc=legend_loc)
        plt.show()