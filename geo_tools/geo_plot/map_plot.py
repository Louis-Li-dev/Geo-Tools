import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import warnings
from typing import Optional, Union, List

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
             country_width: float = 0.2):
        """
        Plots geographical coordinates on a map.

        Parameters:
            title (str): Title of the plot. Default is "Coordinate Plotter".
            xlabel (str): Label for the x-axis (Longitude). Default is "Longitude".
            ylabel (str): Label for the y-axis (Latitude). Default is "Latitude".
            legend_loc (str): Location of the plot legend. Default is "lower left".
            point_color (str): Color of the points plotted. Default is 'red'.
            point_edge_color (str): Edge color of the points plotted. Default is 'darkred'.
            point_size (int): Size of the plotted points. Default is 30.
            land_color (str): Color used for land on the map. Default is 'beige'.
            water_color (str): Color used for water on the map. Default is 'lightblue'.
            coastline_width (float): Width of coastline lines. Default is 0.1.
            country_width (float): Width of country boundary lines. Default is 0.2.

        Returns:
            None: The method displays a plot but does not return any value.
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

        x, y = m(self.filtered_df['Longitude'].values, self.filtered_df['Latitude'].values)
        ax.scatter(x, y, color=point_color, marker='o', edgecolors=point_edge_color,
                   s=point_size, label="Locations")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc=legend_loc)

        plt.show()