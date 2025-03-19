import requests
from shapely.geometry import shape, Point
from shapely.prepared import prep
import numpy as np
from tqdm import tqdm
from typing import Union, List
import pandas as pd

class CountryLocator:
    def __init__(self, geojson_url: str = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"):
        """
        Downloads GeoJSON data and prepares the country geometries.
        
        Parameters:
            geojson_url (str): URL to the GeoJSON file containing country data.
        """
        data = requests.get(geojson_url).json()
        self.countries = {}
        for feature in data["features"]:
            geom = feature["geometry"]
            country = feature["properties"]["ADMIN"]
            self.countries[country] = prep(shape(geom))
        print(f"Loaded {len(self.countries)} countries.")
        self._cache = {}  # Cache for storing coordinate lookup results

    def get_country(self, lon: float, lat: float) -> str:
        """
        Returns the country that contains the point defined by (lon, lat), using cache if available.
        
        Parameters:
            lon (float): Longitude of the point.
            lat (float): Latitude of the point.
            
        Returns:
            str: Country name if found, otherwise "unknown".
        """
        key = (lon, lat)
        if key in self._cache:
            return self._cache[key]

        point = Point(lon, lat)
        for country, geom in self.countries.items():
            if geom.contains(point):
                self._cache[key] = country
                return country
        self._cache[key] = "unknown"
        return "unknown"

    def lookup(self, coords: Union[List, np.ndarray, pd.DataFrame], use_tqdm: bool = True, lat_col: str = "lat", lon_col: str = "lon") -> Union[str, List[str]]:
        """
        Looks up country names for one or more coordinates, using caching and an optional tqdm progress bar.
        
        Parameters:
            coords (Union[List, np.ndarray, pd.DataFrame]):
                - A single coordinate as a tuple or list of two numbers [lon, lat].
                - A list of coordinates, where each coordinate is a tuple or list of two numbers.
                - A NumPy array of shape (n, 2) where each row is [lon, lat].
                - A pandas DataFrame containing columns for latitude and longitude.
            use_tqdm (bool): If True, displays a progress bar when processing multiple coordinates.
            lat_col (str): The column name for latitude when passing a DataFrame.
            lon_col (str): The column name for longitude when passing a DataFrame.
                
        Returns:
            Union(str, List[str]):
                - A single country name if one coordinate is provided.
                - A list of country names if multiple coordinates are provided.
        Examples:
        ```python
        from geo_plot.map_plot import CoordinatePlotter
        locator = coordinate_lookup.CountryLocator()
        df['Country'] = locator.lookup(df, lat_col='Latitude', lon_col="Longitude")
        # Make sure Latitude and Longitude are in the Pandas Data Frame
        # Return the corresponding countries. 
        ```
        """
        # Handle a pandas DataFrame
        if isinstance(coords, pd.DataFrame):
            if lat_col not in coords.columns or lon_col not in coords.columns:
                raise ValueError(f"DataFrame must contain columns '{lat_col}' and '{lon_col}'.")
            coords_list = list(zip(coords[lon_col], coords[lat_col]))
            iterable = tqdm(coords_list, desc="Processing coordinates") if use_tqdm else coords_list
            return [self.get_country(lon, lat) for lon, lat in iterable]
        
        # Handle a single coordinate provided as a list or tuple
        if isinstance(coords, (list, tuple)):
            if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
                return self.get_country(coords[0], coords[1])
            elif all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in coords):
                iterable = tqdm(coords, desc="Processing coordinates") if use_tqdm else coords
                return [self.get_country(lon, lat) for lon, lat in iterable]
            else:
                raise ValueError("For list input, provide either a coordinate pair [lon, lat] or a list of coordinate pairs.")
        
        # Handle numpy arrays
        elif isinstance(coords, np.ndarray):
            if coords.ndim == 1:
                if coords.shape[0] == 2:
                    return self.get_country(coords[0], coords[1])
                else:
                    raise ValueError("A single coordinate numpy array must have shape (2,).")
            elif coords.ndim == 2:
                if coords.shape[1] != 2:
                    raise ValueError("A numpy array of coordinates must have shape (n, 2).")
                iterable = tqdm(coords, desc="Processing coordinates") if use_tqdm else coords
                return [self.get_country(lon, lat) for lon, lat in iterable]
            else:
                raise ValueError("Invalid numpy array shape for coordinates.")
        
        else:
            raise TypeError("Unsupported input type for coordinates. Provide a coordinate pair, a list of coordinate pairs, a numpy array of shape (n, 2), or a pandas DataFrame.")

# Example Usage:
if __name__ == "__main__":
    locator = CountryLocator()
    
    # Single coordinate as a tuple:
    print("Single coordinate:", locator.lookup((10.0, 47.0)))  # Expected output: Austria
    
    # List of coordinates:
    coords_list = [(10.0, 47.0), (2.3522, 48.8566)]
    print("List of coordinates:", locator.lookup(coords_list))
    
    # NumPy array of coordinates:
    coords_array = np.array([[10.0, 47.0], [2.3522, 48.8566]])
    print("NumPy array of coordinates:", locator.lookup(coords_array))
    
    # Pandas DataFrame of coordinates:
    df = pd.DataFrame({
        'lon': [10.0, 2.3522],
        'lat': [47.0, 48.8566]
    })
    print("DataFrame of coordinates:", locator.lookup(df, lat_col="lat", lon_col="lon"))
