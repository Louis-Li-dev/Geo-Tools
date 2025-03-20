import os
import json
import requests
import pickle
from shapely.geometry import shape, Point
from shapely.prepared import prep
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Union, List

class CountryLocator:
    def __init__(self, 
                 geojson_url: str = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson",
                 cache_path: str = None,
                 lookup_cache_path: str = None,
                 auto_install: bool = True,
                 auto_save_lookup_cache: bool = False,
                 verbose: bool = False):
        """
        Downloads GeoJSON data and prepares the country geometries.
        
        Parameters:
            geojson_url (str): URL to the GeoJSON file containing country data.
            cache_path (str): Local path to store/load the geojson data.
            lookup_cache_path (str): Path to store/load the persistent lookup cache for coordinate queries.
            auto_install (bool): If True, download and save the data if cache_path is provided but the file does not exist.
            auto_save_lookup_cache (bool): If True, automatically flush the lookup cache to disk after lookups.
            verbose (bool): If True, print verbose status messages.
        """
        self.verbose = verbose
        self.lookup_cache_path = lookup_cache_path
        self.auto_save_lookup_cache = auto_save_lookup_cache

        # Load GeoJSON data
        if cache_path is not None:
            if os.path.exists(cache_path):
                if self.verbose:
                    print(f"[Verbose] Cache file found at {cache_path}. Loading data from file.")
                with open(cache_path, 'r') as f:
                    data = json.load(f)
            else:
                if auto_install:
                    if self.verbose:
                        print(f"[Verbose] Cache file not found at {cache_path}. Downloading data from {geojson_url} and saving to cache.")
                    data = requests.get(geojson_url).json()
                    with open(cache_path, 'w') as f:
                        json.dump(data, f)
                else:
                    raise FileNotFoundError(f"Cache file not found at {cache_path} and auto_install is disabled.")
        else:
            if self.verbose:
                print("[Verbose] No cache path provided. Downloading data from URL.")
            data = requests.get(geojson_url).json()

        # Prepare country geometries.
        self.countries = {}
        for feature in data["features"]:
            geom = feature["geometry"]
            country = feature["properties"]["ADMIN"]
            self.countries[country] = prep(shape(geom))
        print(f"Loaded {len(self.countries)} countries.")

        # Initialize lookup cache
        if self.lookup_cache_path is not None and os.path.exists(self.lookup_cache_path):
            if self.verbose:
                print(f"[Verbose] Lookup cache found at {self.lookup_cache_path}. Loading lookup cache.")
            with open(self.lookup_cache_path, 'rb') as f:
                self._cache = pickle.load(f)
        else:
            if self.verbose and self.lookup_cache_path is not None:
                print(f"[Verbose] No lookup cache found at {self.lookup_cache_path}. Initializing new lookup cache.")
            self._cache = {}

    def flush_cache(self):
        """
        Flushes the in-memory lookup cache to disk if a lookup_cache_path was provided.
        """
        if self.lookup_cache_path is not None:
            with open(self.lookup_cache_path, 'wb') as f:
                pickle.dump(self._cache, f)
            if self.verbose:
                print(f"[Verbose] Lookup cache flushed to {self.lookup_cache_path}.")

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

    def lookup(self, 
               coords: Union[List, np.ndarray, pd.DataFrame], 
               use_tqdm: bool = True, 
               lat_col: str = "lat", 
               lon_col: str = "lon") -> Union[str, List[str]]:
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
            Union[str, List[str]]:
                - A single country name if one coordinate is provided.
                - A list of country names if multiple coordinates are provided.
                
        Examples:
        ```python
        locator = CountryLocator(lookup_cache_path="lookup_cache.pkl", auto_save_lookup_cache=True)
        df['Country'] = locator.lookup(df, lat_col='Latitude', lon_col="Longitude")
        ```
        """
        result = None
        
        # Handle a pandas DataFrame
        if isinstance(coords, pd.DataFrame):
            if lat_col not in coords.columns or lon_col not in coords.columns:
                raise ValueError(f"DataFrame must contain columns '{lat_col}' and '{lon_col}'.")
            coords_list = list(zip(coords[lon_col], coords[lat_col]))
            iterable = tqdm(coords_list, desc="Processing coordinates") if use_tqdm else coords_list
            result = [self.get_country(lon, lat) for lon, lat in iterable]
        
        # Handle a single coordinate provided as a list or tuple
        elif isinstance(coords, (list, tuple)):
            if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
                result = self.get_country(coords[0], coords[1])
            elif all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in coords):
                iterable = tqdm(coords, desc="Processing coordinates") if use_tqdm else coords
                result = [self.get_country(lon, lat) for lon, lat in iterable]
            else:
                raise ValueError("For list input, provide either a coordinate pair [lon, lat] or a list of coordinate pairs.")
        
        # Handle numpy arrays
        elif isinstance(coords, np.ndarray):
            if coords.ndim == 1:
                if coords.shape[0] == 2:
                    result = self.get_country(coords[0], coords[1])
                else:
                    raise ValueError("A single coordinate numpy array must have shape (2,).")
            elif coords.ndim == 2:
                if coords.shape[1] != 2:
                    raise ValueError("A numpy array of coordinates must have shape (n, 2).")
                iterable = tqdm(coords, desc="Processing coordinates") if use_tqdm else coords
                result = [self.get_country(lon, lat) for lon, lat in iterable]
            else:
                raise ValueError("Invalid numpy array shape for coordinates.")
        else:
            raise TypeError("Unsupported input type for coordinates. Provide a coordinate pair, a list of coordinate pairs, a numpy array of shape (n, 2), or a pandas DataFrame.")
        
        # If auto-saving is enabled, flush the lookup cache after processing.
        if self.lookup_cache_path is not None and self.auto_save_lookup_cache:
            self.flush_cache()

        return result

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
