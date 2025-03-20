import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CoordinateGrid:
    def __init__(self, cell_size, scaling_method="none", verbose=False):
        """
        Initialize the CoordinateGrid.

        Parameters:
        - cell_size (float): The size of each grid cell (in the effective coordinate space).
        - scaling_method (str): Scaling to apply when fitting the data. Options are:
            "none"     - subtract the global min (centers data to start at 0).
            "minmax"   - min-max normalization: (x - min) / (max - min), so data is in [0,1].
            "standard" - standard scaling: (x - mean) / std, then shifted so that minimum is 0.
        - verbose (bool): If True, prints out the scaling parameters and chosen method.
        """
        self.cell_size = cell_size
        self.scaling_method = scaling_method.lower()
        self.verbose = verbose
        self._scaling_info = None  # To store parameters needed to scale and invert
        self.fitted_data = None    # To store transformed (effective) data
        # For our transformation, the effective global min will be (0,0)
        self.global_min_effective = (0, 0)

    def _apply_scaling(self, x, y):
        """
        Scale the raw coordinate arrays (x, y) according to the selected scaling method.
        Returns the effective coordinates and stores the parameters needed for inversion.
        """
        x = np.array(x)
        y = np.array(y)
        
        if self.scaling_method == "none":
            # Simply center the data by subtracting the minimum values.
            min_x, min_y = np.min(x), np.min(y)
            effective_x = x - min_x
            effective_y = y - min_y
            self._scaling_info = {"method": "none", "min_x": min_x, "min_y": min_y}
            if self.verbose:
                print(f"[Verbose] Scaling method: none. Computed global min: ({min_x}, {min_y}).")
                
        elif self.scaling_method == "minmax":
            # Min-max normalization scales data to [0, 1].
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            range_x = max_x - min_x
            range_y = max_y - min_y
            if range_x == 0 or range_y == 0:
                raise ValueError("Range for x or y is zero. Cannot apply min-max scaling.")
            effective_x = (x - min_x) / range_x
            effective_y = (y - min_y) / range_y
            self._scaling_info = {
                "method": "minmax",
                "min_x": min_x, "min_y": min_y,
                "range_x": range_x, "range_y": range_y
            }
            if self.verbose:
                print(f"[Verbose] Scaling method: minmax. min: ({min_x}, {min_y}), range: ({range_x}, {range_y}).")
                
        elif self.scaling_method == "standard":
            # Standard scaling: (x - mean)/std, then shift so that the minimum becomes 0.
            mean_x, std_x = np.mean(x), np.std(x)
            mean_y, std_y = np.mean(y), np.std(y)
            if std_x == 0 or std_y == 0:
                raise ValueError("Standard deviation for x or y is zero. Cannot apply standard scaling.")
            x_scaled = (x - mean_x) / std_x
            y_scaled = (y - mean_y) / std_y
            shift_x, shift_y = np.min(x_scaled), np.min(y_scaled)
            effective_x = x_scaled - shift_x
            effective_y = y_scaled - shift_y
            self._scaling_info = {
                "method": "standard",
                "mean_x": mean_x, "std_x": std_x,
                "mean_y": mean_y, "std_y": std_y,
                "shift_x": shift_x, "shift_y": shift_y
            }
            if self.verbose:
                print(f"[Verbose] Scaling method: standard. Mean: ({mean_x}, {mean_y}), std: ({std_x}, {std_y}), shift: ({shift_x}, {shift_y}).")
        else:
            raise ValueError("Invalid scaling method. Choose 'none', 'minmax', or 'standard'.")
        
        return effective_x, effective_y

    def _inverse_scaling(self, effective_x, effective_y):
        """
        Inverse the scaling transformation on effective coordinates to recover original values.
        """
        info = self._scaling_info
        method = info["method"]
        
        if method == "none":
            # Original = effective + min
            orig_x = effective_x + info["min_x"]
            orig_y = effective_y + info["min_y"]
        elif method == "minmax":
            # Original = effective * range + min
            orig_x = effective_x * info["range_x"] + info["min_x"]
            orig_y = effective_y * info["range_y"] + info["min_y"]
        elif method == "standard":
            # First, invert the shift, then invert standard scaling: orig = (effective + shift)*std + mean
            orig_x = (effective_x + info["shift_x"]) * info["std_x"] + info["mean_x"]
            orig_y = (effective_y + info["shift_y"]) * info["std_y"] + info["mean_y"]
        else:
            raise ValueError("Invalid scaling method encountered during inverse transformation.")
        return orig_x, orig_y

    def fit(self, df=None, longitudes=None, latitudes=None, lon_col='longitude', lat_col='latitude'):
        """
        Fits the grid to the provided coordinate data by computing the effective coordinates.
        The effective coordinates are computed by applying the chosen scaling transformation.
        This centers the data (so that it starts at 0) and stores the parameters for future inversion.
        The transformed (fitted) data is converted into grid cell indices.

        Parameters:
        - df (pandas.DataFrame, optional): DataFrame containing coordinate columns.
        - longitudes (array-like, optional): Array of x-coordinates.
        - latitudes (array-like, optional): Array of y-coordinates.
        - lon_col (str, optional): Column name in df for longitude (default "longitude").
        - lat_col (str, optional): Column name in df for latitude (default "latitude").

        Returns:
        - The fitted data with grid cell indices (as a DataFrame if input was a DataFrame, else a tuple).
        """
        # Extract raw coordinate arrays.
        if df is not None:
            if lon_col not in df.columns or lat_col not in df.columns:
                raise ValueError(f"DataFrame must contain '{lon_col}' and '{lat_col}' columns.")
            x = df[lon_col].values
            y = df[lat_col].values
        elif longitudes is not None and latitudes is not None:
            x = np.array(longitudes)
            y = np.array(latitudes)
        else:
            raise ValueError("Either a DataFrame or both longitudes and latitudes arrays must be provided.")

        # Apply the selected scaling.
        effective_x, effective_y = self._apply_scaling(x, y)
        # In the effective coordinate space, the global minimum is (0, 0)
        self.global_min_effective = (0, 0)
        # Convert effective coordinates to grid cell indices.
        cell_x = (effective_x // self.cell_size).astype(int)
        cell_y = (effective_y // self.cell_size).astype(int)

        # Store fitted data (both effective coordinates and cell indices).
        self.fitted_data = {"effective_x": effective_x, "effective_y": effective_y,
                             "cell_x": cell_x, "cell_y": cell_y}
        
        if self.verbose:
            print("[Verbose] Fitting complete. Grid cell indices computed.")

        if df is not None:
            new_df = df.copy()
            new_df['cell_x'] = cell_x
            new_df['cell_y'] = cell_y
            return new_df
        else:
            return cell_x, cell_y

    def transform(self, x, y):
        """
        Transforms a single coordinate (x, y) into grid cell indices.
        This method uses the scaling parameters computed during fit().
        
        Parameters:
        - x (float): The x-coordinate.
        - y (float): The y-coordinate.

        Returns:
        - (int, int): The grid cell indices (cell_x, cell_y).

        Raises:
        - ValueError: If fit() has not been called.
        """
        if self._scaling_info is None:
            raise ValueError("You must call fit() first to compute scaling parameters.")
        # Apply the same scaling as in fit.
        if self.scaling_method == "none":
            effective_x = x - self._scaling_info["min_x"]
            effective_y = y - self._scaling_info["min_y"]
        elif self.scaling_method == "minmax":
            effective_x = (x - self._scaling_info["min_x"]) / self._scaling_info["range_x"]
            effective_y = (y - self._scaling_info["min_y"]) / self._scaling_info["range_y"]
        elif self.scaling_method == "standard":
            effective_x = (x - self._scaling_info["mean_x"]) / self._scaling_info["std_x"] - self._scaling_info["shift_x"]
            effective_y = (y - self._scaling_info["mean_y"]) / self._scaling_info["std_y"] - self._scaling_info["shift_y"]
        else:
            raise ValueError("Invalid scaling method encountered.")
        
        cell_x = int(effective_x // self.cell_size)
        cell_y = int(effective_y // self.cell_size)
        return cell_x, cell_y

    def transform_bulk(self, df=None, longitudes=None, latitudes=None, lon_col='longitude', lat_col='latitude'):
        """
        Transforms multiple coordinates into grid cell indices using the computed scaling.
        
        Parameters:
        - df (pandas.DataFrame, optional): DataFrame containing coordinate columns.
        - longitudes (array-like, optional): Array of x-coordinates.
        - latitudes (array-like, optional): Array of y-coordinates.
        - lon_col (str, optional): Column name for longitude (default "longitude").
        - lat_col (str, optional): Column name for latitude (default "latitude").
        
        Returns:
        - If df is provided: a new DataFrame with additional 'cell_x' and 'cell_y' columns.
        - Otherwise: a tuple of arrays (cell_x_array, cell_y_array).
        
        Raises:
        - ValueError: If fit() has not been called.
        """
        if self._scaling_info is None:
            raise ValueError("You must call fit() first to compute scaling parameters.")
        
        if df is not None:
            if lon_col not in df.columns or lat_col not in df.columns:
                raise ValueError(f"DataFrame must contain '{lon_col}' and '{lat_col}' columns.")
            x = df[lon_col].values
            y = df[lat_col].values
        elif longitudes is not None and latitudes is not None:
            x = np.array(longitudes)
            y = np.array(latitudes)
        else:
            raise ValueError("Either a DataFrame or both longitudes and latitudes arrays must be provided.")
        
        # Apply scaling
        if self.scaling_method == "none":
            effective_x = x - self._scaling_info["min_x"]
            effective_y = y - self._scaling_info["min_y"]
        elif self.scaling_method == "minmax":
            effective_x = (x - self._scaling_info["min_x"]) / self._scaling_info["range_x"]
            effective_y = (y - self._scaling_info["min_y"]) / self._scaling_info["range_y"]
        elif self.scaling_method == "standard":
            effective_x = (x - self._scaling_info["mean_x"]) / self._scaling_info["std_x"] - self._scaling_info["shift_x"]
            effective_y = (y - self._scaling_info["mean_y"]) / self._scaling_info["std_y"] - self._scaling_info["shift_y"]
        else:
            raise ValueError("Invalid scaling method encountered.")

        cell_x = (effective_x // self.cell_size).astype(int)
        cell_y = (effective_y // self.cell_size).astype(int)
        
        if df is not None:
            new_df = df.copy()
            new_df['cell_x'] = cell_x
            new_df['cell_y'] = cell_y
            return new_df
        else:
            return cell_x, cell_y

    def inverse_transform(self, cell_x=None, cell_y=None, method="center"):
        """
        Inverse transforms grid cell indices back to original coordinate values.
        This method recovers the effective coordinate from the cell indices and then 
        inverts the scaling transformation.
        
        Parameters:
        - cell_x (int or array-like, optional): Grid cell index/indices for x.
        - cell_y (int or array-like, optional): Grid cell index/indices for y.
        - method (str, optional): "center" (default) returns the center of the cell; 
                                  "corner" returns the lower left corner.
        
        Returns:
        - Tuple of coordinates (x, y) corresponding to the grid cell(s). Scalars are returned 
          if the input is scalar.
        
        Raises:
        - ValueError: If no cell indices are provided and no fitted data exists.
        """
        # If no indices provided, try to use fitted data.
        if cell_x is None or cell_y is None:
            if self.fitted_data is None:
                raise ValueError("No cell indices provided and no fitted data available.")
            cell_x = self.fitted_data["cell_x"]
            cell_y = self.fitted_data["cell_y"]

        cell_x = np.array(cell_x)
        cell_y = np.array(cell_y)
        
        # Recover effective coordinates from grid cell indices.
        if method == "corner":
            effective_x = cell_x * self.cell_size
            effective_y = cell_y * self.cell_size
        elif method == "center":
            effective_x = cell_x * self.cell_size + self.cell_size / 2
            effective_y = cell_y * self.cell_size + self.cell_size / 2
        else:
            raise ValueError("Method must be 'center' or 'corner'.")
        
        # Invert the scaling transformation.
        orig_x, orig_y = self._inverse_scaling(effective_x, effective_y)
        
        if orig_x.size == 1:
            return orig_x.item(), orig_y.item()
        return orig_x, orig_y

    def plot_grid(self, df=None, longitudes=None, latitudes=None, lon_col='longitude', lat_col='latitude',
                  title='Grid Cell Plot', x_label='Grid Cell X', y_label='Grid Cell Y'):
        """
        Plots grid cell indices corresponding to the provided coordinate data.
        The data is transformed (with scaling and centering applied) and converted into grid cell indices.
        A scatter plot is generated with grid lines delineating cell boundaries.
        
        Parameters:
        - df (pandas.DataFrame, optional): DataFrame containing coordinate columns.
        - longitudes (array-like, optional): Array of x-coordinates.
        - latitudes (array-like, optional): Array of y-coordinates.
        - lon_col (str, optional): Column name for longitude (default "longitude").
        - lat_col (str, optional): Column name for latitude (default "latitude").
        - title (str, optional): Plot title.
        - x_label (str, optional): Label for the x-axis.
        - y_label (str, optional): Label for the y-axis.
        
        Raises:
        - ValueError: If neither a DataFrame nor both arrays are provided.
        """
        transformed = self.transform_bulk(df=df, longitudes=longitudes, latitudes=latitudes, 
                                            lon_col=lon_col, lat_col=lat_col)
        if isinstance(transformed, pd.DataFrame):
            cell_x = transformed['cell_x'].values
            cell_y = transformed['cell_y'].values
        else:
            cell_x, cell_y = transformed

        plt.figure(figsize=(8, 6))
        plt.scatter(cell_x, cell_y, c='blue', marker='o', label='Grid Cell')

        # Determine boundaries for grid lines based on cell indices.
        min_cell_x, max_cell_x = int(cell_x.min()), int(cell_x.max())
        min_cell_y, max_cell_y = int(cell_y.min()), int(cell_y.max())

        plt.xlim(min_cell_x - 1, max_cell_x + 1)
        plt.ylim(min_cell_y - 1, max_cell_y + 1)

        for x in range(min_cell_x, max_cell_x + 2):
            plt.axvline(x - 0.5, color='grey', linestyle='--', linewidth=0.5)
        for y in range(min_cell_y, max_cell_y + 2):
            plt.axhline(y - 0.5, color='grey', linestyle='--', linewidth=0.5)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc='lower left')
        plt.grid(False)
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Create a grid with cell size 0.1 (in effective coordinate units)
    # Choose a scaling method ("none", "minmax", or "standard") and set verbose to True.
    grid = CoordinateGrid(cell_size=0.1, scaling_method="minmax", verbose=True)
    
    # Example using a DataFrame with custom column names.
    data = {
        'lat': [80, 85, 90],
        'lon': [115, 120, 125]
    }
    df_coords = pd.DataFrame(data)
    
    # Fit the grid to the DataFrame. This computes the scaling parameters and transforms the data.
    fitted_df = grid.fit(df=df_coords, lon_col='lon', lat_col='lat')
    print("Fitted DataFrame with grid cell indices:")
    print(fitted_df)
    
    # Inverse transform the fitted grid cells back to original coordinates (center of cells).
    inv_coords_center = grid.inverse_transform(method="center")
    print("\nInverse transformed coordinates (center of cells):")
    print(inv_coords_center)
    
    # Inverse transform to get the lower left corner of each cell.
    inv_coords_corner = grid.inverse_transform(method="corner")
    print("\nInverse transformed coordinates (lower left corner of cells):")
    print(inv_coords_corner)
    
    # Plot the grid cells using the DataFrame coordinates with custom title and axis labels.
    grid.plot_grid(df=df_coords, lon_col='lon', lat_col='lat', 
                   title="Custom Grid Plot", x_label="Custom X Label", y_label="Custom Y Label")
    
    # Example using two arrays.
    longitudes = [115, 120, 125]
    latitudes = [80, 85, 90]
    cell_x_array, cell_y_array = grid.transform_bulk(longitudes=longitudes, latitudes=latitudes)
    print("\nTransformed arrays:")
    print("cell_x:", cell_x_array)
    print("cell_y:", cell_y_array)
    
    # Plot using the arrays with default title and labels.
    grid.plot_grid(longitudes=longitudes, latitudes=latitudes)
