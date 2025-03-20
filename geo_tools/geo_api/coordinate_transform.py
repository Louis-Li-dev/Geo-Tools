import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional

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
class CoordDiscretizer:
    def __init__(self, start_val: int = 1, mode: str = "order", output: str = "sequence", verbose: bool = False) -> None:
        """
        Initialize the CoordDiscretizer.

        Parameters:
            start_val (int): The starting value for order-based marking.
            mode (str): "order" or "binary". In "order" mode, visited cells will be marked 
                        by the order of visit (start_val, start_val+1, etc.). In "binary" mode,
                        visited cells are simply marked as 1.
            output (str): "sequence" or "matrix". If "sequence", the transformation returns a 
                          dictionary mapping each UID to a list of (cell_x, cell_y) tuples (in visit order).
                          If "matrix", the transformation returns a dictionary mapping each UID to a 
                          2D numpy array (matrix) with the visited cells marked.
            verbose (bool): If True, prints out detailed status messages.
        """
        if mode not in ["order", "binary"]:
            raise ValueError("mode must be 'order' or 'binary'")
        if output not in ["sequence", "matrix"]:
            raise ValueError("output must be 'sequence' or 'matrix'")
        
        self.start_val: int = start_val
        self.mode: str = mode
        self.output: str = output
        self.verbose: bool = verbose
        self.global_max_x: Optional[int] = None  # Store global max_x
        self.global_max_y: Optional[int] = None  # Store global max_y

    def _apply_scaling(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale the raw coordinate arrays (x, y) according to the selected scaling method.
        Returns the effective coordinates and stores the parameters needed for inversion.
        For now, only 'none' is implemented (which centers the data).
        """
        if self.scaling_method == "none":
            min_x, min_y = np.min(x), np.min(y)
            effective_x = x - min_x
            effective_y = y - min_y
            self._scaling_info = {"method": "none", "min_x": min_x, "min_y": min_y}
            if self.verbose:
                print(f"[Verbose] Scaling method: none. Computed global min: ({min_x}, {min_y}).")
        else:
            raise ValueError("Only 'none' scaling is implemented in this version.")
        return effective_x, effective_y

    def _inverse_scaling(self, effective_x: np.ndarray, effective_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse the scaling transformation on effective coordinates to recover original values.
        """
        info = self._scaling_info
        if info is None:
            raise ValueError("Scaling info not available. Call fit_transform first.")
        if info["method"] == "none":
            orig_x = effective_x + info["min_x"]
            orig_y = effective_y + info["min_y"]
        else:
            raise ValueError("Only 'none' scaling is implemented in this version.")
        return orig_x, orig_y

    def fit_transform(
        self,
        df: pd.DataFrame,
        uid_col: str = "Uid",
        time_col: str = "Timestamp",
        date_col: str = "Date",
        lat_col: str = "Latitude",
        lon_col: str = "Longitude"
    ) -> Union[Dict[Union[str, int], List[Tuple[int, int]]], Dict[Union[str, int], np.ndarray]]:
        """
        Transforms coordinate data into sequences by UID using a global max grid size.

        Returns:
            A dictionary mapping UID to either:
              - A sequence of visited (cell_x, cell_y) tuples, if `output="sequence"`
              - A 2D numpy array (matrix), if `output="matrix"`
        """
        # Ensure the timestamp column is in datetime format
        df[time_col] = pd.to_datetime(df[time_col])

        # Step 1: For each UID and Date, select the first record (earliest timestamp)
        daily_first = df.sort_values(time_col).groupby([uid_col, date_col], as_index=False).first()

        # Step 2: Drop duplicate grid cells for each UID
        final_df = daily_first.drop_duplicates(subset=[uid_col, lon_col, lat_col], keep="first")

        # Step 3: Compute **global** max_x and max_y (before grouping by UID)
        self.global_max_x = final_df[lon_col].max()
        self.global_max_y = final_df[lat_col].max()

        if self.verbose:
            print(f"[Verbose] Global max_x: {self.global_max_x}, Global max_y: {self.global_max_y}")

        # Step 4: Group by UID and create sequences
        sequences: Dict[Union[str, int], List[Tuple[int, int]]] = {}
        for uid, group in final_df.groupby(uid_col):
            group_sorted = group.sort_values(time_col)
            seq: List[Tuple[int, int]] = list(zip(group_sorted[lon_col], group_sorted[lat_col]))
            sequences[uid] = seq

        if self.output == "sequence":
            return sequences  # Return sequence if sequence mode is selected

        # Otherwise, create matrix representations
        matrices: Dict[Union[str, int], np.ndarray] = {}
        for uid, seq in sequences.items():
            if not seq:
                matrices[uid] = np.array([])  # Empty case
                continue

            # Use **global** max_x and max_y to create a fixed-size matrix
            mat = np.zeros((self.global_max_x + 1, self.global_max_y + 1), dtype=int)

            if self.mode == "order":
                val = self.start_val
                for (x, y) in seq:
                    mat[x, y] = val
                    val += 1
            elif self.mode == "binary":
                for (x, y) in seq:
                    mat[x, y] = 1

            matrices[uid] = mat
        
        return matrices  # Return matrix representation

    def inverse_transform(
        self,
        input_data: Union[
            np.ndarray,
            Dict[Union[str, int], np.ndarray],
            Dict[Union[str, int], List[Tuple[int, int]]]
        ],
        method: str = "center"
    ) -> Union[List[Tuple[int, int]], Dict[Union[str, int], List[Tuple[int, int]]]]:
        """
        Inverse transforms grid cell indices back to a list of coordinates.

        Parameters:
            input_data: Either a single matrix (np.ndarray) or a dictionary (as produced by fit_transform)
                        mapping UID to either a matrix or a sequence.
            method (str): "center" (default) returns the center of the grid cell; "corner" returns the lower left corner.

        Returns:
            If input_data is a matrix, returns a list of (cell_x, cell_y) tuples.
            If input_data is a dictionary, returns a dictionary mapping each UID to a list of coordinates.
        """
        def _process_matrix(mat: np.ndarray) -> List[Tuple[int, int]]:
            if np.max(mat) > 1:
                # Order mode: sort coordinates by the order value.
                coords = np.argwhere(mat > 0)
                order_vals = [mat[x, y] for x, y in coords]
                sorted_coords = [tuple(coord) for _, coord in sorted(zip(order_vals, coords), key=lambda pair: pair[0])]
                return sorted_coords
            else:
                # Binary mode: return coordinates in random order.
                coords = np.argwhere(mat > 0)
                coords_list = [tuple(coord) for coord in coords]
                np.random.shuffle(coords_list)
                return coords_list

        if isinstance(input_data, np.ndarray):
            return _process_matrix(input_data)
        elif isinstance(input_data, dict):
            result: Dict[Union[str, int], List[Tuple[int, int]]] = {}
            for uid, value in input_data.items():
                if isinstance(value, np.ndarray):
                    result[uid] = _process_matrix(value)
                elif isinstance(value, list):
                    # Assume already a sequence; return as is.
                    result[uid] = value
                else:
                    raise ValueError(f"Unrecognized type for UID {uid}.")
            return result
        else:
            raise ValueError("Input data must be a numpy array or a dictionary.")


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
    # Create a sample DataFrame (assumed to have already been discretized into grid cells).
    data = {
        "Uid": [0, 0, 0, 1, 1],
        "Timestamp": [
            "2009-05-25 20:56:10+00:00",
            "2009-05-25 21:35:28+00:00",
            "2009-05-25 21:42:47+00:00",
            "2010-09-29 20:28:44+00:00",
            "2010-10-08 21:29:31+00:00"
        ],
        "Date": ["2009-05-25", "2009-05-25", "2009-05-25", "2010-09-29", "2010-10-08"],
        "Latitude": [37.774929, 37.600747, 37.600747, 37.792818, 37.737615],
        "Longitude": [-122.419415, -122.382376, -122.382376, -122.392636, -122.240925],
        # For demonstration, assume these are the discretized values:
        "cell_x": [186, 186, 186, 187, 187],
        "cell_y": [91, 90, 90, 91, 90]
    }
    ch_df = pd.DataFrame(data)
    
    # Instantiate CoordDiscretizer for sequence output.
    discretizer_seq = CoordDiscretizer(start_val=1, mode="order", output="sequence", verbose=True)
    sequences: Dict[Union[str, int], List[Tuple[int, int]]] = discretizer_seq.fit_transform(
        ch_df, uid_col="Uid", time_col="Timestamp", date_col="Date", lat_col="cell_y", lon_col="cell_x"
    )
    print("Sequences by UID:")
    for uid, seq in sequences.items():
        print(f"UID {uid}: {seq}")
    
    # Instantiate CoordDiscretizer for matrix output.
    discretizer_mat = CoordDiscretizer(start_val=1, mode="order", output="matrix", verbose=True)
    matrices: Dict[Union[str, int], np.ndarray] = discretizer_mat.fit_transform(
        ch_df, uid_col="Uid", time_col="Timestamp", date_col="Date", lat_col="cell_y", lon_col="cell_x"
    )
    print("\nMatrix representations by UID:")
    for uid, mat in matrices.items():
        print(f"UID {uid} matrix:\n{mat}\n")
    
    # Inverse transform a matrix from UID 0.
    if 0 in matrices:
        inv_seq_from_mat: List[Tuple[int, int]] = discretizer_mat.inverse_transform(matrices[0])
        print("Inverse transformed sequence from UID 0's matrix:")
        print(inv_seq_from_mat)
    
    # Inverse transform using the dict (from transform) directly.
    inv_seq_from_dict: Dict[Union[str, int], List[Tuple[int, int]]] = discretizer_mat.inverse_transform(sequences)
    print("\nInverse transformed sequences from dictionary:")
    for uid, seq in inv_seq_from_dict.items():
        print(f"UID {uid}: {seq}")
