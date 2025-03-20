

#  Geo-Tools  
*A Python package for geospatial data visualization and coordinate-based country lookup.*

---

## Installation  

### Step 1: Clone the Repository  
To use this package locally, clone the repository and install dependencies from `requirements.txt`:
```bash
git clone https://github.com/Louis-Li-dev/Geo-Tools.git
pip install -r requirements.txt
```
---

### Step 2: Install via Pip (Recommended)  
For direct installation from GitHub:
```bash
pip install git+https://github.com/Louis-Li-dev/Geo-Tools.git
```

---

##  How to Use  
*For documentation, go to [https://louis-li-dev.github.io/Geo-Tools/](https://louis-li-dev.github.io/Geo-Tools/)*
###  `geo_plot/map_plot`  
- Supports **1D, 2D, and 3D data visualization**  
- Plots **coordinates on a world map**  

### `geo_api/coordinate_lookup`  
- Converts **latitude/longitude** to **country names**  
- **Note:** Some countries may not be recognized due to dataset limitations  

### `geo_api/coordinate_transform`
- Discretize **coordinates** for easier modeling.
- Plot the transformed coordinates for simple analysis.

### 🔹 Example Usage: Plotting Check-Ins on a World Map  

```python
import pandas as pd
from geo_tools.geo_plot.map_plot import CoordinatePlotter

# Load sample data (modify 'xxx.csv' to your file)
df = pd.read_csv('./xxx.csv', nrows=10000)

# Filter data for specific users and years
plotter = CoordinatePlotter(df[(df.Year == 2009) & (df.Uid == 0)])

# Plot user check-ins on the map
plotter.plot(title='User Check-Ins')
```

- **Expected Output:**

| **Feature** | **Description**                                | **Example**                                                                                                                                         |
|-------------|------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **geo_plot** | Display coordinates on a map                  | <p align="center"><img src="https://github.com/user-attachments/assets/d6ff04eb-a2b1-4f12-add6-00986e303a47" alt="User Check-Ins"></p>          |
| **geo_api**  | Discretize float coordinates into integers      | <p align="center"><img src="https://github.com/user-attachments/assets/cf9373f7-8bf0-4353-b4aa-634ae7330ea0" alt="Discretized Coordinates"></p> |

---

##  References  

- Special thanks to [Stack Overflow](https://stackoverflow.com/questions/20169467/how-to-convert-from-longitude-and-latitude-to-country-or-city) for discussions on coordinate-to-country mapping solutions.  
- This package uses country boundary data from:  
  **[Geo-Countries Dataset](https://github.com/datasets/geo-countries)** (Licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**).  
