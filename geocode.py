import pandas as pd
import zipfile
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point


# Download shapefile data for SF
# wget https://www2.census.gov/geo/tiger/TIGER2017//ROADS/tl_2017_06075_roads.zip
    
# Unzip shapefiles into shapefiles folder using unzip
#with zipfile.ZipFile('tl_2017_06075_roads.zip', 'r') as zip_ref:
#    zip_ref.extractall('shapefiles')

# Import shapefile as GeoDataFrame
ax = gpd.read_file('shapefiles/tl_2017_06075_roads.shp')

print(ax.crs)# output: {'init': 'epsg:4269'}

# Plot the shp file 
ax = ax.plot()

# Import our data
df = pd.read_csv('processed-dataset_processed_san_francisco.csv')

#Incident Category,Latitude,Longitude only keep these columns
df = df[['Incident Category','Latitude','Longitude']]
print(df.head())
print(df.shape)

#Save df to csv
df.to_csv('sf.csv', index=False)

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

print(gdf.head())

#geo

# Plot the data
gdf.plot(ax=ax, color="red")

plt.show()