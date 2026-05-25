import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

df = pd.read_csv("heatwave.csv")

df['date'] = pd.to_datetime(df['date'])

hotspots = (
    df.groupby(['lat', 'lon'])['is_heatwave']
    .sum()
    .reset_index()
)

hotspots.rename(
    columns={'is_heatwave': 'heatwave_count'},
    inplace=True
)

world = gpd.read_file(
    ".venv/lib/python3.14/site-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp"
)

india = world[world['name'] == 'India']

fig, ax = plt.subplots(figsize=(1, 6))

india.plot(
    ax=ax,
    color='whitesmoke',
    edgecolor='black',
    linewidth=1
)

hb = ax.hexbin(
    hotspots['lon'],
    hotspots['lat'],
    C=hotspots['heatwave_count'],
    reduce_C_function=sum,
    gridsize=35,
    cmap='hot',
    mincnt=1
)

cbar = plt.colorbar(hb, ax=ax)

cbar.set_label('Heatwave Frequency')

ax.set_title(
    "India Heatwave Hotspots",
    fontsize=22,
    pad=20
)

ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)

ax.set_xlim(67, 98)
ax.set_ylim(6, 38)

plt.grid(alpha=0.3)

plt.show()