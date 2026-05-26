import pandas as pd
import ast
import requests
import time

df = pd.read_csv("farm_heatwave_features.csv")

monthly_temp_cache = {}

def get_centroid(poly_string):

    data = ast.literal_eval(poly_string)

    coords = data['coords']

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    center_lon = sum(lons) / len(lons)
    center_lat = sum(lats) / len(lats)

    return center_lat, center_lon

def fetch_monthly_avg_temp(
    lat,
    lon,
    month
):

    rounded_lat = round(lat, 1)
    rounded_lon = round(lon, 1)

    cache_key = (
        rounded_lat,
        rounded_lon,
        month
    )

    if cache_key in monthly_temp_cache:
        return monthly_temp_cache[cache_key]

    start_date = f"2024-{month:02d}-01"
    end_date = f"2024-{month:02d}-28"

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={rounded_lat}"
        f"&longitude={rounded_lon}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&daily=temperature_2m_mean"
        f"&timezone=auto"
    )

    response = requests.get(url)

    data = response.json()

    temps = data['daily']['temperature_2m_mean']

    monthly_avg = round(
        sum(temps) / len(temps),
        2
    )

    monthly_temp_cache[
        cache_key
    ] = monthly_avg

    time.sleep(1)

    return monthly_avg

df['upload_time'] = pd.to_datetime(
    df['upload_time']
)

normal_temps = []

for _, row in df.iterrows():

    lat, lon = get_centroid(
        row['polyinfo']
    )

    month = row[
        'upload_time'
    ].month

    try:

        normal_temp = fetch_monthly_avg_temp(
            lat,
            lon,
            month
        )

    except:

        normal_temp = None

    normal_temps.append(
        normal_temp
    )

df['normal_temp'] = normal_temps

df['departure'] = (
    df['rolling_3day_max'] -
    df['normal_temp']
)

print(
    df[
        [
            'name',
            'rolling_3day_max',
            'normal_temp',
            'departure'
        ]
    ]
)

df.to_csv(
    "farm_heatwave_features_updated.csv",
    index=False
)