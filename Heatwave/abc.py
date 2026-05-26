# import pandas as pd
# import ast
# import time
# from geopy.geocoders import Nominatim

# df = pd.read_csv("info.csv")

# geolocator = Nominatim(
#     user_agent="geo_project"
# )

# location_cache = {}

# def extract_data(coord_string):

#     if coord_string in location_cache:
#         return pd.Series(
#             location_cache[coord_string]
#         )

#     data = ast.literal_eval(coord_string)

#     name = data['name']

#     coords = data['coords']

#     lons = [c[0] for c in coords]
#     lats = [c[1] for c in coords]

#     center_lon = sum(lons) / len(lons)
#     center_lat = sum(lats) / len(lats)

#     try:

#         location = geolocator.reverse(
#             f"{center_lat}, {center_lon}"
#         )

#         time.sleep(1.1)

#         if location:
#             address = location.address
#         else:
#             address = None

#     except:
#         address = None

#     result = [name, address]

#     location_cache[coord_string] = result

#     return pd.Series(result)

# df[['name', 'location']] = df['polyinfo'].apply(
#     extract_data
# )

# print(df[['name', 'location']])

# df.to_csv(
#     "updated_info.csv",
#     index=False
# )



import pandas as pd
import ast
import requests
import numpy as np
import time

df = pd.read_csv("updated_info.csv")

temperature_cache = {}

def get_centroid(poly_string):

    data = ast.literal_eval(poly_string)

    coords = data['coords']

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    center_lon = sum(lons) / len(lons)
    center_lat = sum(lats) / len(lats)

    return center_lat, center_lon

def fetch_temperature(lat, lon, date, hour):

    cache_key = (
        round(lat, 3),
        round(lon, 3),
        date,
        hour
    )

    if cache_key in temperature_cache:
        return temperature_cache[cache_key]

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}"
        f"&longitude={lon}"
        f"&start_date={date}"
        f"&end_date={date}"
        f"&hourly=temperature_2m"
        f"&timezone=auto"
    )

    response = requests.get(url)

    data = response.json()

    times = data['hourly']['time']
    temps = data['hourly']['temperature_2m']

    target_time = f"{date}T{hour}"

    for t, temp in zip(times, temps):

        if t == target_time:

            temperature_cache[cache_key] = temp

            return temp

    return np.nan

df['upload_time'] = pd.to_datetime(
    df['upload_time']
)

temp_prev1 = []
temp_prev2 = []
temp_prev3 = []

for _, row in df.iterrows():

    lat, lon = get_centroid(
        row['polyinfo']
    )

    current_time = row['upload_time']

    hour = current_time.strftime("%H:00")

    dates = [
        (current_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        (current_time - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        (current_time - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    ]

    try:

        t1 = fetch_temperature(
            lat,
            lon,
            dates[0],
            hour
        )

        t2 = fetch_temperature(
            lat,
            lon,
            dates[1],
            hour
        )

        t3 = fetch_temperature(
            lat,
            lon,
            dates[2],
            hour
        )

    except:

        t1, t2, t3 = np.nan, np.nan, np.nan

    temp_prev1.append(t1)
    temp_prev2.append(t2)
    temp_prev3.append(t3)

    time.sleep(1)

df['tmax_prev1'] = temp_prev1
df['tmax_prev2'] = temp_prev2
df['tmax_prev3'] = temp_prev3

df['rolling_3day_avg'] = (
    df[
        [
            'tmax_prev1',
            'tmax_prev2',
            'tmax_prev3'
        ]
    ]
    .mean(axis=1)
)

df['rolling_3day_max'] = (
    df[
        [
            'tmax_prev1',
            'tmax_prev2',
            'tmax_prev3'
        ]
    ]
    .max(axis=1)
)

df['rolling_3day_std'] = (
    df[
        [
            'tmax_prev1',
            'tmax_prev2',
            'tmax_prev3'
        ]
    ]
    .std(axis=1)
)

df['temp_change_3day'] = (
    df['tmax_prev1'] -
    df['tmax_prev3']
)

print(
    df[
        [
            'name',
            'tmax_prev1',
            'tmax_prev2',
            'tmax_prev3',
            'rolling_3day_avg',
            'rolling_3day_max',
            'rolling_3day_std',
            'temp_change_3day'
        ]
    ]
)

df.to_csv(
    "farm_heatwave_features.csv",
    index=False
)