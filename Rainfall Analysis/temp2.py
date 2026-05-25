import xarray as xr
import numpy as np

ds = xr.open_dataset("data.nc")

kerala = ds["RAINFALL"].sel(
    LATITUDE=slice(8.0, 12.8),
    LONGITUDE=slice(74.8, 77.5)
)

extreme_heavy_rainfall = np.where(kerala.values > 204.5)

very_heavy_rainfall = np.where(
    (kerala.values > 115.6) &
    (kerala.values <= 204.5)
)

heavy_rainfall = np.where(
    (kerala.values > 64.5) &
    (kerala.values <= 115.6)
)

moderate_rainfall = np.where(
    (kerala.values > 20.4) &
    (kerala.values <= 64.5)
)

light_rainfall = np.where(
    (kerala.values > 2.5) &
    (kerala.values <= 20.4)
)

rainfall = [
    ("Extreme Heavy Rainfall", extreme_heavy_rainfall),
    ("Very Heavy Rainfall", very_heavy_rainfall)
    # ("Heavy Rainfall", heavy_rainfall),
    # ("Moderate Rainfall", moderate_rainfall),
    # ("Light Rainfall", light_rainfall)
]

for category_name, indices in rainfall:

    print("\n")
    print("=" * 60)
    print(category_name)
    print("=" * 60)

    times = kerala.TIME.values[indices[0]]
    lats = kerala.LATITUDE.values[indices[1]]
    lons = kerala.LONGITUDE.values[indices[2]]
    values = kerala.values[indices]

    for time, latitude, longitude, value in zip(
        times,
        lats,
        lons,
        values
    ):

        print(
            "\nDate:", time,
            "\nLatitude:", latitude,
            "| Longitude:", longitude,
            "\nRainfall:", round(float(value), 2), "mm"
        )