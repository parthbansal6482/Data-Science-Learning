import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("data.nc")




kerala = ds["RAINFALL"].sel(
    LATITUDE=slice(8.0, 12.8),
    LONGITUDE=slice(74.8, 77.5)
)


kerala_daily = kerala.max(dim=["LATITUDE", "LONGITUDE"])

threshold = 50




plt.figure(figsize=(12, 5))

plt.plot(
    kerala_daily.TIME.values,
    kerala_daily.values,
    label="Daily Rainfall"
)

plt.axhline(
    y=threshold,
    color='red',
    linestyle='--',
    label=f"Threshold ({threshold} mm)"
)

plt.title("Daily Rainfall in Kerala (2025)")
plt.xlabel("Date")
plt.ylabel("Rainfall (mm)")

plt.legend()

plt.grid(True)

plt.show()
