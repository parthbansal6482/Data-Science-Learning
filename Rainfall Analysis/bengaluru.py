import xarray as xr
import matplotlib.pyplot as plt
from pyextremes import EVA

ds = xr.open_mfdataset(
    "*.nc",
    combine="by_coords"
)

bengaluru_daily = ds["RAINFALL"].sel(
    LATITUDE=12.97,
    LONGITUDE=77.59,
    method="nearest"
)


series = bengaluru_daily.to_series()
eva = EVA(series)
eva.get_extremes(
    method="POT",
    threshold=20
)
eva.fit_model()
eva.plot_extremes()
eva.plot_return_values()
plt.show()

