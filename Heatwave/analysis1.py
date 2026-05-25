# Are heatwave days increasing yearly?

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("heatwave.csv")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

yearly_heatwaves = df.groupby('year')['is_heatwave'].sum()
print(yearly_heatwaves)


# Fit linear trendline
z = np.polyfit(
    yearly_heatwaves.index,
    yearly_heatwaves.values,
    1
)

p = np.poly1d(z)

# Plot
plt.figure(figsize=(12,6))

plt.plot(
    yearly_heatwaves.index,
    yearly_heatwaves.values,
    marker='o',
    label='Heatwave Days'
)

plt.plot(
    yearly_heatwaves.index,
    p(yearly_heatwaves.index),
    linestyle='--',
    label='Trendline'
)

plt.xlabel("Year")
plt.ylabel("Number of Heatwave Days")
plt.title("Trend in Heatwave Days Over Years")

plt.legend()
plt.grid(True)

plt.show()

print("Trend slope:", z[0])