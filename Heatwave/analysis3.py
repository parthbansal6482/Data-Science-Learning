# Which month experiences the most heatwaves?

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heatwave.csv")

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

selected_year = 2000

month_map = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec'
}

if selected_year == "all":

    monthly_heatwaves = (
        df.groupby('month')['is_heatwave']
        .sum()
    )

    title = "Monthly Heatwave Frequency (All Years)"

else:

    filtered_df = df[df['year'] == selected_year]

    monthly_heatwaves = (
        filtered_df.groupby('month')['is_heatwave']
        .sum()
    )

    title = f"Monthly Heatwave Frequency ({selected_year})"

months = [
    month_map[m]
    for m in monthly_heatwaves.index
]

plt.figure(figsize=(10, 6))

plt.bar(
    months,
    monthly_heatwaves.values
)

plt.title(
    title,
    fontsize=18
)

plt.xlabel("Month", fontsize=14)
plt.ylabel("Number of Heatwave Days", fontsize=14)

plt.grid(axis='y', alpha=0.3)

plt.show()

max_month = monthly_heatwaves.idxmax()

print(
    f"Month with highest heatwaves: {month_map[max_month]}"
)