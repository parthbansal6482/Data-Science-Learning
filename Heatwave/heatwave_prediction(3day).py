import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

df = pd.read_csv("heatwave.csv")

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

df = df.sort_values(
    by=['lat', 'lon', 'date']
)

df['tmax_prev1'] = (
    df.groupby(['lat', 'lon'])['tmax_c']
    .shift(1)
)

df['tmax_prev2'] = (
    df.groupby(['lat', 'lon'])['tmax_c']
    .shift(2)
)

df['tmax_prev3'] = (
    df.groupby(['lat', 'lon'])['tmax_c']
    .shift(3)
)

df['rolling_3day_avg'] = (
    df.groupby(['lat', 'lon'])['tmax_c']
    .transform(
        lambda x: x.rolling(3).mean()
    )
)

df['temp_change_3day'] = (
    df['tmax_c'] - df['tmax_prev3']
)

df = df.dropna()

features = [
    'month',
    'tmax_prev1',
    'tmax_prev2',
    'tmax_prev3',
    'rolling_3day_avg',
    'temp_change_3day'
]

X = df[features]

y = df['is_severe_heatwave']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)

model = RandomForestClassifier(
    n_estimators=50,
    random_state=1,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


sample_data = pd.DataFrame([{
    'month': 5,
    'tmax_prev1': 40.5,
    'tmax_prev2': 41.2,
    'tmax_prev3': 39.8,
    'rolling_3day_avg': 40.5,
    'temp_change_3day': 1.25
}])

probability = model.predict_proba(sample_data)

print(
    "\nNo Severe Heatwave :",
    probability[0][0] * 100,
    "%"
)

print(
    "Severe Heatwave :",
    probability[0][1] * 100,
    "%"
)

prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("Severe Heatwave Predicted")
else:
    print("No Severe Heatwave")
    