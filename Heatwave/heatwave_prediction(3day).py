import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

base_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_dir, "heatwave.csv"))

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year

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

df['rolling_3day_max'] = (
    df.groupby(['lat', 'lon'])['tmax_c']
    .transform(
        lambda x: x.rolling(3).max()
    )
)

df['rolling_3day_std'] = (
    df.groupby(['lat', 'lon'])['tmax_c']
    .transform(
        lambda x: x.rolling(3).std()
    )
)

df['temp_change_3day'] = (
    df['tmax_c'] - df['tmax_prev3']
)

df = df.dropna()

features = [
    'lat',
    'lon',
    'tmax_prev1',
    'tmax_prev2',
    'tmax_prev3',
    'rolling_3day_avg',
    'rolling_3day_max',
    'rolling_3day_std',
    'temp_change_3day'
]

train_df = df[df['year'] <= 2015]
test_df = df[df['year'] > 2015]

X_train = train_df[features]
y_train = train_df['is_severe_heatwave']

X_test = test_df[features]
y_test = test_df['is_severe_heatwave']

model = RandomForestClassifier(
    n_estimators=50,
    random_state=1,
    n_jobs=-1,
    class_weight='balanced'
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

threshold = 0.30

predictions = (
    probabilities > threshold
).astype(int)

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

auc_score = roc_auc_score(
    y_test,
    probabilities
)

print("\nROC-AUC Score:")
print(auc_score)

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

plt.figure(figsize=(12, 6))

plt.bar(
    feature_importance['Feature'],
    feature_importance['Importance']
)

plt.title("Feature Importance")

plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(rotation=30)

plt.grid(axis='y', alpha=0.3)

plt.show()

fpr, tpr, thresholds = roc_curve(
    y_test,
    probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(alpha=0.3)

plt.show()

sample_data = pd.DataFrame([{
    'lat': 28.5,
    'lon': 77.5,
    'tmax_prev1': 42.5,
    'tmax_prev2': 41.8,
    'tmax_prev3': 40.9,
    'rolling_3day_avg': 41.73,
    'rolling_3day_max': 42.5,
    'rolling_3day_std': 0.80,
    'temp_change_3day': 1.6
}])

sample_probability = model.predict_proba(
    sample_data
)

print(
    "\nNo Severe Heatwave :",
    sample_probability[0][0] * 100,
    "%"
)

print(
    "Severe Heatwave :",
    sample_probability[0][1] * 100,
    "%"
)

sample_prediction = (
    sample_probability[0][1] > threshold
)

if sample_prediction:
    print("Severe Heatwave Predicted")
else:
    print("No Severe Heatwave")