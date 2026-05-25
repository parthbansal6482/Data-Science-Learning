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

df = df.dropna()

features = [
    'tmax_c',
    'tmax_normal'
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
    n_estimators=100,
    random_state=1
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
    'tmax_c': 41.04999923706055,
    'tmax_normal': 33.518856048583984,
}])

probability = model.predict_proba(sample_data)

print("No Severe Heatwave :",probability[0][0] * 100 , "%" )
print("Severe Heatwave :",probability[0][1] * 100 , "%" )

prediction = model.predict(sample_data)
if prediction[0] == 1:
    print("Severe Heatwave Predicted")
else:
    print("No Severe Heatwave")