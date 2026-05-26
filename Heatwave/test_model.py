import joblib
import pandas as pd

model = joblib.load(
    "severe_heatwave_model.pkl"
)

threshold = 0.30


farm_df = pd.read_csv("farm_heatwave_features_updated.csv")

farm_features = [
    'tmax_prev1',
    'tmax_prev2',
    'tmax_prev3',
    'rolling_3day_avg',
    'rolling_3day_max',
    'rolling_3day_std',
    'temp_change_3day',
    'tmax_normal',
    'departure'
]

farm_df = farm_df.dropna(
    subset=farm_features
)

farm_probabilities = model.predict_proba(
    farm_df[farm_features]
)[:, 1]

farm_predictions = (
    farm_probabilities > threshold
).astype(int)

farm_df['severe_heatwave_probability'] = (
    farm_probabilities * 100
)

farm_df['severe_heatwave_prediction'] = (
    farm_predictions
)

print(
    farm_df[
        [
            'name',
            'location',
            'severe_heatwave_probability',
            'severe_heatwave_prediction'
        ]
    ]
)

farm_df.to_csv("farm_heatwave_predictions1.csv",index=False)



# sample data testing
# sample_data = pd.DataFrame([{
#     'tmax_prev1': 46.0,
#     'tmax_prev2': 35.0,
#     'tmax_prev3': 34.0,
#     'rolling_3day_avg': 38.33,
#     'rolling_3day_max': 46.0,
#     'rolling_3day_std': 6.6,
#     'temp_change_3day': 12.0,
#     'tmax_normal': 33.0,
#     'departure': 13.0
# }])

# sample_probability = model.predict_proba(
#     sample_data
# )

# print(
#     "\nNo Severe Heatwave :",
#     sample_probability[0][0] * 100,
#     "%"
# )

# print(
#     "Severe Heatwave :",
#     sample_probability[0][1] * 100,
#     "%"
# )

# sample_prediction = (
#     sample_probability[0][1] > threshold
# )

# if sample_prediction:
#     print("Severe Heatwave Predicted")
# else:
#     print("No Severe Heatwave")