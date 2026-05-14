import joblib
import pandas as pd

print("RUNNING SPORTS PREDICTIONS")

model = joblib.load(
    "models/sports_model.pkl"
)

match = pd.DataFrame({

    "home_attack": [92],
    "away_attack": [84],

    "home_defense": [90],
    "away_defense": [79],

    "home_form": [12],
    "away_form": [8]
})

prediction = model.predict(match)[0]

probabilities = model.predict_proba(match)[0]

home_win_probability = round(
    probabilities[1] * 100,
    2
)

away_win_probability = round(
    probabilities[0] * 100,
    2
)

print("MATCH PREDICTION")

print({
    "home_team": "Real Madrid",
    "away_team": "Barcelona",
    "prediction": int(prediction),
    "home_win_probability":
        home_win_probability,
    "away_win_probability":
        away_win_probability
})
