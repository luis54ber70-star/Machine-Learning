import pandas as pd
import numpy as np
import os

print("PREPARING SPORTS DATA")

matches = 500

data = pd.DataFrame({

    "home_attack":
        np.random.randint(60, 100, matches),

    "away_attack":
        np.random.randint(60, 100, matches),

    "home_defense":
        np.random.randint(60, 100, matches),

    "away_defense":
        np.random.randint(60, 100, matches),

    "home_form":
        np.random.randint(0, 15, matches),

    "away_form":
        np.random.randint(0, 15, matches),

    "home_win":
        np.random.randint(0, 2, matches)
})

os.makedirs("data/processed", exist_ok=True)

data.to_csv(
    "data/processed/sports_data.csv",
    index=False
)

print("DATASET CREATED")
print(data.head())
