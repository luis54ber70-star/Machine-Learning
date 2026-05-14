import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("TRAINING SPORTS MODEL")

data = pd.read_csv(
    "data/processed/sports_data.csv"
)

X = data.drop(columns=["home_win"])

y = data["home_win"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"MODEL ACCURACY: {accuracy}")

joblib.dump(
    model,
    "models/sports_model.pkl"
)

print("MODEL SAVED")
