import pandas as pd
import xgboost as xgb
import joblib
import os

FEEDBACK_FILE = "data/feedback.csv"
MODEL_PATH = "models/model.json"

def retrain():
    if not os.path.exists(FEEDBACK_FILE):
        print("No hay datos para re-entrenar.")
        return

    df = pd.read_csv(FEEDBACK_FILE)
    
    # Solo usamos datos que ya tienen el resultado real (actual_outcome)
    train_df = df.dropna(subset=['actual_outcome'])
    
    if len(train_df) < 10: # Mínimo de datos nuevos para re-entrenar
        print("Datos insuficientes para mejorar el modelo.")
        return

    # Separar X e y (Ajusta los nombres de tus columnas)
    # Excluimos columnas de control
    X = train_df.drop(columns=['prediction_id', 'prediction', 'timestamp', 'actual_outcome'])
    y = train_df['actual_outcome']

    # Cargar modelo anterior para "Warm Start" (Seguir aprendiendo)
    new_model = xgb.XGBRegressor() # O XGBClassifier() según tu caso
    
    if os.path.exists(MODEL_PATH):
        new_model.fit(X, y, xgb_model=MODEL_PATH)
    else:
        new_model.fit(X, y)

    # Guardar el modelo mejorado
    joblib.dump(new_model, MODEL_PATH)
    print(f"Modelo actualizado con {len(train_df)} registros nuevos.")

if __name__ == "__main__":
    retrain()
