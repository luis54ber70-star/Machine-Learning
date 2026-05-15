import joblib
import pandas as pd
import os
from datetime import datetime
import uuid

print("--- EJECUTANDO PREDICCIONES DEPORTIVAS ---")

# 1. CARGAR EL MODELO
MODEL_PATH = "models/sports_model.pkl"
FEEDBACK_PATH = "data/feedback.csv"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: No se encontró el modelo en {MODEL_PATH}. Ejecuta primero train_model.py")
    exit(1)

model = joblib.load(MODEL_PATH)

# 2. DATOS DEL PARTIDO (Features)
# Nota: Asegúrate de que estas columnas coincidan exactamente con las de tu entrenamiento
match_data = {
    "home_attack": [92],
    "away_attack": [84],
    "home_defense": [90],
    "away_defense": [79],
    "home_form": [12],
    "away_form": [8]
}

match_df = pd.DataFrame(match_data)

# 3. REALIZAR PREDICCIÓN
prediction = model.predict(match_df)[0]
probabilities = model.predict_proba(match_df)[0]

home_win_probability = round(probabilities[1] * 100, 2)
away_win_probability = round(probabilities[0] * 100, 2)

# 4. GENERAR ID Y LOG (Para el aprendizaje continuo)
prediction_id = str(uuid.uuid4())[:8] # ID corto para seguimiento
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"\nPARTIDO: Real Madrid vs Barcelona")
print(f"ID PREDICCIÓN: {prediction_id}")
print(f"PROBABILIDAD LOCAL: {home_win_probability}%")
print(f"PROBABILIDAD VISITANTE: {away_win_probability}%")
print(f"RESULTADO PREDICHO: {'Gana Local (1)' if prediction == 1 else 'Gana Visitante/Empate (0)'}")

# 5. GUARDAR EN FEEDBACK.CSV
# Creamos un diccionario con los datos de entrada + la predicción
log_entry = match_data.copy()
log_entry["prediction_id"] = [prediction_id]
log_entry["timestamp"] = [timestamp]
log_entry["prediction"] = [int(prediction)]
log_entry["home_win"] = [None] # Esto queda vacío hasta que sepamos el resultado real

log_df = pd.DataFrame(log_entry)

# Guardar (si no existe el archivo, crea cabecera; si existe, añade fila)
os.makedirs("data", exist_ok=True)
log_df.to_csv(FEEDBACK_PATH, mode='a', header=not os.path.exists(FEEDBACK_PATH), index=False)

print(f"\n[SISTEMA] Predicción guardada en {FEEDBACK_PATH}")
print("Para que el modelo aprenda, edita el CSV y pon el resultado real en la columna 'home_win'.")
