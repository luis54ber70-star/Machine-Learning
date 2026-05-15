from fastapi import FastAPI
import pandas as pd
import joblib
import uuid
from datetime import datetime
import os

app = FastAPI(title="ML Model API - Feedback Loop")

MODEL_PATH = "models/model.json" # Ajusta a tu ruta
FEEDBACK_FILE = "data/feedback.csv"

# Cargar el modelo al iniciar
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

@app.post("/predict")
def predict(data: dict):
    """
    Recibe datos, devuelve predicción y genera un ID único para seguimiento.
    """
    if model is None:
        return {"error": "Modelo no encontrado"}
    
    # Convertir dict a DataFrame
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    prediction_id = str(uuid.uuid4())
    
    # Guardar temporalmente la entrada y la predicción
    log_entry = data.copy()
    log_entry["prediction_id"] = prediction_id
    log_entry["prediction"] = float(prediction)
    log_entry["timestamp"] = datetime.now().isoformat()
    log_entry["actual_outcome"] = None # Se llenará después
    
    # Guardar en el CSV de feedback
    df_log = pd.DataFrame([log_entry])
    df_log.to_csv(FEEDBACK_FILE, mode='a', header=not os.path.exists(FEEDBACK_FILE), index=False)
    
    return {"prediction_id": prediction_id, "prediction": float(prediction)}

@app.post("/feedback")
def feedback(prediction_id: str, actual_outcome: float):
    """
    Recibe el resultado real de una predicción previa para mejorar el modelo.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return {"error": "No hay registros de feedback"}
    
    df = pd.read_csv(FEEDBACK_FILE)
    if prediction_id in df['prediction_id'].values:
        df.loc[df['prediction_id'] == prediction_id, 'actual_outcome'] = actual_outcome
        df.to_csv(FEEDBACK_FILE, index=False)
        return {"status": "Feedback recibido correctamente"}
    
    return {"error": "ID de predicción no encontrado"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
