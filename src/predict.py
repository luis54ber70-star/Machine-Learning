import pandas as pd
import joblib
from config import Config

class FootballPredictor:
    def __init__(self):
        self.model = joblib.load(Config.MODEL_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)
        self.mapeo_ganador = {0: "Empate (X)", 1: "Local (1)", 2: "Visitante (2)"}

    def predict_match(self, input_data: dict) -> dict:
        df_match = pd.DataFrame([input_data])
        scaled_data = self.scaler.transform(df_match[Config.FEATURES])
        
        preds = self.model.predict(scaled_data)[0]
        probs = self.model.predict_proba(scaled_data)
        
        # El formato multi-output de scikit-learn genera una lista de arrays para probabilidades
        prob_ganador = probs[0][0][preds[0]] * 100
        prob_ambos = probs[1][0][preds[1]] * 100
        prob_mas = probs[2][0][preds[2]] * 100

        return {
            "ganador": {"resultado": self.mapeo_ganador[preds[0]], "probabilidad": round(prob_ganador, 2)},
            "ambos_anotan": {"resultado": "Sí" if preds[1] == 1 else "No", "probabilidad": round(prob_ambos, 2)},
            "mas_2_5_goles": {"resultado": "Más de 2.5" if preds[2] == 1 else "Menos de 2.5", "probabilidad": round(prob_mas, 2)}
        }
