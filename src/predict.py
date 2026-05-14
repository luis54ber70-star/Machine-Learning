import os
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any
from config import Config

logger = logging.getLogger(__name__)


class FootballPredictor:
    """Clase para realizar predicciones de partidos de fútbol."""
    
    def __init__(self):
        """Inicializa el predictor cargando modelo y escalador."""
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(
                f"No se encuentra el modelo entrenado: {Config.MODEL_PATH}. "
                "Ejecuta primero train.py"
            )
        if not os.path.exists(Config.SCALER_PATH):
            raise FileNotFoundError(
                f"No se encuentra el escalador: {Config.SCALER_PATH}. "
                "Ejecuta primero features.py"
            )
        
        self.model = joblib.load(Config.MODEL_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)
        
        self.mapeo_ganador = {0: "Empate (X)", 1: "Local (1)", 2: "Visitante (2)"}
        logger.info("Predictor inicializado correctamente")
    
    def predict_match(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predice el resultado de un partido.
        
        Args:
            input_data: Diccionario con características del partido
                Ejemplo: {
                    'rendimiento_local_forma': 0.7,
                    'rendimiento_vis_forma': 0.5,
                    'promedio_goles_local': 2.0,
                    'promedio_goles_vis': 1.5
                }
        
        Returns:
            Diccionario con predicciones y probabilidades
        """
        # Validar entrada
        self._validate_input(input_data)
        
        # Crear DataFrame con los datos de entrada
        df_match = pd.DataFrame([input_data])
        
        # Escalar datos
        scaled_data = self.scaler.transform(df_match[Config.FEATURES])
        
        # Realizar predicciones
        preds = self.model.predict(scaled_data)[0]
        probs = self.model.predict_proba(scaled_data)
        
        # Procesar probabilidades para cada target
        resultados = {}
        
        # Target: Ganador
        ganador_prob = probs[0][0][preds[0]] * 100 if len(probs[0][0]) > preds[0] else 33.33
        resultados['ganador'] = {
            "resultado": self.mapeo_ganador[preds[0]],
            "probabilidad": round(float(ganador_prob), 2),
            "codigo": int(preds[0])
        }
        
        # Target: Ambos anotan
        if len(probs) > 1 and len(probs[1][0]) > 1:
            ambos_prob = probs[1][0][preds[1]] * 100
        else:
            ambos_prob = 50.0
        resultados['ambos_anotan'] = {
            "resultado": "Sí" if preds[1] == 1 else "No",
            "probabilidad": round(float(ambos_prob), 2),
            "codigo": int(preds[1])
        }
        
        # Target: Más/Menos 2.5 goles
        if len(probs) > 2 and len(probs[2][0]) > 1:
            mas_prob = probs[2][0][preds[2]] * 100
        else:
            mas_prob = 50.0
        resultados['mas_2_5_goles'] = {
            "resultado": "Más de 2.5" if preds[2] == 1 else "Menos de 2.5",
            "probabilidad": round(float(mas_prob), 2),
            "codigo": int(preds[2])
        }
        
        logger.info(f"Predicción realizada: {resultados['ganador']['resultado']}")
        
        return resultados
    
    def _validate_input(self, input_data: Dict[str, float]) -> None:
        """Valida los datos de entrada."""
        required_features = Config.FEATURES
        missing_features = [f for f in required_features if f not in input_data]
        
        if missing_features:
            raise ValueError(f"Características faltantes: {missing_features}")
        
        for feature in required_features:
            if not isinstance(input_data[feature], (int, float)):
                raise ValueError(f"Característica {feature} debe ser numérica")
    
    def predict_batch(self, matches_data: list) -> list:
        """
        Predice resultados para múltiples partidos.
        
        Args:
            matches_data: Lista de diccionarios con datos de partidos
            
        Returns:
            Lista de predicciones
        """
        results = []
        for match_data in matches_data:
            try:
                prediction = self.predict_match(match_data)
                prediction['input_data'] = match_data
                results.append(prediction)
            except Exception as e:
                logger.error(f"Error al predecir partido {match_data}: {e}")
                results.append({'error': str(e), 'input_data': match_data})
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo."""
        return {
            'model_type': type(self.model).__name__,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'features': Config.FEATURES,
            'targets': Config.TARGETS
        }


def create_example_input() -> Dict[str, float]:
    """Crea un ejemplo de entrada para pruebas."""
    return {
        'rendimiento_local_forma': 0.7,  # 70% victorias en últimos 5 partidos
        'rendimiento_vis_forma': 0.5,    # 50% victorias en últimos 5 partidos
        'promedio_goles_local': 2.0,     # Promedio de goles como local
        'promedio_goles_vis': 1.5        # Promedio de goles como visitante
    }


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    try:
        predictor = FootballPredictor()
        
        # Predicción individual
        input_data = create_example_input()
        result = predictor.predict_match(input_data)
        
        print("\n=== RESULTADOS DE PREDICCIÓN ===")
        print(f"Ganador: {result['ganador']['resultado']} (Prob: {result['ganador']['probabilidad']}%)")
        print(f"Ambos anotan: {result['ambos_anotan']['resultado']} (Prob: {result['ambos_anotan']['probabilidad']}%)")
        print(f"Más/Menos 2.5: {result['mas_2_5_goles']['resultado']} (Prob: {result['mas_2_5_goles']['probabilidad']}%)")
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
