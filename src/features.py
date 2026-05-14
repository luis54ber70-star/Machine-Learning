import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from config import Config

def process_features() -> None:
    print("[FEATURES] Leyendo datos y construyendo variables objetivo reales...")
    if not os.path.exists(Config.RAW_DATA_PATH):
        raise FileNotFoundError("Ejecuta primero ingest.py para descargar datos.")
        
    df = pd.read_csv(Config.RAW_DATA_PATH)
    
    # 1. Target Ganador (1 = Local, 2 = Visitante, 0 = Empate)
    df['target_ganador'] = np.where(df['goles_local'] > df['goles_visitante'], 1,
                                    np.where(df['goles_local'] < df['goles_visitante'], 2, 0))
    
    # 2. Target Ambos Anotan (1 = Sí, 0 = No)
    df['target_ambos_anotan'] = np.where((df['goles_local'] > 0) & (df['goles_visitante'] > 0), 1, 0)
    
    # 3. Target Más/Menos 2.5 Goles (1 = Más, 0 = Menos)
    df['target_mas_2_5'] = np.where((df['goles_local'] + df['goles_visitante']) > 2.5, 1, 0)
    
    df.to_csv(Config.PROCESSED_DATA_PATH, index=False)
    
    # Guardar escalador estadístico para inferencia
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    scaler = StandardScaler()
    scaler.fit(df[Config.FEATURES])
    joblib.dump(scaler, Config.SCALER_PATH)
    print(f"[FEATURES] Pipeline de variables completado. Escalador guardado.")

if __name__ == "__main__":
    process_features()
