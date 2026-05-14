import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_espn_matches.csv")
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_espn_matches.csv")
    MODEL_PATH = os.path.join(MODELS_DIR, "football_espn_model.pkl")
    SCALER_PATH = os.path.join(MODELS_DIR, "data_espn_scaler.pkl")
    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_ESTIMATORS = 150
    MAX_DEPTH = 10
    
    # Características extraídas de la API de ESPN
    FEATURES = [
        'rendimiento_local_forma',
        'rendimiento_vis_forma',
        'promedio_goles_local',
        'promedio_goles_vis'
    ]
    TARGETS = ['target_ganador', 'target_ambos_anotan', 'target_mas_2_5']
