import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    # Crear directorios si no existen
    for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_espn_matches.csv")
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_espn_matches.csv")
    MODEL_PATH = os.path.join(MODELS_DIR, "football_espn_model.pkl")
    SCALER_PATH = os.path.join(MODELS_DIR, "data_espn_scaler.pkl")
    
    # Configuración del modelo
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
    
    # Configuración de API ESPN
    ESPN_API_BASE_URL = os.getenv("ESPN_API_BASE_URL", "https://site.api.espn.com/apis/site/v2/sports/soccer")
    ESPN_API_KEY = os.getenv("ESPN_API_KEY", "")
    
    # Ligas soportadas
    LIGAS = ["eng.1", "esp.1"]
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.path.join(LOGS_DIR, "pipeline.log")
