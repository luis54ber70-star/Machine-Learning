import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import Config

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Clase para procesamiento de características y creación de targets."""
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
    
    def load_data(self) -> pd.DataFrame:
        """Carga y valida los datos crudos."""
        if not os.path.exists(Config.RAW_DATA_PATH):
            raise FileNotFoundError(
                f"No se encuentra el archivo de datos crudos: {Config.RAW_DATA_PATH}. "
                "Ejecuta primero ingest.py"
            )
        
        self.df = pd.read_csv(Config.RAW_DATA_PATH)
        logger.info(f"Datos cargados exitosamente: {len(self.df)} registros")
        
        # Validar columnas requeridas
        required_cols = ['goles_local', 'goles_visitante']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en los datos: {missing_cols}")
        
        return self.df
    
    def create_targets(self) -> pd.DataFrame:
        """Crea las variables objetivo."""
        logger.info("Creando variables objetivo...")
        
        # Target: Ganador (1=Local, 2=Visitante, 0=Empate)
        self.df['target_ganador'] = np.where(
            self.df['goles_local'] > self.df['goles_visitante'], 1,
            np.where(self.df['goles_local'] < self.df['goles_visitante'], 2, 0)
        )
        
        # Target: Ambos anotan (1=Sí, 0=No)
        self.df['target_ambos_anotan'] = np.where(
            (self.df['goles_local'] > 0) & (self.df['goles_visitante'] > 0), 1, 0
        )
        
        # Target: Más/Menos 2.5 goles (1=Más, 0=Menos)
        self.df['target_mas_2_5'] = np.where(
            (self.df['goles_local'] + self.df['goles_visitante']) > 2.5, 1, 0
        )
        
        # Verificar balance de clases
        for target in Config.TARGETS:
            class_counts = self.df[target].value_counts()
            logger.info(f"Distribución de {target}: {class_counts.to_dict()}")
        
        return self.df
    
    def split_and_scale_data(self) -> tuple:
        """Divide y escala los datos."""
        logger.info("Dividiendo datos en entrenamiento y prueba...")
        
        X = self.df[Config.FEATURES]
        y = self.df[Config.TARGETS]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y['target_ganador']
        )
        
        logger.info(f"Datos de entrenamiento: {len(self.X_train)} registros")
        logger.info(f"Datos de prueba: {len(self.X_test)} registros")
        
        # Escalar características
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def save_artifacts(self):
        """Guarda los artefactos del procesamiento."""
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        
        # Guardar datos procesados
        self.df.to_csv(Config.PROCESSED_DATA_PATH, index=False)
        logger.info(f"Datos procesados guardados en: {Config.PROCESSED_DATA_PATH}")
        
        # Guardar scaler
        joblib.dump(self.scaler, Config.SCALER_PATH)
        logger.info(f"Scaler guardado en: {Config.SCALER_PATH}")
        
        # Guardar datos de validación
        validation_data = {
            'X_test': self.X_test,
            'y_test': self.y_test,
            'feature_names': Config.FEATURES,
            'target_names': Config.TARGETS
        }
        joblib.dump(validation_data, os.path.join(Config.MODELS_DIR, "validation_data.pkl"))
        logger.info("Datos de validación guardados")


def process_features() -> None:
    """Función principal de procesamiento de características."""
    try:
        processor = FeatureProcessor()
        processor.load_data()
        processor.create_targets()
        processor.split_and_scale_data()
        processor.save_artifacts()
        logger.info("Procesamiento de características completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en procesamiento de características: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Configurar logging si se ejecuta standalone
    logging.basicConfig(level=logging.INFO)
    process_features()
