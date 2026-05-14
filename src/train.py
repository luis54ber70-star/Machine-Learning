import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
from config import Config

logger = logging.getLogger(__name__)


class FootballModelTrainer:
    """Clase para entrenar modelos de predicción de fútbol."""
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
    
    def load_data(self) -> None:
        """Carga datos procesados y escalador."""
        if not os.path.exists(Config.PROCESSED_DATA_PATH):
            raise FileNotFoundError(
                f"No se encuentra el archivo de datos procesados: {Config.PROCESSED_DATA_PATH}"
            )
        
        df = pd.read_csv(Config.PROCESSED_DATA_PATH)
        self.scaler = joblib.load(Config.SCALER_PATH)
        
        X = df[Config.FEATURES]
        y = df[Config.TARGETS]
        
        # División estratificada
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE,
            stratify=y['target_ganador']
        )
        
        # Escalar datos
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"Datos cargados: {len(df)} registros totales")
        logger.info(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}")
    
    def train_model(self) -> RandomForestClassifier:
        """Entrena el modelo Random Forest."""
        logger.info("Iniciando entrenamiento del modelo...")
        
        self.model = RandomForestClassifier(
            n_estimators=Config.N_ESTIMATORS,
            max_depth=Config.MAX_DEPTH,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        
        # Entrenar para cada target
        self.model.fit(self.X_train, self.y_train)
        
        logger.info("Modelo entrenado exitosamente")
        return self.model
    
    def evaluate_model(self) -> dict:
        """Evalúa el modelo con múltiples métricas."""
        logger.info("Evaluando modelo...")
        
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)
        
        results = {}
        
        for i, target_name in enumerate(Config.TARGETS):
            y_true = self.y_test.iloc[:, i]
            y_pred_target = y_pred[:, i]
            
            # Métricas principales
            accuracy = accuracy_score(y_true, y_pred_target)
            precision = precision_score(y_true, y_pred_target, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred_target, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred_target, average='weighted', zero_division=0)
            
            results[target_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            logger.info(f"\n--- Métricas para {target_name} ---")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            
            # Reporte detallado
            logger.info(f"\nClassification Report {target_name}:")
            logger.info(f"\n{classification_report(y_true, y_pred_target, zero_division=0)}")
        
        # Validación cruzada
        logger.info("\n--- Validación Cruzada (5-fold) ---")
        cv_scores = cross_val_score(
            self.model, 
            np.concatenate([self.X_train, self.X_test]),
            pd.concat([self.y_train, self.y_test]).values,
            cv=5,
            scoring='accuracy'
        )
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        results['cross_validation'] = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std()
        }
        
        return results
    
    def save_model(self) -> None:
        """Guarda el modelo entrenado."""
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        joblib.dump(self.model, Config.MODEL_PATH)
        logger.info(f"Modelo guardado en: {Config.MODEL_PATH}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Obtiene la importancia de características."""
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': Config.FEATURES,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info("\n--- Importancia de Características ---")
        logger.info(feature_importance_df.to_string())
        
        return feature_importance_df


def run_training() -> None:
    """Función principal de entrenamiento."""
    try:
        trainer = FootballModelTrainer()
        trainer.load_data()
        trainer.train_model()
        metrics = trainer.evaluate_model()
        trainer.save_model()
        trainer.get_feature_importance()
        
        logger.info("\n=== ENTRENAMIENTO COMPLETADO EXITOSAMENTE ===")
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()
