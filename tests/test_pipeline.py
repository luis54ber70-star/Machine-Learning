"""
Tests unitarios para el pipeline de Machine Learning.
Ejecutar con: pytest tests/test_pipeline.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import Mock, patch

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.ingest import ESPNDataIngestor, ingest_raw_data
from src.features import FeatureProcessor, process_features
from src.train import FootballModelTrainer, run_training
from src.predict import FootballPredictor, create_example_input


class TestConfig:
    """Tests para la configuración."""

    def test_directories_exist(self):
        """Verifica que los directorios necesarios existan."""
        assert os.path.exists(Config.DATA_DIR), f"DATA_DIR no existe: {Config.DATA_DIR}"
        assert os.path.exists(Config.MODELS_DIR), f"MODELS_DIR no existe: {Config.MODELS_DIR}"
        assert os.path.exists(Config.LOGS_DIR), f"LOGS_DIR no existe: {Config.LOGS_DIR}"

    def test_features_list(self):
        """Verifica que las características estén definidas."""
        assert len(Config.FEATURES) == 4
        assert 'rendimiento_local_forma' in Config.FEATURES

    def test_targets_list(self):
        """Verifica que los targets estén definidos."""
        assert len(Config.TARGETS) == 3
        assert 'target_ganador' in Config.TARGETS


class TestIngest:
    """Tests para la ingesta de datos."""

    @patch('src.ingest.requests.get')
    def test_fetch_league_data_success(self, mock_get):
        """Simula una respuesta exitosa de la API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'events': [{'id': '1', 'name': 'Test Match'}]}
        mock_get.return_value = mock_response

        ingestor = ESPNDataIngestor()
        events = ingestor._fetch_league_data('eng.1')
        assert len(events) == 1
        assert events[0]['name'] == 'Test Match'

    @patch('src.ingest.requests.get')
    def test_fetch_league_data_failure(self, mock_get):
        """Simula un error de conexión."""
        mock_get.side_effect = Exception("Connection error")
        ingestor = ESPNDataIngestor()
        events = ingestor._fetch_league_data('eng.1')
        assert events == []

    def test_process_form_empty(self):
        """Procesa cadena de forma vacía."""
        ingestor = ESPNDataIngestor()
        result = ingestor._process_form("")
        assert result['win_rate'] == 0.5
        assert result['total_matches'] == 0

    def test_process_form_valid(self):
        """Procesa cadena de forma válida."""
        ingestor = ESPNDataIngestor()
        result = ingestor._process_form("W-D-L-W-W")
        assert result['win_rate'] == 0.6  # 3 victorias de 5
        assert result['total_matches'] == 5

    def test_generate_sample_data(self):
        """Genera datos de ejemplo correctamente."""
        ingestor = ESPNDataIngestor()
        ingestor._generate_sample_data()
        assert len(ingestor.partidos_procesados) == 3


class TestFeatures:
    """Tests para el procesamiento de características."""

    @pytest.fixture
    def sample_dataframe(self):
        """Crea un DataFrame de prueba."""
        return pd.DataFrame({
            'goles_local': [2, 1, 3],
            'goles_visitante': [1, 1, 0],
            'rendimiento_local_forma': [0.8, 0.7, 0.9],
            'rendimiento_vis_forma': [0.6, 0.7, 0.5],
            'promedio_goles_local': [2.1, 1.9, 2.5],
            'promedio_goles_vis': [1.8, 1.9, 1.5],
            'partido_name': ['Match1', 'Match2', 'Match3']
        })

    def test_create_targets(self, sample_dataframe):
        """Verifica la creación de variables objetivo."""
        processor = FeatureProcessor()
        processor.df = sample_dataframe.copy()
        processor.create_targets()
        assert 'target_ganador' in processor.df.columns
        assert 'target_ambos_anotan' in processor.df.columns
        assert 'target_mas_2_5' in processor.df.columns
        # Primer partido: local=2, visit=1 -> ganador=1 (local)
        assert processor.df['target_ganador'].iloc[0] == 1
        # Segundo partido: 1-1 -> ambos anotan=1, más 2.5? 2 goles->0
        assert processor.df['target_ambos_anotan'].iloc[1] == 1
        assert processor.df['target_mas_2_5'].iloc[1] == 0

    def test_split_and_scale(self, sample_dataframe):
        """Verifica división y escalado."""
        processor = FeatureProcessor()
        processor.df = sample_dataframe.copy()
        processor.create_targets()
        X_train, X_test, y_train, y_test = processor.split_and_scale_data()
        assert X_train.shape[0] == 2  # 80% de 3 -> 2 (train)
        assert X_test.shape[0] == 1   # 20% -> 1
        assert X_train.shape[1] == 4
        # Verificar que está escalado (media~0, std~1)
        assert abs(np.mean(X_train[0])) < 1  # aproximadamente


class TestTrain:
    """Tests para el entrenamiento."""

    @pytest.fixture
    def setup_training_data(self, tmp_path):
        """Crea datos procesados temporales para entrenamiento."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # Crear DataFrame de ejemplo
        df = pd.DataFrame({
            'rendimiento_local_forma': [0.8, 0.7, 0.9, 0.6, 0.5],
            'rendimiento_vis_forma': [0.6, 0.7, 0.5, 0.8, 0.4],
            'promedio_goles_local': [2.1, 1.9, 2.5, 1.8, 2.0],
            'promedio_goles_vis': [1.8, 1.9, 1.5, 2.2, 1.6],
            'target_ganador': [1, 0, 1, 2, 0],
            'target_ambos_anotan': [1, 1, 0, 1, 0],
            'target_mas_2_5': [1, 0, 1, 1, 0]
        })
        processed_path = data_dir / "processed_espn_matches.csv"
        df.to_csv(processed_path, index=False)
        # Crear scaler ficticio
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = df[Config.FEATURES]
        scaler.fit(X)
        scaler_path = models_dir / "data_espn_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        return tmp_path

    def test_load_data_and_train(self, monkeypatch, setup_training_data):
        """Verifica carga de datos y entrenamiento."""
        # Modificar rutas de Config para usar directorio temporal
        tmp = setup_training_data
        monkeypatch.setattr(Config, 'PROCESSED_DATA_PATH', str(tmp / "data" / "processed_espn_matches.csv"))
        monkeypatch.setattr(Config, 'SCALER_PATH', str(tmp / "models" / "data_espn_scaler.pkl"))
        monkeypatch.setattr(Config, 'MODELS_DIR', str(tmp / "models"))
        # También ajustar otros paths que puedan interferir (no es necesario para este test)
        trainer = FootballModelTrainer()
        trainer.load_data()
        trainer.train_model()
        assert trainer.model is not None
        # Evaluar
        metrics = trainer.evaluate_model()
        assert 'target_ganador' in metrics
        assert metrics['target_ganador']['accuracy'] >= 0


class TestPredict:
    """Tests para la predicción."""

    @pytest.fixture
    def setup_model_and_scaler(self, tmp_path):
        """Crea modelo y scaler temporales."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # Crear scaler
        scaler = StandardScaler()
        fake_X = np.random.rand(10, 4)
        scaler.fit(fake_X)
        joblib.dump(scaler, models_dir / "data_espn_scaler.pkl")
        # Crear modelo (multioutput)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        y_multi = np.random.randint(0, 3, (10, 3))  # 3 targets
        model.fit(fake_X, y_multi)
        joblib.dump(model, models_dir / "football_espn_model.pkl")
        return tmp_path

    def test_predict_match(self, monkeypatch, setup_model_and_scaler):
        """Verifica que la predicción retorne el formato esperado."""
        tmp = setup_model_and_scaler
        monkeypatch.setattr(Config, 'MODEL_PATH', str(tmp / "models" / "football_espn_model.pkl"))
        monkeypatch.setattr(Config, 'SCALER_PATH', str(tmp / "models" / "data_espn_scaler.pkl"))

        predictor = FootballPredictor()
        input_data = create_example_input()
        result = predictor.predict_match(input_data)
        assert 'ganador' in result
        assert 'ambos_anotan' in result
        assert 'mas_2_5_goles' in result
        assert 'resultado' in result['ganador']
        assert 'probabilidad' in result['ganador']

    def test_predict_batch(self, monkeypatch, setup_model_and_scaler):
        """Verifica predicción por lotes."""
        tmp = setup_model_and_scaler
        monkeypatch.setattr(Config, 'MODEL_PATH', str(tmp / "models" / "football_espn_model.pkl"))
        monkeypatch.setattr(Config, 'SCALER_PATH', str(tmp / "models" / "data_espn_scaler.pkl"))

        predictor = FootballPredictor()
        matches = [create_example_input(), create_example_input()]
        results = predictor.predict_batch(matches)
        assert len(results) == 2
