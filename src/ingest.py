import os
import pandas as pd
import requests
import logging
from typing import List, Dict, Any
from datetime import datetime
from config import Config

# Configurar logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ESPNDataIngestor:
    """Clase para ingerir datos de la API de ESPN."""
    
    def __init__(self):
        self.base_url = Config.ESPN_API_BASE_URL
        self.ligas = Config.LIGAS
        self.partidos_procesados: List[Dict[str, Any]] = []
    
    def _fetch_league_data(self, liga: str) -> List[Dict]:
        """
        Obtiene datos de una liga específica.
        
        Args:
            liga: Código de la liga (ej: "eng.1")
            
        Returns:
            Lista de eventos de la liga
        """
        url = f"{self.base_url}/{liga}/scoreboard"
        params = {
            "limit": 200,
            "dates": datetime.now().strftime("%Y%m%d")
        }
        
        try:
            logger.info(f"Conectando a la API de ESPN para liga {liga}...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Datos obtenidos exitosamente para liga {liga}")
            return data.get("events", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al conectar con liga {liga}: {e}")
            return []
        except ValueError as e:
            logger.error(f"Error al parsear JSON para liga {liga}: {e}")
            return []
    
    def _process_event(self, event: Dict) -> Dict[str, Any]:
        """
        Procesa un evento de partido individual.
        
        Args:
            event: Evento de la API de ESPN
            
        Returns:
            Diccionario con datos procesados del partido
        """
        competitions = event.get("competitions", [])
        if not competitions:
            return None
        
        competition = competitions[0]
        teams = competition.get("competitors", [])
        if len(teams) < 2:
            return None
        
        home_team = next((t for t in teams if t.get("homeAway") == "home"), None)
        away_team = next((t for t in teams if t.get("homeAway") == "away"), None)
        
        if not home_team or not away_team:
            return None
        
        # Procesar estado de forma
        form_local = self._process_form(home_team.get("form", ""))
        form_vis = self._process_form(away_team.get("form", ""))
        
        # Procesar estadísticas
        home_stats = competition.get("headlines", [{}])[0] if competition.get("headlines") else {}
        away_stats = competition.get("headlines", [{}])[1] if len(competition.get("headlines", [])) > 1 else {}
        
        return {
            'partido_name': event.get("name", "Partido Desconocido"),
            'goles_local': int(home_team.get("score", 0)),
            'goles_visitante': int(away_team.get("score", 0)),
            'rendimiento_local_forma': float(form_local.get("win_rate", 0.5)),
            'rendimiento_vis_forma': float(form_vis.get("win_rate", 0.5)),
            'promedio_goles_local': float(home_stats.get("description", "1.5").replace("GPG:", "").strip()) if home_stats else 1.5,
            'promedio_goles_vis': float(away_stats.get("description", "1.2").replace("GPG:", "").strip()) if away_stats else 1.2
        }
    
    def _process_form(self, form_string: str) -> Dict[str, float]:
        """
        Procesa la cadena de forma (ej: "W-D-L-W-W").
        
        Args:
            form_string: Cadena de forma del equipo
            
        Returns:
            Diccionario con métricas de forma
        """
        if not form_string:
            return {"win_rate": 0.5, "total_matches": 0}
        
        results = form_string.split("-")
        total = len(results)
        wins = results.count("W")
        
        return {
            "win_rate": float(wins / total) if total > 0 else 0.5,
            "total_matches": total
        }
    
    def ingest_data(self) -> pd.DataFrame:
        """
        Ejecuta la ingesta completa de datos.
        
        Returns:
            DataFrame con datos procesados
        """
        logger.info("Iniciando proceso de ingesta de datos...")
        
        for liga in self.ligas:
            events = self._fetch_league_data(liga)
            
            for event in events:
                partido = self._process_event(event)
                if partido:
                    self.partidos_procesados.append(partido)
        
        if not self.partidos_procesados:
            logger.warning("No se encontraron partidos. Usando datos de ejemplo...")
            self._generate_sample_data()
        
        df = pd.DataFrame(self.partidos_procesados)
        logger.info(f"Total de partidos procesados: {len(df)}")
        
        return df
    
    def _generate_sample_data(self):
        """Genera datos de ejemplo cuando no hay conexión a la API."""
        logger.info("Generando datos de ejemplo...")
        sample_data = [
            {
                'partido_name': 'Manchester City vs Liverpool',
                'goles_local': 2,
                'goles_visitante': 1,
                'rendimiento_local_forma': 0.8,
                'rendimiento_vis_forma': 0.6,
                'promedio_goles_local': 2.1,
                'promedio_goles_vis': 1.8
            },
            {
                'partido_name': 'Real Madrid vs Barcelona',
                'goles_local': 1,
                'goles_visitante': 1,
                'rendimiento_local_forma': 0.7,
                'rendimiento_vis_forma': 0.7,
                'promedio_goles_local': 1.9,
                'promedio_goles_vis': 1.9
            },
            {
                'partido_name': 'Bayern Munich vs Dortmund',
                'goles_local': 3,
                'goles_visitante': 0,
                'rendimiento_local_forma': 0.9,
                'rendimiento_vis_forma': 0.5,
                'promedio_goles_local': 2.5,
                'promedio_goles_vis': 1.5
            }
        ]
        self.partidos_procesados.extend(sample_data)


def ingest_raw_data() -> None:
    """Función principal de ingesta que orquesta el proceso."""
    try:
        ingestor = ESPNDataIngestor()
        df = ingestor.ingest_data()
        
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        df.to_csv(Config.RAW_DATA_PATH, index=False)
        logger.info(f"Datos crudos guardados correctamente en: {Config.RAW_DATA_PATH}")
        
    except Exception as e:
        logger.error(f"Error crítico en la ingesta: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    ingest_raw_data()
