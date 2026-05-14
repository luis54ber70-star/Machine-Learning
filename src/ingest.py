import os
import pandas as pd
import requests
from config import Config

def ingest_raw_data() -> None:
    print("[INGEST] Conectando con los endpoints de la API de ESPN...")
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    # Consumimos la Premier League (eng.1) y LaLiga (esp.1) para robustecer el dataset
    ligas = ["eng.1", "esp.1"]
    partidos_procesados = []
    
    for liga in ligas:
        url = f"espn.com{liga}/scoreboard"
        try:
            response = requests.get(url, params={"limit": 200}, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[INGEST] Error al conectar con liga {liga}: {e}")
            continue

        for event in data.get("events", []):
            competitions = event.get("competitions", [])
            if not competitions:
                continue
            
            competition = competitions[0]
            teams = competition.get("competitors", [])
            if len(teams) < 2:
                continue
                
            home_team = next((t for t in teams if t.get("homeAway") == "home"), None)
            away_team = next((t for t in teams if t.get("homeAway") == "away"), None)
            
            if not home_team or not away_team:
                continue
                
            # Procesar el estado de forma ("form") provisto por ESPN (ej: "W-D-L-W-W")
            form_local = home_team.get("form", "W-D-L").split("-")
            form_vis = away_team.get("form", "W-D-L").split("-")
            
            partido = {
                'partido_name': event.get("name", "Partido Desconocido"),
                'goles_local': int(home_team.get("score", 0)),
                'goles_visitante': int(away_team.get("score", 0)),
                'rendimiento_local_forma': float(form_local.count("W") / max(len(form_local), 1)),
                'rendimiento_vis_forma': float(form_vis.count("W") / max(len(form_vis), 1)),
                'promedio_goles_local': float(home_team.get("statistics", [{"value": 1.5}])[0].get("value", 1.5)),
                'promedio_goles_vis': float(away_team.get("statistics", [{"value": 1.2}])[0].get("value", 1.2))
            }
            partidos_procesados.append(partido)
            
    df = pd.DataFrame(partidos_procesados)
    if df.empty:
        raise ValueError("[INGEST] No se pudieron extraer partidos de la API.")
        
    df.to_csv(Config.RAW_DATA_PATH, index=False)
    print(f"[INGEST] Datos crudos guardados correctamente en: {Config.RAW_DATA_PATH}")

if __name__ == "__main__":
    ingest_raw_data()
