import pandas as pd
import joblib
import os

def ejecutar_predicciones_jornada():
    # Verificar que el modelo exista
    if not os.path.exists('models/modelo_futbol.pkl'):
        raise FileNotFoundError("El modelo entrenado no existe. Ejecuta primero train.py")
        
    model = joblib.load('models/modelo_futbol.pkl')
    scaler = joblib.load('models/scaler_futbol.pkl')
    
    # Partidos de la nueva jornada a pronosticar
    partidos_jornada = [
        {
            'partido': 'Real Madrid vs Barcelona',
            'rendimiento_local_5_partidos': 0.85, 'rendimiento_vis_5_partidos': 0.80,
            'xg_local_promedio': 2.3, 'xg_visitante_promedio': 2.1,
            'goles_concedidos_local': 0.8, 'goles_concedidos_vis': 0.9
        },
        {
            'partido': 'Manchester City vs Chelsea',
            'rendimiento_local_5_partidos': 0.90, 'rendimiento_vis_5_partidos': 0.50,
            'xg_local_promedio': 2.5, 'xg_visitante_promedio': 1.2,
            'goles_concedidos_local': 0.6, 'goles_concedidos_vis': 1.5
        }
    ]
    
    mapeo_ganador = {0: "Empate (X)", 1: "Local (1)", 2: "Visitante (2)"}
    
    reporte = "=== REPORTE AUTOMÁTICO DE PREDICCIONES DE FÚTBOL ===\n\n"
    
    for p in partidos_jornada:
        # Remover llave de texto para el modelo
        features = {k: v for k, v in p.items() if k != 'partido'}
        df_partido = pd.DataFrame([features])
        
        # Escalar e inferir
        scaled_features = scaler.transform(df_partido)
        preds = model.predict(scaled_features)[0]
        probs = model.predict_proba(scaled_features)
        
        # Formatear outputs utilizando las probabilidades del array correspondiente
        p_ganador = mapeo_ganador[preds[0]]
        prob_ganador = probs[0][0][preds[0]] * 100
        
        p_ambos = "Sí" if preds[1] == 1 else "No"
        prob_ambos = probs[1][0][preds[1]] * 100
        
        p_goles = "Más de 2.5" if preds[2] == 1 else "Menos de 2.5"
        prob_goles = probs[2][0][preds[2]] * 100
        
        reporte += f"Partido: {p['partido']}\n"
        reporte += f"  • Pronóstico Ganador: {p_ganador} ({prob_ganador:.1f}%)\n"
        reporte += f"  • Ambos Anotan: {p_ambos} ({prob_ambos:.1f}%)\n"
        reporte += f"  • Línea de Goles: {p_goles} ({prob_goles:.1f}%)\n"
        reporte += "-" * 40 + "\n"
        
    # Guardar reporte final
    with open('reporte_predicciones.txt', 'w', encoding='utf-8') as f:
        f.write(reporte)
    
    print("Reporte 'reporte_predicciones.txt' generado de forma exitosa.")

if __name__ == "__main__":
    ejecutar_predicciones_jornada()
