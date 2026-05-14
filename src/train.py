import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def generar_datos_y_entrenar():
    print("Creando directorios necesarios...")
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. Simulación y guardado de datos históricos
    print("Generando datos históricos de fútbol...")
    np.random.seed(42)
    n_partidos = 1500
    
    data = {
        'rendimiento_local_5_partidos': np.random.uniform(0.2, 1.0, n_partidos),
        'rendimiento_vis_5_partidos': np.random.uniform(0.1, 0.9, n_partidos),
        'xg_local_promedio': np.random.uniform(0.8, 2.6, n_partidos),
        'xg_visitante_promedio': np.random.uniform(0.6, 2.2, n_partidos),
        'goles_concedidos_local': np.random.uniform(0.5, 2.0, n_partidos),
        'goles_concedidos_vis': np.random.uniform(0.7, 2.3, n_partidos)
    }
    df = pd.DataFrame(data)
    
    goles_local = np.random.poisson(df['xg_local_promedio'] * 1.1)
    goles_vis = np.random.poisson(df['xg_visitante_promedio'] * 0.9)
    
    df['target_ganador'] = np.where(goles_local > goles_vis, 1, np.where(goles_local < goles_vis, 2, 0))
    df['target_ambos_anotan'] = np.where((goles_local > 0) & (goles_vis > 0), 1, 0)
    df['target_mas_2_5'] = np.where((goles_local + goles_vis) > 2.5, 1, 0)
    
    df.to_csv('data/partidos_historicos.csv', index=False)
    
    # 2. Preparación y Escalado
    X = df.drop(columns=['target_ganador', 'target_ambos_anotan', 'target_mas_2_5'])
    y = df[['target_ganador', 'target_ambos_anotan', 'target_mas_2_5']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 3. Entrenamiento
    print("Entrenando el modelo Multi-Output (Random Forest)...")
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 4. Guardar artefactos
    joblib.dump(model, 'models/modelo_futbol.pkl')
    joblib.dump(scaler, 'models/scaler_futbol.pkl')
    print("¡Modelo y escalador guardados con éxito en la carpeta models/!")

if __name__ == "__main__":
    generar_datos_y_entrenar()
