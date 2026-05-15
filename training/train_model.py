import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuración de rutas
BASE_DATA_PATH = "data/processed/sports_data.csv"
FEEDBACK_DATA_PATH = "data/feedback.csv"
MODEL_PATH = "models/sports_model.pkl"

print("--- INICIANDO ENTRENAMIENTO CONTINUO ---")

# 1. CARGA DE DATOS
# Cargamos los datos base
if not os.path.exists(BASE_DATA_PATH):
    print(f"ERROR: No se encontró el archivo base en {BASE_DATA_PATH}")
    exit(1)

data = pd.read_csv(BASE_DATA_PATH)
print(f"Datos base cargados: {len(data)} registros.")

# 2. INTEGRACIÓN DE FEEDBACK (Aquí es donde el modelo mejora)
# Si existe el archivo de feedback con resultados reales, lo combinamos
if os.path.exists(FEEDBACK_DATA_PATH):
    feedback_data = pd.read_csv(FEEDBACK_DATA_PATH)
    # Filtramos solo los registros que ya tienen el resultado real (actual_outcome)
    # Suponiendo que tu feedback tiene las mismas columnas que tu sports_data
    new_data = feedback_data.dropna(subset=['home_win']) 
    
    if len(new_data) > 0:
        print(f"Añadiendo {len(new_data)} nuevos registros de feedback...")
        data = pd.concat([data, new_data], ignore_index=True).drop_duplicates()
        # Opcional: Guardar la base de datos actualizada para la próxima vez
        data.to_csv(BASE_DATA_PATH, index=False)
    else:
        print("No hay feedback nuevo con resultados reales todavía.")

# 3. PREPARACIÓN
X = data.drop(columns=["home_win"])
y = data["home_win"]

# Verificamos que tengamos suficientes datos para entrenar
if len(data) < 10:
    print("Datos insuficientes para entrenar.")
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. ENTRENAMIENTO
# Usamos RandomForest con los parámetros que tenías
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

print("Entrenando el modelo...")
model.fit(X_train, y_train)

# 5. EVALUACIÓN
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"PRECISIÓN DEL MODELO: {accuracy:.4f}")

# 6. GUARDADO INTELIGENTE
# Creamos la carpeta models si no existe
os.makedirs("models", exist_ok=True)

# Opcional: Solo guardar si la precisión es aceptable
if accuracy > 0.5: # Ajusta este umbral según tu necesidad
    joblib.dump(model, MODEL_PATH)
    print(f"MODELO GUARDADO EXITOSAMENTE en {MODEL_PATH}")
else:
    print("ADVERTENCIA: El modelo no mejoró la precisión mínima. No se sobrescribió.")

print("--- FIN DEL PROCESO ---")
