import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from config import Config

def run_training() -> None:
    print("[TRAIN] Entrenando Clasificador Multietiqueta con datos de ESPN...")
    df = pd.read_csv(Config.PROCESSED_DATA_PATH)
    scaler = joblib.load(Config.SCALER_PATH)
    
    X = df[Config.FEATURES]
    y = df[Config.TARGETS]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=Config.N_ESTIMATORS,
        max_depth=Config.MAX_DEPTH,
        random_state=Config.RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluación
    preds = model.predict(X_test_scaled)
    print("\n--- METRICAS DE CONFIANZA DEL MODELO ---")
    for i, target_name in enumerate(Config.TARGETS):
        acc = accuracy_score(y_test.iloc[:, i], preds[:, i])
        print(f"Precisión (Accuracy) en {target_name}: {acc * 100:.2f}%")
        
    joblib.dump(model, Config.MODEL_PATH)
    print(f"\n[TRAIN] Binario exportado exitosamente en: {Config.MODEL_PATH}")

if __name__ == "__main__":
    run_training()
