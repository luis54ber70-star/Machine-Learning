import sys
from src.ingest import ingest_raw_data
from src.features import process_features
from src.train import run_training

def main():
    print("=== INICIANDO PIPELINE MLOPS AUTOMÁTICO (ESPN) ===")
    try:
        ingest_raw_data()
        process_features()
        run_training()
        print("=== PIPELINE EJECUTADO COMPLETO Y LISTO PARA LA API ===")
    except Exception as e:
        print(f"[ERROR]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
