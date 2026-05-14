#!/usr/bin/env python3
"""
Pipeline completo de Machine Learning para predicción de fútbol.
Ejecuta: Ingesta -> Features -> Entrenamiento -> Evaluación
"""

import sys
import logging
import argparse
from datetime import datetime
from config import Config


def setup_logging():
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Pipeline MLOps para predicción de fútbol con datos de ESPN'
    )
    parser.add_argument(
        '--skip-ingest', 
        action='store_true',
        help='Omitir paso de ingesta de datos'
    )
    parser.add_argument(
        '--skip-features', 
        action='store_true',
        help='Omitir paso de procesamiento de características'
    )
    parser.add_argument(
        '--skip-train', 
        action='store_true',
        help='Omitir paso de entrenamiento'
    )
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Ejecutar en modo prueba (usa datos de ejemplo)'
    )
    return parser.parse_args()


def main():
    """Función principal del pipeline."""
    logger = setup_logging()
    args = parse_arguments()
    
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE MLOPS AUTOMÁTICO (ESPN)")
    logger.info(f"Inicio: {start_time}")
    logger.info("=" * 60)
    
    try:
        # Paso 1: Ingesta de datos
        if not args.skip_ingest:
            logger.info("\n[1/3] INICIANDO INGESTA DE DATOS...")
            from src.ingest import ingest_raw_data
            ingest_raw_data()
            logger.info("[1/3] INGESTA COMPLETADA ✅")
        else:
            logger.info("[1/3] INGESTA OMITIDA ⏭️")
        
        # Paso 2: Procesamiento de características
        if not args.skip_features:
            logger.info("\n[2/3] INICIANDO PROCESAMIENTO DE CARACTERÍSTICAS...")
            from src.features import process_features
            process_features()
            logger.info("[2/3] PROCESAMIENTO COMPLETADO ✅")
        else:
            logger.info("[2/3] PROCESAMIENTO OMITIDO ⏭️")
        
        # Paso 3: Entrenamiento
        if not args.skip_train:
            logger.info("\n[3/3] INICIANDO ENTRENAMIENTO DEL MODELO...")
            from src.train import run_training
            run_training()
            logger.info("[3/3] ENTRENAMIENTO COMPLETADO ✅")
        else:
            logger.info("[3/3] ENTRENAMIENTO OMITIDO ⏭️")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE EJECUTADO EXITOSAMENTE 🎯")
        logger.info(f"Duración total: {duration.total_seconds():.2f} segundos")
        logger.info(f"Finalización: {end_time}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline falló: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
