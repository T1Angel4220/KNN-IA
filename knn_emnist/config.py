"""
Módulo de Configuración Central para Clasificador KNN - EMNIST Digits.

Define los hiperparámetros del algoritmo, rutas dinámicas de conjuntos de datos
y configuraciones de ejecución por lotes (batch processing).
"""

import os
from pathlib import Path

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Carga de variables desde .env (si existe) sin requerir dependencias externas
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

# Rutas de los conjuntos de datos EMNIST (Configurables vía entorno o por defecto en data/)
RUTA_TRAIN = os.getenv("EMNIST_TRAIN_PATH", str(DATA_DIR / "emnist-digits-train.csv"))
RUTA_TEST  = os.getenv("EMNIST_TEST_PATH",  str(DATA_DIR / "emnist-digits-test.csv"))

# Hiperparámetros de K-Nearest Neighbors
K_VECINOS    = 5            # Número de vecinos más cercanos (k)
METRICA      = 'euclidean'  # 'euclidean', 'manhattan', 'cosine'
PESO_VECINOS = 'uniforme'   # 'uniforme' o 'distancia' (ponderación inversa)

# Límites de muestras para evaluación y control de rendimiento
# (Colocar en None para procesar el dataset completo)
MAX_TRAIN = 10_000
MAX_TEST  = 2_000

# Tamaño de lote para inferencia vectorizada (evita desbordamiento de RAM)
BATCH_PREDICCION = 200

# Número de clases esperadas (dígitos 0 al 9)
N_CLASES = 10
