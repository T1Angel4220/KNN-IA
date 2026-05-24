"""
Módulo: datos/cargar_datos.py
Responsabilidad:
    - Leer los archivos CSV del dataset EMNIST Digits
    - Separar píxeles (X) y etiquetas (y)
    - Normalizar los valores de píxel al rango [0, 1]
    - Aplicar un subconjunto opcional para acelerar pruebas
"""

# pyrefly: ignore [missing-import]
import numpy as np
from config import RUTA_TRAIN, RUTA_TEST, MAX_TRAIN, MAX_TEST


# ──────────────────────────────────────────────────────────────
def cargar_datos() -> tuple:
    """
    Carga el dataset EMNIST Digits desde los archivos CSV configurados.

    Retorna
    -------
    X_train : ndarray (n_train, 784)   - Píxeles de entrenamiento
    y_train : ndarray (n_train,)       - Etiquetas de entrenamiento
    X_test  : ndarray (n_test,  784)   - Píxeles de testeo
    y_test  : ndarray (n_test,)        - Etiquetas de testeo
    n_features : int                   - Número de características (784)
    """
    print("=" * 55)
    print("  CARGA DEL DATASET EMNIST DIGITS")
    print("=" * 55)

    print("\nCargando datos de entrenamiento...")
    datos_train = np.loadtxt(RUTA_TRAIN, delimiter=',', dtype='float32')

    print("Cargando datos de testeo...")
    datos_test = np.loadtxt(RUTA_TEST, delimiter=',', dtype='float32')

    # -- Separar etiquetas de píxeles --
    y_train = datos_train[:, 0].astype(int)
    X_train = datos_train[:, 1:].astype('float32')

    y_test = datos_test[:, 0].astype(int)
    X_test = datos_test[:, 1:].astype('float32')

    # -- Subconjunto opcional (para agilizar pruebas con KNN) --
    if MAX_TRAIN is not None and MAX_TRAIN < len(X_train):
        X_train = X_train[:MAX_TRAIN]
        y_train = y_train[:MAX_TRAIN]
        print(f"\n[AVISO] Usando solo {MAX_TRAIN} muestras de entrenamiento (MAX_TRAIN)")

    if MAX_TEST is not None and MAX_TEST < len(X_test):
        X_test = X_test[:MAX_TEST]
        y_test = y_test[:MAX_TEST]
        print(f"[AVISO] Usando solo {MAX_TEST} muestras de testeo (MAX_TEST)")

    n_features = X_train.shape[1]

    print(f"\n  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")
    print(f"  n_features (columnas) : {n_features}")
    print(f"  Clases (dígitos)      : {np.unique(y_train)}")

    return X_train, y_train, X_test, y_test, n_features


# ──────────────────────────────────────────────────────────────
def normalizar(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Normaliza los valores de píxel dividiéndolos entre 255.

    Parámetros
    ----------
    X_train, X_test : arrays de píxeles crudos [0, 255]

    Retorna
    -------
    X_train, X_test normalizados al rango [0, 1]
    """
    X_train_norm = X_train / 255.0
    X_test_norm  = X_test  / 255.0

    print("\nNormalización aplicada: píxeles / 255.0")
    print(f"  Rango X_train -> min: {X_train_norm.min():.2f}  "
          f"max: {X_train_norm.max():.2f}")

    return X_train_norm, X_test_norm
