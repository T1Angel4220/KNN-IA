"""
Módulo de Carga y Preprocesamiento de Datos EMNIST Digits.

Responsabilidad:
  - Leer conjuntos de entrenamiento y prueba en formato CSV.
  - Separar etiquetas de píxeles y aplicar muestreo si se especifica límite.
  - Normalizar intensidades de píxeles al rango [0.0, 1.0].
"""

import os
from typing import Tuple
import numpy as np
from config import RUTA_TRAIN, RUTA_TEST, MAX_TRAIN, MAX_TEST


def cargar_datos() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Carga los conjuntos de datos EMNIST Digits desde las rutas configuradas.

    Retorna:
        Tuple:
            - X_train (np.ndarray): Píxeles de entrenamiento con forma (n_muestras, n_features).
            - y_train (np.ndarray): Etiquetas enteras de entrenamiento.
            - X_test (np.ndarray): Píxeles de prueba con forma (n_muestras, n_features).
            - y_test (np.ndarray): Etiquetas enteras de prueba.
            - n_features (int): Número de atributos por muestra (ej. 784 para 28x28).
    """
    print("=" * 55)
    print("  CARGA DEL DATASET EMNIST DIGITS")
    print("=" * 55)

    if not os.path.exists(RUTA_TRAIN):
        raise FileNotFoundError(
            f"\n[ERROR] No se encontró el dataset de entrenamiento en:\n  -> {RUTA_TRAIN}\n"
            "Verifique haber colocado los CSV en la carpeta 'data/' o configure la ruta en '.env'."
        )

    if not os.path.exists(RUTA_TEST):
        raise FileNotFoundError(
            f"\n[ERROR] No se encontró el dataset de prueba en:\n  -> {RUTA_TEST}\n"
            "Verifique haber colocado los CSV en la carpeta 'data/' o configure la ruta en '.env'."
        )

    print("\nCargando datos de entrenamiento...")
    datos_train = np.loadtxt(RUTA_TRAIN, delimiter=',', dtype='float32')

    print("Cargando datos de testeo...")
    datos_test = np.loadtxt(RUTA_TEST, delimiter=',', dtype='float32')

    # La primera columna contiene la etiqueta numérica de la clase
    y_train = datos_train[:, 0].astype(int)
    X_train = datos_train[:, 1:].astype('float32')

    y_test = datos_test[:, 0].astype(int)
    X_test = datos_test[:, 1:].astype('float32')

    # Submuestreo para optimizar tiempos de ejecución si está configurado
    if MAX_TRAIN is not None and MAX_TRAIN < len(X_train):
        X_train = X_train[:MAX_TRAIN]
        y_train = y_train[:MAX_TRAIN]
        print(f"\n[AVISO] Usando {MAX_TRAIN} muestras de entrenamiento (MAX_TRAIN)")

    if MAX_TEST is not None and MAX_TEST < len(X_test):
        X_test = X_test[:MAX_TEST]
        y_test = y_test[:MAX_TEST]
        print(f"[AVISO] Usando {MAX_TEST} muestras de testeo (MAX_TEST)")

    n_features = X_train.shape[1]

    print(f"\n  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")
    print(f"  n_features (columnas) : {n_features}")
    print(f"  Clases (dígitos)      : {np.unique(y_train)}")

    return X_train, y_train, X_test, y_test, n_features


def normalizar(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Escala los valores de píxeles del rango estándar [0, 255] al rango [0.0, 1.0].

    Parámetros:
        X_train (np.ndarray): Conjunto de entrenamiento en escala de grises.
        X_test (np.ndarray): Conjunto de prueba en escala de grises.

    Retorna:
        Tuple[np.ndarray, np.ndarray]: Matrices normalizadas de tipo float32.
    """
    X_train_norm = X_train / 255.0
    X_test_norm  = X_test  / 255.0

    print("\nNormalización aplicada: píxeles / 255.0")
    print(f"  Rango X_train -> min: {X_train_norm.min():.2f}  max: {X_train_norm.max():.2f}")

    return X_train_norm, X_test_norm
