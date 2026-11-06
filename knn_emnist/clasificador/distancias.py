"""
Módulo de Cálculo de Distancias Vectorizadas.

Responsabilidad:
  - Calcular distancias matemáticas entre vectores de características.
  - Soporta métricas estándar: Euclidiana (L2), Manhattan (L1) y Coseno.
  - Implementación vectorizada mediante `scipy.spatial.distance.cdist`.
"""

from typing import Union
import numpy as np
from scipy.spatial.distance import cdist


def calcular_distancias(X_train: np.ndarray,
                        muestra: np.ndarray,
                        metrica: str = 'euclidean') -> np.ndarray:
    """
    Calcula la distancia entre una única muestra de consulta y todos los puntos de entrenamiento.

    Parámetros:
        X_train (np.ndarray): Matriz de entrenamiento con forma (n_train, n_features).
        muestra (np.ndarray): Vector de características con forma (n_features,) o (1, n_features).
        metrica (str): Métrica de distancia ('euclidean', 'manhattan', 'cosine').

    Retorna:
        np.ndarray: Vector 1D con las distancias a cada punto de entrenamiento (n_train,).
    """
    muestra_2d = muestra.reshape(1, -1)
    distancias = cdist(muestra_2d, X_train, metric=metrica)
    return distancias.flatten()


def calcular_distancias_batch(X_train: np.ndarray,
                              X_query: np.ndarray,
                              metrica: str = 'euclidean') -> np.ndarray:
    """
    Calcula de manera matricial las distancias entre un lote de muestras de consulta
    y todos los puntos de entrenamiento.

    Parámetros:
        X_train (np.ndarray): Matriz de entrenamiento (n_train, n_features).
        X_query (np.ndarray): Lote de consulta (n_query, n_features).
        metrica (str): Métrica de distancia ('euclidean', 'manhattan', 'cosine').

    Retorna:
        np.ndarray: Matriz de distancias de dimensiones (n_query, n_train).
    """
    return cdist(X_query, X_train, metric=metrica)
