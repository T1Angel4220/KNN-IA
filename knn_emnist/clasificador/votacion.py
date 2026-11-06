"""
Módulo de Selección y Votación para K-Vecinos Más Cercanos (KNN).

Responsabilidad:
  - Seleccionar los K índices con menor distancia.
  - Implementar votación por mayoría simple (uniforme).
  - Implementar votación ponderada por la inversa de la distancia.
"""

from typing import Tuple
import numpy as np


def obtener_k_vecinos(distancias: np.ndarray,
                      y_train: np.ndarray,
                      k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identifica los K vecinos con menor distancia respecto a la muestra consultada.

    Parámetros:
        distancias (np.ndarray): Vector de distancias a los puntos de entrenamiento (n_train,).
        y_train (np.ndarray): Vector de etiquetas reales de entrenamiento (n_train,).
        k (int): Cantidad de vecinos a seleccionar.

    Retorna:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - indices_k: Índices de los K vecinos más cercanos.
            - etiquetas_k: Clases asociadas a los K vecinos.
            - distancias_k: Distancias de los K vecinos.
    """
    indices_ordenados = np.argsort(distancias)
    indices_k         = indices_ordenados[:k]

    etiquetas_k  = y_train[indices_k]
    distancias_k = distancias[indices_k]

    return indices_k, etiquetas_k, distancias_k


def voto_uniforme(etiquetas_k: np.ndarray, n_clases: int) -> int:
    """
    Determina la clase ganadora mediante votación por mayoría simple (cada vecino vale 1 voto).

    Parámetros:
        etiquetas_k (np.ndarray): Etiquetas de los K vecinos más cercanos.
        n_clases (int): Cantidad total de clases posibles.

    Retorna:
        int: Clase predicha (etiqueta con mayor frecuencia).
    """
    conteos = np.bincount(etiquetas_k, minlength=n_clases)
    return int(np.argmax(conteos))


def voto_por_distancia(etiquetas_k: np.ndarray,
                       distancias_k: np.ndarray,
                       n_clases: int) -> int:
    """
    Determina la clase mediante votación ponderada por la inversa de la distancia:
        peso = 1 / (distancia + epsilon)

    Vecinos más cercanos aportan mayor peso al conteo de su clase.

    Parámetros:
        etiquetas_k (np.ndarray): Etiquetas de los K vecinos más cercanos.
        distancias_k (np.ndarray): Distancias asociadas a dichos vecinos.
        n_clases (int): Cantidad total de clases posibles.

    Retorna:
        int: Clase predicha con mayor peso acumulado.
    """
    EPSILON = 1e-10
    pesos = 1.0 / (distancias_k + EPSILON)

    pesos_por_clase = np.zeros(n_clases, dtype=float)
    for etiqueta, peso in zip(etiquetas_k, pesos):
        pesos_por_clase[etiqueta] += peso

    return int(np.argmax(pesos_por_clase))
