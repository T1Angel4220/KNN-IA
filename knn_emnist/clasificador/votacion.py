"""
Módulo: clasificador/votacion.py
Responsabilidad:
    - Dado un vector de distancias y las etiquetas del conjunto de
      entrenamiento, seleccionar los K vecinos más cercanos y
      determinar la clase ganadora por votación.
    - Soporte para votación UNIFORME y por DISTANCIA (ponderada).

Implementación completamente manual (sin sklearn).
"""

# pyrefly: ignore [missing-import]
import numpy as np


# ──────────────────────────────────────────────────────────────
def obtener_k_vecinos(distancias: np.ndarray,
                      y_train: np.ndarray,
                      k: int) -> tuple:
    """
    Encuentra los K vecinos más cercanos a partir de las distancias.

    Parámetros
    ----------
    distancias : ndarray (n_train,)
        Distancias de la muestra de consulta a cada punto de entrenamiento.
    y_train    : ndarray (n_train,)
        Etiquetas del conjunto de entrenamiento.
    k          : int
        Número de vecinos a seleccionar.

    Retorna
    -------
    indices_k  : ndarray (k,)  - índices de los K vecinos más cercanos
    etiquetas_k: ndarray (k,)  - etiquetas de los K vecinos
    distancias_k: ndarray (k,) - distancias de los K vecinos
    """
    # argsort devuelve índices que ORDENARÍAN el array de menor a mayor
    indices_ordenados = np.argsort(distancias)          # todos ordenados
    indices_k         = indices_ordenados[:k]           # tomar los primeros K

    etiquetas_k  = y_train[indices_k]
    distancias_k = distancias[indices_k]

    return indices_k, etiquetas_k, distancias_k


# ──────────────────────────────────────────────────────────────
def voto_uniforme(etiquetas_k: np.ndarray, n_clases: int) -> int:
    """
    Votación por mayoría simple: cada vecino tiene el mismo peso (1 voto).

    Parámetros
    ----------
    etiquetas_k : ndarray (k,) - etiquetas de los K vecinos
    n_clases    : int          - número total de clases (10 para dígitos)

    Retorna
    -------
    clase_ganadora : int - dígito predicho
    """
    conteos = np.bincount(etiquetas_k, minlength=n_clases)
    return int(np.argmax(conteos))


# ──────────────────────────────────────────────────────────────
def voto_por_distancia(etiquetas_k: np.ndarray,
                       distancias_k: np.ndarray,
                       n_clases: int) -> int:
    """
    Votación ponderada por distancia: los vecinos MÁS CERCANOS tienen
    mayor influencia en la decisión final.

    Peso de cada vecino = 1 / distancia  (evita división por cero)

    Parámetros
    ----------
    etiquetas_k  : ndarray (k,) - etiquetas de los K vecinos
    distancias_k : ndarray (k,) - distancias de los K vecinos
    n_clases     : int          - número total de clases

    Retorna
    -------
    clase_ganadora : int - dígito predicho
    """
    # Evitar división por cero: si distancia == 0, ese vecino es
    # una copia exacta -> le damos peso infinito (muy alto)
    EPSILON = 1e-10
    pesos = 1.0 / (distancias_k + EPSILON)

    # Acumular pesos por clase
    pesos_por_clase = np.zeros(n_clases, dtype=float)
    for etiqueta, peso in zip(etiquetas_k, pesos):
        pesos_por_clase[etiqueta] += peso

    return int(np.argmax(pesos_por_clase))
