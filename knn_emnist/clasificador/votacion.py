# pyrefly: ignore [missing-import]
import numpy as np

def obtener_k_vecinos(distancias: np.ndarray,
                      y_train: np.ndarray,
                      k: int) -> tuple:
    indices_ordenados = np.argsort(distancias)
    indices_k         = indices_ordenados[:k]

    etiquetas_k  = y_train[indices_k]
    distancias_k = distancias[indices_k]

    return indices_k, etiquetas_k, distancias_k

def voto_uniforme(etiquetas_k: np.ndarray, n_clases: int) -> int:
    conteos = np.bincount(etiquetas_k, minlength=n_clases)
    return int(np.argmax(conteos))

def voto_por_distancia(etiquetas_k: np.ndarray,
                       distancias_k: np.ndarray,
                       n_clases: int) -> int:
    EPSILON = 1e-10
    pesos = 1.0 / (distancias_k + EPSILON)

    pesos_por_clase = np.zeros(n_clases, dtype=float)
    for etiqueta, peso in zip(etiquetas_k, pesos):
        pesos_por_clase[etiqueta] += peso

    return int(np.argmax(pesos_por_clase))
