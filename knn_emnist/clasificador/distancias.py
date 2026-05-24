# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.spatial.distance import cdist

def calcular_distancias(X_train: np.ndarray,
                        muestra: np.ndarray,
                        metrica: str = 'euclidean') -> np.ndarray:
    muestra_2d = muestra.reshape(1, -1)
    distancias = cdist(muestra_2d, X_train, metric=metrica)
    return distancias.flatten()

def calcular_distancias_batch(X_train: np.ndarray,
                               X_query: np.ndarray,
                               metrica: str = 'euclidean') -> np.ndarray:
    return cdist(X_query, X_train, metric=metrica)
