"""
Módulo: clasificador/distancias.py
Responsabilidad:
    - Calcular la distancia entre un punto de testeo y TODOS los
      puntos de entrenamiento usando scipy (o numpy como alternativa).
    - Soportar métricas: euclidiana, manhattan y coseno.
    - Retornar un vector 1D de distancias.

Nota académica:
    NO se usa sklearn.neighbors.KNeighborsClassifier ni fitKNN.
    Las distancias se calculan con scipy.spatial.distance.cdist,
    que es solo una herramienta de cálculo matemático (como usar
    numpy para multiplicaciones matriciales), NO un clasificador.
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.spatial.distance import cdist


# ──────────────────────────────────────────────────────────────
def calcular_distancias(X_train: np.ndarray,
                        muestra: np.ndarray,
                        metrica: str = 'euclidean') -> np.ndarray:
    """
    Calcula la distancia entre UNA muestra de testeo y TODAS las
    muestras de entrenamiento.

    Parámetros
    ----------
    X_train : ndarray (n_train, n_features)
        Conjunto de entrenamiento almacenado en el modelo.
    muestra : ndarray (n_features,) o (1, n_features)
        Punto de consulta para el cual calcular distancias.
    metrica : str
        Nombre de la métrica de distancia:
          'euclidean'  -> sqrt( sum( (a-b)^2 ) )
          'manhattan'  -> sum( |a-b| )
          'cosine'     -> 1 - cosine_similarity(a, b)

    Retorna
    -------
    distancias : ndarray (n_train,)
        Vector con la distancia de la muestra a cada punto de entrenamiento.
    """
    muestra_2d = muestra.reshape(1, -1)              # (1, n_features)
    distancias = cdist(muestra_2d, X_train, metric=metrica)
    return distancias.flatten()                      # (n_train,)


# ──────────────────────────────────────────────────────────────
def calcular_distancias_batch(X_train: np.ndarray,
                               X_query: np.ndarray,
                               metrica: str = 'euclidean') -> np.ndarray:
    """
    Calcula distancias entre un LOTE de muestras de testeo y todos
    los puntos de entrenamiento de forma vectorizada.

    Parámetros
    ----------
    X_train : ndarray (n_train, n_features)
    X_query : ndarray (n_query, n_features)
    metrica : str - métrica de distancia

    Retorna
    -------
    distancias : ndarray (n_query, n_train)
        Matriz de distancias: distancias[i, j] = dist(X_query[i], X_train[j])
    """
    return cdist(X_query, X_train, metric=metrica)
