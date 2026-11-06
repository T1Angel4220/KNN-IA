"""
Módulo del Clasificador K-Nearest Neighbors (KNN).

Implementación manual del algoritmo KNN desde cero (Scratch Implementation):
  - Paradigma Lazy Learning (memorización de datos en etapa de ajuste).
  - Cálculo de distancias vectorizadas entre lote de testeo y memoria.
  - Votación uniforme o ponderada por distancia.
  - Inferencia segmentada en lotes (batch processing) para optimizar memoria RAM.
"""

from typing import Optional
import numpy as np
from config import K_VECINOS, METRICA, PESO_VECINOS, N_CLASES, BATCH_PREDICCION
from clasificador.distancias import calcular_distancias_batch
from clasificador.votacion import (obtener_k_vecinos,
                                  voto_uniforme,
                                  voto_por_distancia)


class ClasificadorKNN:
    """
    Clasificador K-Nearest Neighbors implementado con NumPy y SciPy.

    Atributos:
        k (int): Número de vecinos más cercanos a considerar.
        metrica (str): Métrica geométrica para calcular distancias ('euclidean', 'manhattan', 'cosine').
        peso (str): Estrategia de ponderación ('uniforme' | 'distancia').
        n_clases (int): Número de clases del problema de clasificación.
        X_train_ (np.ndarray): Puntos de características almacenados tras el ajuste.
        y_train_ (np.ndarray): Etiquetas almacenadas tras el ajuste.
    """

    def __init__(self,
                 k: int = K_VECINOS,
                 metrica: str = METRICA,
                 peso: str = PESO_VECINOS,
                 n_clases: int = N_CLASES) -> None:
        self.k = k
        self.metrica = metrica
        self.peso = peso
        self.n_clases = n_clases
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def ajustar(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fase de ajuste del clasificador (Lazy Learning).

        Almacena el conjunto de datos de entrenamiento en memoria para
        compararlo durante la fase de predicción.

        Parámetros:
            X_train (np.ndarray): Matriz de entrenamiento (n_muestras, n_features).
            y_train (np.ndarray): Vector de etiquetas reales (n_muestras,).
        """
        print("\n" + "=" * 55)
        print("  AJUSTE DEL CLASIFICADOR KNN (Lazy Learning)")
        print("=" * 55)
        print(f"  Hiperparámetros:")
        print(f"    - K (vecinos) : {self.k}")
        print(f"    - Métrica     : {self.metrica}")
        print(f"    - Ponderación : {self.peso}")
        print(f"    - Clases      : {self.n_clases}")

        self.X_train_ = np.asarray(X_train, dtype='float32')
        self.y_train_ = np.asarray(y_train, dtype=int)

        print(f"\n  Datos memorizados:")
        print(f"    X_train : {self.X_train_.shape}")
        print(f"    y_train : {self.y_train_.shape}")
        print("\n  [OK] Datos de entrenamiento memorizados correctamente.")

    def _predecir_batch(self, X_query: np.ndarray) -> np.ndarray:
        """
        Ejecuta la predicción para un subconjunto (lote) de muestras.

        1. Computa matriz de distancias (n_lote, n_train).
        2. Selecciona los K vecinos más cercanos por cada muestra.
        3. Aplica regla de votación seleccionada.

        Parámetros:
            X_query (np.ndarray): Lote de muestras (n_query, n_features).

        Retorna:
            np.ndarray: Vector de clases predichas para el lote.
        """
        matriz_dist = calcular_distancias_batch(self.X_train_, X_query, self.metrica)
        predicciones = np.empty(len(X_query), dtype=int)

        for i in range(len(X_query)):
            _, etiquetas_k, distancias_k = obtener_k_vecinos(
                matriz_dist[i], self.y_train_, self.k
            )

            if self.peso == 'distancia':
                predicciones[i] = voto_por_distancia(etiquetas_k, distancias_k, self.n_clases)
            else:
                predicciones[i] = voto_uniforme(etiquetas_k, self.n_clases)

        return predicciones

    def predecir(self, X_test: np.ndarray,
                 batch_size: int = BATCH_PREDICCION) -> np.ndarray:
        """
        Genera predicciones sobre todo el conjunto de prueba dividiéndolo en lotes.

        Parámetros:
            X_test (np.ndarray): Conjunto de prueba (n_test, n_features).
            batch_size (int): Cantidad de muestras a evaluar simultáneamente.

        Retorna:
            np.ndarray: Vector con las etiquetas predichas para todas las muestras.
        """
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("El clasificador no ha sido ajustado. Llame a .ajustar() primero.")

        n_total = len(X_test)
        y_predichas = np.empty(n_total, dtype=int)

        print(f"\n  Iniciando predicción sobre {n_total} muestras en lotes de {batch_size}...")

        for inicio in range(0, n_total, batch_size):
            fin = min(inicio + batch_size, n_total)
            X_lote = X_test[inicio:fin]

            y_predichas[inicio:fin] = self._predecir_batch(X_lote)

            porcentaje = fin / n_total * 100
            barra_len  = 30
            relleno    = int(barra_len * fin / n_total)
            barra      = "#" * relleno + "-" * (barra_len - relleno)
            print(f"\r  [{barra}] {porcentaje:5.1f}%  ({fin}/{n_total})", end="", flush=True)

        print()
        return y_predichas

    def __repr__(self) -> str:
        estado = "ajustado" if self.X_train_ is not None else "sin ajustar"
        return (f"ClasificadorKNN(k={self.k}, metrica='{self.metrica}', "
                f"peso='{self.peso}', estado={estado})")
