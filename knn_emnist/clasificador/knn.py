# pyrefly: ignore [missing-import]
import numpy as np
from config import K_VECINOS, METRICA, PESO_VECINOS, N_CLASES, BATCH_PREDICCION
from clasificador.distancias import calcular_distancias_batch
from clasificador.votacion   import (obtener_k_vecinos,
                                     voto_uniforme,
                                     voto_por_distancia)

class ClasificadorKNN:
    def __init__(self,
                 k: int = K_VECINOS,
                 metrica: str = METRICA,
                 peso: str = PESO_VECINOS,
                 n_clases: int = N_CLASES):
        self.k        = k
        self.metrica  = metrica
        self.peso     = peso
        self.n_clases = n_clases
        self.X_train_ = None
        self.y_train_ = None

    def ajustar(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        print("\n" + "=" * 55)
        print("  AJUSTE DEL CLASIFICADOR KNN (Lazy Learning)")
        print("=" * 55)

        self.X_train_ = X_train.copy()
        self.y_train_ = y_train.copy()

        print(f"\n  Parámetros del clasificador:")
        print(f"    k        (vecinos)  : {self.k}")
        print(f"    metrica             : {self.metrica}")
        print(f"    peso votacion       : {self.peso}")
        print(f"    n_clases            : {self.n_clases}")
        print(f"\n  Datos almacenados:")
        print(f"    X_train : {self.X_train_.shape}")
        print(f"    y_train : {self.y_train_.shape}")
        print("\n  [OK] Datos de entrenamiento memorizados correctamente.")

    def _predecir_batch(self, X_query: np.ndarray) -> np.ndarray:
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
        if self.X_train_ is None:
            raise RuntimeError("El clasificador no ha sido ajustado. "
                               "Llame a .ajustar(X_train, y_train) primero.")

        n_total      = len(X_test)
        y_predichas  = np.empty(n_total, dtype=int)
        n_batches    = int(np.ceil(n_total / batch_size))

        print(f"\n  Prediciendo {n_total} muestras en {n_batches} batches "
              f"(batch_size={batch_size})...")

        for idx_batch in range(n_batches):
            inicio = idx_batch * batch_size
            fin    = min(inicio + batch_size, n_total)
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
                f"peso='{self.peso}', estado='{estado}')")
