# pyrefly: ignore [missing-import]
import numpy as np

from datos.cargar_datos import cargar_datos, normalizar
from clasificador.knn import ClasificadorKNN
from evaluacion.metricas         import calcular_metricas, imprimir_resultados
from evaluacion.matriz_confusion import construir_matriz, imprimir_matriz
from visualizacion.ver_ejemplos       import (mostrar_muestras_con_predicciones,
                                               mostrar_errores)
from visualizacion.graficar_resultados import (graficar_exactitud_error,
                                                graficar_precision_por_clase,
                                                graficar_matriz_confusion)
from config import K_VECINOS, METRICA, N_CLASES

if __name__ == '__main__':
    print("=" * 55)
    print("  CLASIFICADOR KNN - EMNIST DIGITS")
    print("  Implementación manual (sin fitKNN ni sklearn.KNN)")
    print("=" * 55)

    X_train, y_train, X_test, y_test, _ = cargar_datos()
    X_train, X_test = normalizar(X_train, X_test)

    knn = ClasificadorKNN()
    knn.ajustar(X_train, y_train)
    print(f"\n  Clasificador: {knn}")

    print("\n" + "=" * 55)
    print("  PREDICCIÓN SOBRE DATOS DE TESTEO")
    print("=" * 55)
    y_predichas = knn.predecir(X_test)

    exactitud, error = calcular_metricas(y_test, y_predichas)
    imprimir_resultados(y_test, y_predichas, exactitud, error)

    matriz = construir_matriz(y_test, y_predichas, n_clases=N_CLASES)
    imprimir_matriz(matriz, n_clases=N_CLASES)

    mostrar_muestras_con_predicciones(X_test, y_test, y_predichas, n_muestras=5)
    mostrar_errores(X_test, y_test, y_predichas, n_errores=3)

    print("\n" + "=" * 55)
    print("  GENERANDO GRÁFICOS PNG...")
    print("=" * 55)
    graficar_exactitud_error(exactitud, error, k=K_VECINOS, metrica=METRICA)
    graficar_precision_por_clase(y_test, y_predichas, n_clases=N_CLASES)
    graficar_matriz_confusion(matriz, n_clases=N_CLASES)

    print("\n" + "=" * 55)
    print("  CLASIFICACIÓN KNN FINALIZADA")
    print("=" * 55)
