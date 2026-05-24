"""
========================================================
  main.py  -  Clasificador KNN Manual para EMNIST Digits
========================================================
Autor: Clasificador implementado sin usar fitKNN.
Flujo del programa:
  1. Cargar y normalizar datos EMNIST Digits (CSV)
  2. Ajustar el clasificador KNN (memorizar datos de entrenamiento)
  3. Predecir etiquetas del conjunto de testeo (por batches)
  4. Calcular exactitud, error y mostrar resultados
  5. Construir e imprimir la matriz de confusión
  6. Mostrar ejemplos de dígitos con sus predicciones (ASCII)
  7. Generar gráficos PNG con los resultados

Módulos:
  datos/          → cargar_datos.py
  clasificador/   → distancias.py, votacion.py, knn.py
  evaluacion/     → metricas.py, matriz_confusion.py
  visualizacion/  → ver_ejemplos.py, graficar_resultados.py
  config.py       → parámetros centrales (K, métrica, rutas, etc.)
"""

# pyrefly: ignore [missing-import]
import numpy as np

# -- Datos --
from datos.cargar_datos import cargar_datos, normalizar

# -- Clasificador KNN (implementación manual) --
from clasificador.knn import ClasificadorKNN

# -- Evaluación --
from evaluacion.metricas         import calcular_metricas, imprimir_resultados
from evaluacion.matriz_confusion import construir_matriz, imprimir_matriz

# -- Visualización --
from visualizacion.ver_ejemplos       import (mostrar_muestras_con_predicciones,
                                               mostrar_errores)
from visualizacion.graficar_resultados import (graficar_exactitud_error,
                                                graficar_precision_por_clase,
                                                graficar_matriz_confusion)
from config import K_VECINOS, METRICA, N_CLASES


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':

    print("=" * 55)
    print("  CLASIFICADOR KNN - EMNIST DIGITS")
    print("  Implementación manual (sin fitKNN ni sklearn.KNN)")
    print("=" * 55)

    # ── 1. CARGA Y NORMALIZACIÓN ─────────────────────────────
    X_train, y_train, X_test, y_test, _ = cargar_datos()
    X_train, X_test = normalizar(X_train, X_test)

    # ── 2. AJUSTE DEL CLASIFICADOR KNN ───────────────────────
    knn = ClasificadorKNN()
    knn.ajustar(X_train, y_train)
    print(f"\n  Clasificador: {knn}")

    # ── 3. PREDICCIÓN ────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PREDICCIÓN SOBRE DATOS DE TESTEO")
    print("=" * 55)
    y_predichas = knn.predecir(X_test)

    # ── 4. MÉTRICAS ──────────────────────────────────────────
    exactitud, error = calcular_metricas(y_test, y_predichas)
    imprimir_resultados(y_test, y_predichas, exactitud, error)

    # ── 5. MATRIZ DE CONFUSIÓN ───────────────────────────────
    matriz = construir_matriz(y_test, y_predichas, n_clases=N_CLASES)
    imprimir_matriz(matriz, n_clases=N_CLASES)

    # ── 6. VISUALIZACIÓN ASCII ───────────────────────────────
    mostrar_muestras_con_predicciones(X_test, y_test, y_predichas, n_muestras=5)
    mostrar_errores(X_test, y_test, y_predichas, n_errores=3)

    # ── 7. GRÁFICOS PNG ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("  GENERANDO GRÁFICOS PNG...")
    print("=" * 55)
    graficar_exactitud_error(exactitud, error, k=K_VECINOS, metrica=METRICA)
    graficar_precision_por_clase(y_test, y_predichas, n_clases=N_CLASES)
    graficar_matriz_confusion(matriz, n_clases=N_CLASES)

    print("\n" + "=" * 55)
    print("  CLASIFICACIÓN KNN FINALIZADA")
    print("=" * 55)
