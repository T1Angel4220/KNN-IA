"""
==============================================================================
  Clasificador KNN Manual para Reconocimiento de Dígitos (EMNIST Digits)
==============================================================================
Pipeline de Ejecución:
  1. Carga y normalización del conjunto EMNIST Digits (CSV).
  2. Ajuste del modelo KNN (Lazy Learning: memorización en memoria).
  3. Inferencia por lotes (batch processing) sobre conjunto de prueba.
  4. Evaluación cuantitativa (Exactitud y Tasa de Error global).
  5. Cálculo y visualización en consola de Matriz de Confusión.
  6. Muestreo cualitativo en consola mediante arte ASCII (con transposición).
  7. Exportación de visualizaciones gráficas a la carpeta 'resultados/'.
"""

import sys
from pathlib import Path
import numpy as np

# Asegura que los submódulos sean accesibles sin importar el directorio de llamada
DIRECTORIO_ACTUAL = Path(__file__).resolve().parent
if str(DIRECTORIO_ACTUAL) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_ACTUAL))

# Carga y preprocesamiento
from datos.cargar_datos import cargar_datos, normalizar

# Clasificador KNN implementado desde cero
from clasificador.knn import ClasificadorKNN

# Evaluación y métricas
from evaluacion.metricas import calcular_metricas, imprimir_resultados
from evaluacion.matriz_confusion import construir_matriz, imprimir_matriz

# Visualización (Terminal y Matplotlib)
from visualizacion.ver_ejemplos import (mostrar_muestras_con_predicciones,
                                        mostrar_errores)
from visualizacion.graficar_resultados import (graficar_exactitud_error,
                                               graficar_precision_por_clase,
                                               graficar_matriz_confusion)

# Configuración global
from config import K_VECINOS, METRICA, N_CLASES


def main() -> None:
    """Función principal orquestadora del experimento."""
    print("=" * 60)
    print("   CLASIFICADOR KNN MANUAL - EMNIST DIGITS")
    print("   Implementación Scratch (NumPy / SciPy)")
    print("=" * 60)

    # 1. Carga y Normalización de datos
    X_train, y_train, X_test, y_test, _ = cargar_datos()
    X_train, X_test = normalizar(X_train, X_test)

    # 2. Inicialización y Ajuste del Clasificador
    knn = ClasificadorKNN(k=K_VECINOS, metrica=METRICA, n_clases=N_CLASES)
    knn.ajustar(X_train, y_train)
    print(f"\n  Clasificador configurado: {knn}")

    # 3. Predicción vectorizada en lotes
    print("\n" + "=" * 60)
    print("  PREDICCIÓN SOBRE DATOS DE PRUEBA")
    print("=" * 60)
    y_predichas = knn.predecir(X_test)

    # 4. Cálculo de métricas globales
    exactitud, error = calcular_metricas(y_test, y_predichas)
    imprimir_resultados(y_test, y_predichas, exactitud, error)

    # 5. Matriz de confusión
    matriz = construir_matriz(y_test, y_predichas, n_clases=N_CLASES)
    imprimir_matriz(matriz, n_clases=N_CLASES)

    # 6. Inspección visual en consola (ASCII)
    mostrar_muestras_con_predicciones(X_test, y_test, y_predichas, n_muestras=5)
    mostrar_errores(X_test, y_test, y_predichas, n_errores=3)

    # 7. Generación de gráficos estáticos
    print("\n" + "=" * 60)
    print("  GENERANDO GRÁFICOS PNG EN 'resultados/'...")
    print("=" * 60)
    graficar_exactitud_error(exactitud, error, k=K_VECINOS, metrica=METRICA)
    graficar_precision_por_clase(y_test, y_predichas, n_clases=N_CLASES)
    graficar_matriz_confusion(matriz, n_clases=N_CLASES)

    print("\n" + "=" * 60)
    print("  EJECUCIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)


if __name__ == '__main__':
    main()
