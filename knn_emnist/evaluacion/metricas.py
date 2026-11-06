"""
Módulo de Métricas de Evaluación para Clasificación.

Calcula e imprime indicadores globales de rendimiento:
  - Exactitud (Accuracy): proporción de aciertos sobre el total.
  - Tasa de Error: proporción de clasificaciones incorrectas.
"""

from typing import Tuple
import numpy as np


def calcular_metricas(y_verdaderas: np.ndarray,
                      y_predichas: np.ndarray) -> Tuple[float, float]:
    """
    Calcula la exactitud (accuracy) y la tasa de error global.

    Parámetros:
        y_verdaderas (np.ndarray): Etiquetas reales del conjunto de evaluación.
        y_predichas (np.ndarray): Etiquetas obtenidas por el clasificador.

    Retorna:
        Tuple[float, float]: (exactitud, error) en rango [0.0, 1.0].
    """
    correctas = np.sum(y_predichas == y_verdaderas)
    total     = len(y_verdaderas)
    exactitud = float(correctas / total) if total > 0 else 0.0
    error     = 1.0 - exactitud
    return exactitud, error


def imprimir_resultados(y_verdaderas: np.ndarray,
                        y_predichas: np.ndarray,
                        exactitud: float,
                        error: float) -> None:
    """
    Muestra en consola un resumen formateado de los resultados de evaluación.

    Parámetros:
        y_verdaderas (np.ndarray): Clases reales.
        y_predichas (np.ndarray): Clases estimadas.
        exactitud (float): Métrica de exactitud calculada.
        error (float): Métrica de error calculada.
    """
    total     = len(y_verdaderas)
    correctas = int(exactitud * total)

    print("\n" + "=" * 55)
    print("  RESULTADO FINAL - CLASIFICADOR KNN")
    print("=" * 55)
    print(f"\n  Primeras 10 predicciones : {y_predichas[:10]}")
    print(f"  Primeras 10 reales       : {y_verdaderas[:10]}")
    print(f"\n  Total muestras testeo    : {total}")
    print(f"  Predicciones correctas   : {correctas}")
    print(f"  Exactitud                : {exactitud:.4f}  ({exactitud * 100:.2f}%)")
    print(f"  Error                    : {error:.4f}  ({error * 100:.2f}%)")

    print("\n  Evaluación del clasificador:")
    if error <= 0.10:
        print("  [OK] Clasificador ADECUADO  (error <= 10%, exactitud >= 90%)")
    elif error <= 0.20:
        print("  [~~] Clasificador ACEPTABLE (error <= 20%), considere ajustar K")
    else:
        print("  [--] Clasificador necesita mejoras (error > 20%)")

    print("=" * 55)
