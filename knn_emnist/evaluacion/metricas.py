"""
Módulo: evaluacion/metricas.py
Responsabilidad:
    - Calcular exactitud y error del clasificador KNN.
    - Mostrar comparación entre predicciones y valores reales.
    - Evaluar si el modelo alcanza el umbral de calidad (error <= 0.10).
"""

# pyrefly: ignore [missing-import]
import numpy as np


# ──────────────────────────────────────────────────────────────
def calcular_metricas(y_verdaderas: np.ndarray,
                      y_predichas: np.ndarray) -> tuple:
    """
    Calcula exactitud global y error de clasificación.

    Parámetros
    ----------
    y_verdaderas : ndarray (n,) - etiquetas reales
    y_predichas  : ndarray (n,) - etiquetas predichas por el KNN

    Retorna
    -------
    exactitud : float  - fracción de predicciones correctas [0, 1]
    error     : float  - fracción de predicciones incorrectas [0, 1]
    """
    correctas = np.sum(y_predichas == y_verdaderas)
    total     = len(y_verdaderas)
    exactitud = correctas / total
    error     = 1.0 - exactitud
    return exactitud, error


# ──────────────────────────────────────────────────────────────
def imprimir_resultados(y_verdaderas: np.ndarray,
                        y_predichas: np.ndarray,
                        exactitud: float,
                        error: float) -> None:
    """
    Muestra los resultados finales de evaluación en consola.

    Parámetros
    ----------
    y_verdaderas, y_predichas : arrays de etiquetas
    exactitud, error          : métricas calculadas
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
