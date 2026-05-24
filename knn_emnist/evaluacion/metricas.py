# pyrefly: ignore [missing-import]
import numpy as np

def calcular_metricas(y_verdaderas: np.ndarray,
                      y_predichas: np.ndarray) -> tuple:
    correctas = np.sum(y_predichas == y_verdaderas)
    total     = len(y_verdaderas)
    exactitud = correctas / total
    error     = 1.0 - exactitud
    return exactitud, error

def imprimir_resultados(y_verdaderas: np.ndarray,
                        y_predichas: np.ndarray,
                        exactitud: float,
                        error: float) -> None:
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
