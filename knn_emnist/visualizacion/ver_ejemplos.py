"""
Módulo: visualizacion/ver_ejemplos.py
Responsabilidad:
    - Mostrar ejemplos de dígitos del dataset como texto ASCII.
    - Visualizar con matplotlib algunas muestras con sus predicciones.
    - Mostrar vecinos más cercanos de un dígito de testeo dado.
"""

# pyrefly: ignore [missing-import]
import numpy as np


# ──────────────────────────────────────────────────────────────
def mostrar_digito_ascii(pixeles: np.ndarray,
                         etiqueta: int = -1,
                         prediccion: int = -1) -> None:
    """
    Imprime un dígito 28×28 como arte ASCII en la consola.

    Parámetros
    ----------
    pixeles    : ndarray (784,) o (28, 28) - valores [0, 1]
    etiqueta   : int - clase real (-1 si no se conoce)
    prediccion : int - clase predicha (-1 si no aplica)
    """
    imagen = pixeles.reshape(28, 28)
    escala = " .:-=+*#%@"    # 10 niveles de gris

    if etiqueta >= 0 and prediccion >= 0:
        status = "OK" if etiqueta == prediccion else "XX"
        print(f"  Real: {etiqueta} | Pred: {prediccion} [{status}]")
    elif etiqueta >= 0:
        print(f"  Dígito real: {etiqueta}")

    print("  +" + "-" * 28 + "+")
    for fila in imagen:
        linea = "  |"
        for px in fila:
            idx = int(px * (len(escala) - 1))
            linea += escala[idx]
        linea += "|"
        print(linea)
    print("  +" + "-" * 28 + "+\n")


# ──────────────────────────────────────────────────────────────
def mostrar_muestras_con_predicciones(X_test: np.ndarray,
                                      y_test: np.ndarray,
                                      y_predichas: np.ndarray,
                                      n_muestras: int = 10) -> None:
    """
    Muestra en consola ASCII las primeras N muestras de testeo
    con sus etiquetas reales y predicciones del KNN.

    Parámetros
    ----------
    X_test       : ndarray (n_test, 784)
    y_test       : ndarray (n_test,)
    y_predichas  : ndarray (n_test,)
    n_muestras   : int - cuántas muestras mostrar
    """
    print("\n" + "=" * 55)
    print(f"  VISUALIZACIÓN DE {n_muestras} MUESTRAS DE TESTEO")
    print("=" * 55)

    for i in range(min(n_muestras, len(X_test))):
        print(f"\n  --- Muestra #{i + 1} ---")
        mostrar_digito_ascii(X_test[i], y_test[i], y_predichas[i])


# ──────────────────────────────────────────────────────────────
def mostrar_errores(X_test: np.ndarray,
                    y_test: np.ndarray,
                    y_predichas: np.ndarray,
                    n_errores: int = 5) -> None:
    """
    Muestra en ASCII las primeras N predicciones INCORRECTAS.

    Parámetros
    ----------
    X_test, y_test, y_predichas : arrays de testeo y predicciones
    n_errores : int - cuántos errores mostrar
    """
    indices_error = np.where(y_predichas != y_test)[0]

    print("\n" + "=" * 55)
    print(f"  PRIMERAS {n_errores} PREDICCIONES INCORRECTAS")
    print("=" * 55)

    if len(indices_error) == 0:
        print("\n  ¡Sin errores! Clasificador perfecto.")
        return

    for idx in indices_error[:n_errores]:
        print(f"\n  --- Error en muestra #{idx} ---")
        mostrar_digito_ascii(X_test[idx], y_test[idx], y_predichas[idx])
