"""
Módulo de Visualización en Terminal (ASCII Art).

Permite renderizar dígitos de 28x28 píxeles directamente en la consola
sin necesidad de interfaz gráfica. Incluye corrección de orientación (transpuesta)
específica para el formato de almacenamiento de EMNIST.
"""

import numpy as np


def mostrar_digito_ascii(pixeles: np.ndarray,
                         etiqueta: int = -1,
                         prediccion: int = -1) -> None:
    """
    Renderiza un vector de 784 píxeles en una matriz ASCII de 28x28 en consola.

    Parámetros:
        pixeles (np.ndarray): Vector de características (784 píxeles normalizados o brutos).
        etiqueta (int): Clase real (opcional).
        prediccion (int): Clase predicha por el clasificador (opcional).
    """
    # EMNIST almacena las imágenes transpuestas respecto a la orientación estándar
    imagen = pixeles.reshape(28, 28).T
    escala = " .:-=+*#%@"

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
            idx = max(0, min(idx, len(escala) - 1))
            linea += escala[idx]
        linea += "|"
        print(linea)
    print("  +" + "-" * 28 + "+\n")


def mostrar_muestras_con_predicciones(X_test: np.ndarray,
                                      y_test: np.ndarray,
                                      y_predichas: np.ndarray,
                                      n_muestras: int = 10) -> None:
    """
    Despliega un conjunto de muestras evaluadas junto a su predicción y etiqueta real.

    Parámetros:
        X_test (np.ndarray): Píxeles de prueba.
        y_test (np.ndarray): Clases verdaderas.
        y_predichas (np.ndarray): Clases estimadas.
        n_muestras (int): Cantidad de ejemplos a desplegar.
    """
    print("\n" + "=" * 55)
    print(f"  VISUALIZACIÓN DE {n_muestras} MUESTRAS DE TESTEO")
    print("=" * 55)

    for i in range(min(n_muestras, len(X_test))):
        print(f"\n  --- Muestra #{i + 1} ---")
        mostrar_digito_ascii(X_test[i], y_test[i], y_predichas[i])


def mostrar_errores(X_test: np.ndarray,
                    y_test: np.ndarray,
                    y_predichas: np.ndarray,
                    n_errores: int = 5) -> None:
    """
    Localiza y muestra los casos donde la predicción difiere de la clase real.

    Parámetros:
        X_test (np.ndarray): Píxeles de prueba.
        y_test (np.ndarray): Clases verdaderas.
        y_predichas (np.ndarray): Clases estimadas.
        n_errores (int): Cantidad máxima de fallos a desplegar.
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
