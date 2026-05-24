# pyrefly: ignore [missing-import]
import numpy as np

def mostrar_digito_ascii(pixeles: np.ndarray,
                         etiqueta: int = -1,
                         prediccion: int = -1) -> None:
    imagen = pixeles.reshape(28, 28)
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
            linea += escala[idx]
        linea += "|"
        print(linea)
    print("  +" + "-" * 28 + "+\n")

def mostrar_muestras_con_predicciones(X_test: np.ndarray,
                                      y_test: np.ndarray,
                                      y_predichas: np.ndarray,
                                      n_muestras: int = 10) -> None:
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
