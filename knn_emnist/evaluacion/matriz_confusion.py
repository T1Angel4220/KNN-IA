"""
Módulo de Construcción e Interpretación de Matriz de Confusión.

Responsabilidad:
  - Construir matriz de confusión bidimensional pura con NumPy.
  - Imprimir en consola la tabla cruzada de aciertos y errores por clase.
  - Identificar los pares de clases más frecuentemente confundidos.
"""

import numpy as np


def construir_matriz(y_verdaderas: np.ndarray,
                     y_predichas: np.ndarray,
                     n_clases: int = 10) -> np.ndarray:
    """
    Construye la matriz de confusión cuadrática (n_clases x n_clases).
    Filas corresponden a la clase real y columnas a la clase predicha.

    Parámetros:
        y_verdaderas (np.ndarray): Etiquetas reales.
        y_predichas (np.ndarray): Etiquetas predichas.
        n_clases (int): Número de categorías.

    Retorna:
        np.ndarray: Matriz de enteros de forma (n_clases, n_clases).
    """
    matriz = np.zeros((n_clases, n_clases), dtype=int)
    for real, pred in zip(y_verdaderas, y_predichas):
        matriz[real][pred] += 1
    return matriz


def imprimir_matriz(matriz: np.ndarray, n_clases: int = 10) -> None:
    """
    Despliega la matriz de confusión formateada en consola con porcentajes
    de precisión individuales y top de confusiones.

    Parámetros:
        matriz (np.ndarray): Matriz de confusión calculada.
        n_clases (int): Número total de clases.
    """
    print("\n" + "=" * 65)
    print("   MATRIZ DE CONFUSION (calculada con NumPy)")
    print("=" * 65)
    print("   Filas = Clase REAL  |  Columnas = Clase PREDICHA\n")

    encabezado = "Real\\Pred |" + "".join(f"  {j:3d}" for j in range(n_clases))
    print(encabezado)
    print("-" * len(encabezado))

    for i in range(n_clases):
        fila = f"  Dígito {i} |"
        for j in range(n_clases):
            if i == j:
                fila += f" [{matriz[i][j]:3d}]"
            else:
                fila += f"  {matriz[i][j]:3d} "

        total_real    = np.sum(matriz[i, :])
        correctos     = matriz[i][i]
        precision_cls = (correctos / total_real * 100) if total_real > 0 else 0.0
        fila += f"  | {precision_cls:.1f}%"
        print(fila)

    print("-" * len(encabezado))

    exactitud_global = np.trace(matriz) / np.sum(matriz)
    print(f"\n  Exactitud global  : {exactitud_global * 100:.2f}%")
    print(f"  Total correctos   : {np.trace(matriz)}")
    print(f"  Total errores     : {np.sum(matriz) - np.trace(matriz)}")

    print("\n  Top 5 confusiones más frecuentes:")
    errores = []
    for i in range(n_clases):
        for j in range(n_clases):
            if i != j and matriz[i][j] > 0:
                errores.append((matriz[i][j], i, j))
    errores.sort(reverse=True)
    for cantidad, real, pred in errores[:5]:
        print(f"    Real={real} predicho como {pred}: {cantidad} veces")

    print("=" * 65)
