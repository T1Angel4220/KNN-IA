"""
Módulo: evaluacion/matriz_confusion.py
Responsabilidad:
    - Construir la matriz de confusión manualmente con NumPy.
    - Imprimir la matriz de forma legible en consola.
    - Mostrar precisión por clase y top-5 errores más frecuentes.
"""

# pyrefly: ignore [missing-import]
import numpy as np


# ──────────────────────────────────────────────────────────────
def construir_matriz(y_verdaderas: np.ndarray,
                     y_predichas: np.ndarray,
                     n_clases: int = 10) -> np.ndarray:
    """
    Construye la matriz de confusión usando únicamente NumPy.

    Parámetros
    ----------
    y_verdaderas : ndarray (n,) - etiquetas reales
    y_predichas  : ndarray (n,) - etiquetas predichas
    n_clases     : int          - número de clases (10 dígitos)

    Retorna
    -------
    matriz : ndarray (n_clases, n_clases)
        matriz[i, j] = número de veces que la clase real 'i'
                       fue predicha como clase 'j'
    """
    matriz = np.zeros((n_clases, n_clases), dtype=int)
    for real, pred in zip(y_verdaderas, y_predichas):
        matriz[real][pred] += 1
    return matriz


# ──────────────────────────────────────────────────────────────
def imprimir_matriz(matriz: np.ndarray, n_clases: int = 10) -> None:
    """
    Imprime la matriz de confusión con formato tabular en consola.

    Parámetros
    ----------
    matriz   : ndarray (n_clases, n_clases)
    n_clases : int
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

    # Top-5 confusiones más frecuentes
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
