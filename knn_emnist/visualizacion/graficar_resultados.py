"""
Módulo: visualizacion/graficar_resultados.py
Responsabilidad:
    - Graficar con matplotlib la exactitud y error global del KNN.
    - Graficar la precisión por clase (barra horizontal).
    - Guardar los gráficos como PNG en la carpeta resultados/.
    - Visualizar la matriz de confusión como mapa de calor.
"""

# pyrefly: ignore [missing-import]
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [import-shadowed, missing-import]
import matplotlib
# pyrefly: ignore [import-shadowed]
matplotlib.use('Agg')               # backend sin pantalla
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt


CARPETA_RESULTADOS = os.path.join(os.path.dirname(__file__), '..', 'resultados')


# ──────────────────────────────────────────────────────────────
def _asegurar_carpeta() -> str:
    """Crea la carpeta resultados/ si no existe y retorna su ruta."""
    ruta = os.path.abspath(CARPETA_RESULTADOS)
    os.makedirs(ruta, exist_ok=True)
    return ruta


# ──────────────────────────────────────────────────────────────
def graficar_exactitud_error(exactitud: float,
                             error: float,
                             k: int,
                             metrica: str) -> None:
    """
    Genera un gráfico de barras con exactitud y error globales.

    Parámetros
    ----------
    exactitud, error : float - métricas globales [0, 1]
    k                : int   - K usado en el clasificador
    metrica          : str   - métrica de distancia usada
    """
    carpeta = _asegurar_carpeta()
    fig, ax = plt.subplots(figsize=(6, 4))

    categorias = ['Exactitud', 'Error']
    valores    = [exactitud, error]
    colores    = ['#2ecc71', '#e74c3c']

    barras = ax.bar(categorias, valores, color=colores, width=0.4, edgecolor='white')

    for barra, val in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.01,
                f'{val * 100:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Proporción', fontsize=11)
    ax.set_title(f'KNN  |  k={k}  |  métrica: {metrica}', fontsize=12, pad=12)
    ax.axhline(y=0.90, color='gray', linestyle='--', linewidth=1, label='Umbral 90%')
    ax.legend(fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(carpeta, 'exactitud_error.png')
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"\n  [PNG] Gráfico guardado en: {ruta}")


# ──────────────────────────────────────────────────────────────
def graficar_precision_por_clase(y_verdaderas: np.ndarray,
                                 y_predichas: np.ndarray,
                                 n_clases: int = 10) -> None:
    """
    Gráfico de barras horizontal con la precisión (%) por clase.

    Parámetros
    ----------
    y_verdaderas, y_predichas : arrays de etiquetas
    n_clases                  : int
    """
    carpeta = _asegurar_carpeta()

    precisiones = []
    for c in range(n_clases):
        mascara   = (y_verdaderas == c)
        total_c   = np.sum(mascara)
        correctos = np.sum((y_verdaderas == c) & (y_predichas == c))
        prec      = correctos / total_c * 100 if total_c > 0 else 0.0
        precisiones.append(prec)

    fig, ax = plt.subplots(figsize=(7, 5))
    colores = ['#3498db' if p >= 90 else '#e67e22' if p >= 75 else '#e74c3c'
               for p in precisiones]

    barras = ax.barh(range(n_clases), precisiones, color=colores, edgecolor='white')

    for barra, p in zip(barras, precisiones):
        ax.text(p + 0.5, barra.get_y() + barra.get_height() / 2,
                f'{p:.1f}%', va='center', fontsize=9)

    ax.set_yticks(range(n_clases))
    ax.set_yticklabels([f'Dígito {c}' for c in range(n_clases)])
    ax.set_xlim(0, 115)
    ax.set_xlabel('Precisión (%)', fontsize=11)
    ax.set_title('Precisión por clase - KNN', fontsize=12, pad=12)
    ax.axvline(x=90, color='gray', linestyle='--', linewidth=1, label='Umbral 90%')
    ax.legend(fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(carpeta, 'precision_por_clase.png')
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  [PNG] Gráfico guardado en: {ruta}")


# ──────────────────────────────────────────────────────────────
def graficar_matriz_confusion(matriz: np.ndarray,
                              n_clases: int = 10) -> None:
    """
    Visualiza la matriz de confusión como mapa de calor.

    Parámetros
    ----------
    matriz   : ndarray (n_clases, n_clases)
    n_clases : int
    """
    carpeta = _asegurar_carpeta()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matriz, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n_clases))
    ax.set_yticks(range(n_clases))
    ax.set_xticklabels(range(n_clases))
    ax.set_yticklabels(range(n_clases))
    ax.set_xlabel('Clase Predicha', fontsize=11)
    ax.set_ylabel('Clase Real', fontsize=11)
    ax.set_title('Matriz de Confusión - KNN EMNIST', fontsize=12, pad=12)

    umbral = matriz.max() / 2
    for i in range(n_clases):
        for j in range(n_clases):
            color = 'white' if matriz[i, j] > umbral else 'black'
            ax.text(j, i, str(matriz[i, j]),
                    ha='center', va='center', color=color, fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(carpeta, 'matriz_confusion.png')
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  [PNG] Gráfico guardado en: {ruta}")
