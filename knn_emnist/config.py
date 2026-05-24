# ============================================================
#  CONFIGURACION CENTRAL - Clasificador KNN EMNIST
# ============================================================

# Rutas del dataset EMNIST Digits (CSV)
RUTA_TRAIN = r'C:\Users\ayuqu\Downloads\DB\emnist-digits-train.csv'
RUTA_TEST  = r'C:\Users\ayuqu\Downloads\DB\emnist-digits-test.csv'

# ---- Parametros del clasificador KNN ----
K_VECINOS        = 5          # Numero de vecinos a considerar
METRICA          = 'euclidean'  # Distancia: 'euclidean' | 'manhattan' | 'cosine'
PESO_VECINOS     = 'uniforme'   # 'uniforme' (voto igual) | 'distancia' (voto ponderado)

# ---- Subconjunto de datos (para acelerar la prueba) ----
# Coloca None para usar TODO el dataset (puede tardar bastante)
MAX_TRAIN        = 10_000  # Muestras de entrenamiento a usar
MAX_TEST         = 2_000   # Muestras de testeo a usar

# ---- Tamanio del batch para prediccion (evita agotar RAM) ----
BATCH_PREDICCION = 200

# ---- Digitos disponibles (0-9) ----
N_CLASES = 10
