import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# --- 1. Cargar el Conjunto de Datos ---
file_path = 'Datos.csv' # Asegúrate que este sea el nombre de tu archivo

# Verificación para ayudarte a depurar
print("--- Verificación de Archivo ---")
print(f"Buscando el archivo: {os.path.abspath(file_path)}")
if os.path.exists(file_path):
    print("✅ ¡Archivo encontrado!")
else:
    print(f"❌ ¡ERROR! No se encontró el archivo '{file_path}'.")
    exit()
print("-----------------------------")

try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"✅ Archivo '{file_path}' cargado exitosamente.")
except Exception as e:
    print(f"⚠️ Ocurrió un error al leer el archivo: {e}")
    exit()

# --- 2. Adaptar Nombres de Columnas y Tipos ---
# Verificamos que la columna 'exam_score' exista
if 'exam_score' not in df.columns:
    print("❌ ¡ERROR! La columna 'exam_score' no se encontró en el archivo.")
    exit()

# Identificar columnas numéricas, categóricas y binarias
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

binary_cols = []
for col in df.columns:
    if df[col].nunique() == 2:
        binary_cols.append(col)

# Las columnas a escalar son las numéricas que no son binarias ni la calificación
numeric_cols_to_scale = [col for col in numeric_cols if col not in binary_cols and col != 'exam_score']

print("\n--- Columnas Identificadas ---")
print(f"Numéricas a escalar: {numeric_cols_to_scale}")
print(f"Binarias o Categóricas (se codificarán): {list(set(categorical_cols + binary_cols))}")


# --- 3. Manejo de Valores Faltantes (Imputación) ---
print(f"\nValores faltantes antes de la imputación:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
for col in numeric_cols_to_scale:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in list(set(categorical_cols + binary_cols)):
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\n✅ Valores faltantes gestionados.")

# --- 4. Codificación de Variables Categóricas (One-Hot Encoding) ---
cols_to_encode = list(set(categorical_cols + binary_cols) - set(numeric_cols))
df_procesado = pd.get_dummies(df, columns=cols_to_encode, drop_first=True, dtype=int)
print("\n✅ Variables categóricas y binarias codificadas.")

# --- 5. Escalado de Variables Numéricas (Estandarización) ---
scaler = StandardScaler()
cols_to_scale_final = [c for c in numeric_cols_to_scale if c in df_procesado.columns]
df_procesado[cols_to_scale_final] = scaler.fit_transform(df_procesado[cols_to_scale_final])
print("\n✅ Variables numéricas escaladas.")

# --- 6. Guardar el Resultado Final ---
output_file = 'Datos_UAEH_preprocesados.csv'
df_procesado.to_csv(output_file, index=False)
print(f"\n💾 DataFrame preprocesado guardado como '{output_file}'")

print("\n--- Vista previa del DataFrame final preprocesado ---")
print(df_procesado.head())