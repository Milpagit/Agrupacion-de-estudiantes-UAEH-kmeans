# Op1_unificado.py (v2 - con imputación)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer  # ✅ 1. Importar el Imputador
from sklearn.pipeline import Pipeline      # ✅ 2. Importar el Pipeline

# --- PASO 1: CARGA DE DATOS CRUDOS ---
try:
    df = pd.read_csv('Datos.csv', encoding='latin1')
    print("✅ Archivo 'Datos.csv' cargado correctamente.")
except FileNotFoundError:
    print("⚠️ Archivo 'Datos.csv' no encontrado.")
    exit()

# --- PASO 2: DEFINIR VARIABLES PREDICTORAS Y OBJETIVO ---
umbral_aprobacion = 7
df['aprobo'] = (df['exam_score'] > umbral_aprobacion).astype(int)

y = df['aprobo']
predictor_cols = [
    'study_hours_per_day', 'social_media_hours', 'netflix_hours',
    'part_time_job', 'attendance_percentage', 'sleep_hours',
    'diet_quality', 'exercise_frequency', 'parental_education_level',
    'mental_health_rating', 'extracurricular_participation',
    'academic_load', 'study_method', 'motivation_level',
    'time_management_tools', 'stress_level'
]
X = df[predictor_cols]

# --- PASO 3: DEFINIR PREPROCESADOR UNIFICADO (CON IMPUTACIÓN) ---
# Identificar columnas numéricas y categóricas
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# ✅ 3. Crear "líneas de ensamblaje" (Pipelines) para cada tipo de dato
# Para datos numéricos: 1º Rellenar huecos con la mediana, 2º Escalar
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Para datos categóricos: 1º Rellenar huecos con el valor más frecuente, 2º Codificar
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Crear el ColumnTransformer final usando los pipelines
preprocessor_regression = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)

# --- PASO 4: DIVISIÓN DE DATOS ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PASO 5: APLICAR PREPROCESAMIENTO ---
X_train_processed = preprocessor_regression.fit_transform(X_train)
X_test_processed = preprocessor_regression.transform(X_test)

print("✅ Datos preprocesados con imputación y escalado.")

# --- PASO 6: ENTRENAR MODELO ---
modelo_regresion = LogisticRegression(max_iter=2000, class_weight='balanced')
modelo_regresion.fit(X_train_processed, y_train)
print("✅ Modelo entrenado correctamente.")

# --- PASO 7: EVALUACIÓN ---
y_pred = modelo_regresion.predict(X_test_processed)
print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# --- PASO 8: GUARDAR LOS DOS ARCHIVOS CLAVE ---
joblib.dump(modelo_regresion, 'modelo_regresion_aprobacion.pkl')
joblib.dump(preprocessor_regression, 'preprocessor_regresion.pkl')

print("\n💾 ¡Éxito! Se guardaron dos archivos:")
print("1. modelo_regresion_aprobacion.pkl (El modelo)")
print("2. preprocessor_regresion.pkl (La receta completa de preprocesamiento)")