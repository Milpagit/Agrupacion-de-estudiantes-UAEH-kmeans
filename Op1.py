# ============================================
# 📘 MODELO DE REGRESIÓN LOGÍSTICA - UAEH
# Predicción de aprobación de examen (umbral = 7)
# ============================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# --- PASO 1: CARGA DE DATOS ---
try:
    df = pd.read_csv('Datos_UAEH_preprocesados.csv')
    print("✅ Archivo 'Datos_UAEH_preprocesados.csv' cargado correctamente.")
except FileNotFoundError:
    print("⚠️ Archivo 'Datos_UAEH_preprocesados.csv' no encontrado.")
    exit()

# --- PASO 2: CREAR VARIABLE OBJETIVO ---
umbral_aprobacion = 7  # ✅ umbral corregido
df['aprobo'] = (df['exam_score'] > umbral_aprobacion).astype(int)

# Separar variables predictoras (X) y variable objetivo (y)
y = df['aprobo']
X = df.drop(columns=['exam_score', 'aprobo'])

# --- PASO 3: TRATAMIENTO DE VARIABLES ---
# Convertir categóricas a numéricas (si existen)
X = pd.get_dummies(X, drop_first=True)

# --- PASO 4: DIVISIÓN DE DATOS ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"📊 Tamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"📊 Tamaño del conjunto de prueba: {X_test.shape}\n")

# --- PASO 5: ESCALAR DATOS ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- PASO 6: DISTRIBUCIÓN DE CLASES ---
print("⚖️ Distribución de clases en el dataset completo:")
print(y.value_counts(normalize=True).map("{:.2%}".format), "\n")

# --- PASO 7: ENTRENAR MODELO ---
modelo_regresion = LogisticRegression(
    max_iter=2000,
    C=0.5,  # Regularización más fuerte
    class_weight='balanced'  # Rebalancea las clases automáticamente
)
modelo_regresion.fit(X_train, y_train)
print("✅ Modelo entrenado correctamente.\n")

# --- PASO 8: EVALUACIÓN ---
print("=" * 40)
print("🔍 EVALUACIÓN DEL MODELO")
print("=" * 40 + "\n")

# Predicciones
y_pred = modelo_regresion.predict(X_test)
y_proba = modelo_regresion.predict_proba(X_test)[:, 1]

# --- 8.1 EXACTITUD ---
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Exactitud: {accuracy:.2f}")
print(f"➡️ El modelo acierta en el {accuracy:.2%} de las predicciones.\n")

# --- 8.2 MATRIZ DE CONFUSIÓN ---
conf_matrix = confusion_matrix(y_test, y_pred)
print("📊 Matriz de Confusión:")
print(conf_matrix, "\n")

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Pred: No Aprobó', 'Pred: Sí Aprobó'],
            yticklabels=['Real: No Aprobó', 'Real: Sí Aprobó'])
plt.title("Matriz de Confusión - Regresión Logística")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()

# --- 8.3 REPORTE DE CLASIFICACIÓN ---
print("📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Aprobó', 'Sí Aprobó']))

# --- 8.4 AUC Y CURVA ROC ---
auc = roc_auc_score(y_test, y_proba)
print(f"🔥 AUC del modelo: {auc:.2f}\n")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC - Regresión Logística")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- 8.5 IMPORTANCIA DE VARIABLES ---
try:
    coeficientes = pd.Series(modelo_regresion.coef_[0], index=X.columns)
    coef_ordenado = coeficientes.sort_values(ascending=False)

    print("🔎 Variables que más influyen en aprobar:")
    print(coef_ordenado.head(10), "\n")

    plt.figure(figsize=(8, 5))
    coef_ordenado.head(10).plot(kind='bar', color='teal')
    plt.title("Top 10 Variables que más influyen en aprobar")
    plt.ylabel("Coeficiente (importancia)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("⚠️ No se pudieron graficar las variables más influyentes:", e)

# --- PASO 9: GUARDAR MODELO Y ESCALADOR ---
joblib.dump(modelo_regresion, 'modelo_regresion_aprobacion.pkl')
joblib.dump(scaler, 'scaler_regresion_aprobacion.pkl')
print("💾 Modelo y escalador guardados correctamente.\n")

print("✅ Proceso finalizado con éxito.")
