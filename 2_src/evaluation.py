# Análisis numérico y datos
import numpy as np
import pandas as pd
from scipy import stats
import scipy.optimize as opt

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.style as style

# Machine Learning - Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline

import warnings

# Configuraciones
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

import joblib


# Definir el nombre de tu archivo (asegúrate de que la ruta sea correcta)
nombre_archivo = '3_models/random_forest_modelofinalOK.joblib' 

# Cargar el modelo en la variable 'modelok'
try:
    modelofinalOK = joblib.load(nombre_archivo)
    
    # El modelo cargado es el Pipeline completo (Scaler + Random Forest)
    print(f"✅ Modelo '{nombre_archivo}' cargado exitosamente.")
    print(f"Tipo de objeto cargado: {type(modelofinalOK)}")
    
except FileNotFoundError:
    print(f"❌ Error: Archivo no encontrado en la ruta: {nombre_archivo}")
    print("Por favor, verifica la ruta o asegúrate de que el archivo esté en el mismo directorio.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")

# Ahora puedes usar 'modelok' para hacer predicciones (predict_proba)
# Ejemplo: modelok.predict_proba(X_new)


##################################################################################
X_train_SIMPLIFIED = pd.read_csv('0_data/2_train/X_train_SIMPLIFIED.csv')
print("💾Has leído 'X_train_SIMPLIFIED.csv' correctamente💾")

y_train = pd.read_csv('0_data/2_train/y_train.csv')
print("💾Has leído 'y_train.csv' correctamente💾")

X_test_SIMPLIFIED = pd.read_csv('0_data/3_test/X_test_SIMPLIFIED.csv')
print("💾Has leído 'X_test_SIMPLIFIED.csv' correctamente💾")

y_test = pd.read_csv('0_data/3_test/y_test.csv')
print("💾Has leído 'y_test.csv' correctamente💾")
##################################################################################

y_pred = modelofinalOK.predict(X_test_SIMPLIFIED)
accuracy = accuracy_score(y_test, y_pred)

Y_pred_train = modelofinalOK.predict(X_train_SIMPLIFIED)
accuracy_train = accuracy_score(y_train, Y_pred_train)

print("\n--- RESUMEN FINAL ---")
print(f"Features predictivas (X) usadas: {len(X_train_SIMPLIFIED.columns)}")
print(f"Tamaño del set de entrenamiento: {len(X_train_SIMPLIFIED)} partidos")
print(f"Precisión en el set de ENTRENAMIENTO (Train): {accuracy_train:.4f}")
print(f"Precisión del Random Forest (test): {accuracy:.4f}")
# 1. Obtener las predicciones discretas y las probabilidades
y_proba = modelofinalOK.predict_proba(X_test_SIMPLIFIED)[:, 1] # Probabilidad de que P1 gane (Clase 1)

# --- MATRIZ DE CONFUSIÓN ---
print("\n--- MATRIZ DE CONFUSIÓN (TEST) ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# --- CLASSIFICATION REPORT (Precision, Recall, F1) ---
print("\n--- CLASSIFICATION REPORT (TEST) ---")
# La clase '1' es P1 Gana, la clase '0' es P2 Gana (P1 Pierde)
print(classification_report(y_test, y_pred))

# --- ROC AUC SCORE ---
# Mide la habilidad del modelo para distinguir entre clases
roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC AUC Score (TEST): {roc_auc:.4f}")
# 1. Obtener el Random Forest Estimator desde el Pipeline
# La variable 'modelok' es un Pipeline, el estimador RF es el segundo paso (índice 1).
rf_estimator = modelofinalOK.steps[1][1]

# 2. Obtener la importancia de las features
feature_importances = pd.Series(
    rf_estimator.feature_importances_, 
    index=X_train_SIMPLIFIED.columns # Usamos las columnas de X_train para las etiquetas
)

# 3. Visualizar las Top 10 Features
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).sort_values(ascending=True).plot(
    kind='barh', 
    color='skyblue'
)
plt.title('Top 10 Importancia de Features (Random Forest)', fontsize=14)
plt.xlabel('Importancia de Gini', fontsize=12)
plt.tight_layout()
plt.show()

print(feature_importances.nlargest(10))