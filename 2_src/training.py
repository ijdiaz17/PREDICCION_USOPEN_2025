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


##################################################################################
df = pd.read_csv('0_data/1_processed/partidos_final.csv')
print("✅Has leído el csv 'partidos_final.csv' correctamente✅")
##################################################################################

df['Date'] = pd.to_datetime(df['Date'])

# MODELO FINAL OK
# 1. Definición de la lista de features a MANTENER (Simplificación)

FINAL_FEATURES = [
    'Elo_Diff',
    'Elo_Surface_Diff',
    'H2H_Advantage',
    'Recent_Form_10_Diff',
    'Age_Diff',
    'Tournament_Titles_Diff_Dynamic', #nueva
    'Grand_Titles_Diff_Dynamic', #nueva
    'Round_Ordinal',
    'Series_Ordinal',
    'Court_Indoor',
    'Best_of_5',
    'Surface_Clay',
    'Surface_Grass',
    'Surface_Hard',
    # Eliminamos Rank_Diff y Pts_Diff
]

# --- 5. DEFINICIÓN FINAL DE MATRIZ DE FEATURES Y ENTRENAMIENTO ---

# Matriz de Features (X)
X = df[FINAL_FEATURES]
y = df['Target']

# 2. Re-entrenar el modelo con los parámetros óptimos del Grid Search
# Necesitarás tus variables X_train_SIMPLIFIED y y_train.

# 3. Evaluar de nuevo las métricas en X_test_SIMPLIFIED.
# División CRONOLÓGICA (Entrenar en el pasado, probar en el futuro)
X_train_SIMPLIFIED = df[df['Date'].dt.year < 2016][FINAL_FEATURES]
y_train = df[df['Date'].dt.year < 2016]['Target']

X_test_SIMPLIFIED = df[df['Date'].dt.year >= 2016][FINAL_FEATURES]
y_test = df[df['Date'].dt.year >= 2016]['Target']


##################################################################################
X_train_SIMPLIFIED.to_csv('0_data/2_train/X_train_SIMPLIFIED.csv', index=False)
print("💾Has exportado 'X_train_SIMPLIFIED.csv' en '0_data/2_train/' correctamente💾")

y_train.to_csv('0_data/2_train/y_train.csv', index=False)
print("💾Has exportado 'y_train.csv' en '0_data/2_train' correctamente💾")

X_test_SIMPLIFIED.to_csv('0_data/3_test/X_test_SIMPLIFIED.csv', index=False)
print("💾Has exportado 'X_test_SIMPLIFIED.csv' en '0_data/3_test' correctamente💾")

y_test.to_csv('0_data/3_test/y_test.csv', index=False)
print("💾Has exportado 'y_test.csv' en '0_data/3_test' correctamente💾")
##################################################################################



# Crear el Pipeline (escalador + modelo RF)
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Grid de Hiperparámetros
param_grid = {
    'model__n_estimators': [100, 250, 400],
    'model__max_depth': [8, 12, 16, 20],
    'model__min_samples_split': [5, 10],
    'model__max_features': ['sqrt', 0.5, 0.7],
}

# -----------------------------------------------------------
# 3. Ejecución del Grid Search
# -----------------------------------------------------------

# Definir GridSearchCV
# Usamos 'roc_auc' como métrica para encontrar el mejor modelo.
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,  # 3-Fold Cross-Validation (suficiente para Time Series Split)
    scoring='roc_auc',
    verbose=2,
    n_jobs=-1
)

# El Grid Search debe ejecutarse SÓLO sobre el conjunto de entrenamiento.
# Asegúrate de usar las variables X_train y y_train de tu división temporal.
grid_search.fit(X_train_SIMPLIFIED, y_train)

# -----------------------------------------------------------
# 4. Resultados (Post-ejecución)
# -----------------------------------------------------------

print(f"Mejor ROC AUC: {grid_search.best_score_:.4f}")
print(f"Mejores Parámetros: {grid_search.best_params_}")
# Guardar el mejor modelo encontrado
modelok1 = grid_search.best_estimator_


import joblib

# Definir el nombre del archivo para guardar el modelo
modelofinalOK = '3_models/random_forest_modelofinalOK.joblib'
# Guardar el modelo
joblib.dump(modelok1, modelofinalOK)
print(f"Modelo guardado exitosamente como: {modelofinalOK}")
