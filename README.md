![image](./0_data/usopen.png)

# Predicción del Ganador del US Open 2025

## Resumen del Proyecto
Se implementa un modelo de Machine Learning (ML) para simular y predecir el resultado del último Grand Slam del año. Utilizamos una metodología de Series Temporales y el Método Monte Carlo para calcular la probabilidad de campeonato de cada jugador.  

El principal valor reside en la Ingeniería de Características retrospectivas y la Validación Cronológica para construir un modelo robusto y libre de data leakage.  

## Metodología Clave
Modelo de Habilidad Dinámica (Elo): se utiliza un rating Elo dinámico y específico por superficie (Elo_Surface_Diff) para medir la habilidad de un jugador justo antes de cada partido.  
Validación Temporal: el entrenamiento se realizó en datos pasados y se probó en datos futuros (Train < 2016, Test > 2016) para asegurar que el modelo generalice.  
Rendimiento Científico: el modelo Random Forest final logró un ROC AUC de 0.72 en el set de prueba, demostrando una alta capacidad de discriminación del riesgo.  

## 📂 Estructura del proyecto
El proyecto sigue una arquitectura modular para garantizar la separación de responsabilidades y la reproducibilidad.

```
/PREDICCION_USOPEN_2025
├── 0_data/                 # Almacenamiento de datos y sets de entrenamiento
│   ├── 0_raw/              # Datos originales (ej: atp_tennis.csv)
│   ├── 1_processed/        # Datos con Feature Engineering (partidos_final.csv)
│   ├── 2_train/            # X_train y y_train (datos <= 2016)
│   └── 3_test/             # X_test y y_test (datos >= 2016)
│
├── 1_notebooks/                                         # Desarrollo y experimentación
│   ├── 01_Fuentes.ipynb                                 # Adquisición y unión de datos.
│   ├── 02_LimpiezaEDA.ipynb                             # Limpieza, EDA y Feature Engineering (Incluye Títulos Dinámicos).
│   ├── 03_Entrenamiento_Evaluacion.ipynb                # Tuning (Grid Search) y evaluación de modelos.
│   ├── 04_Logica_determinista_montecarlo1.ipynb         # Declaración de fx y lógicas para dos simulaciones
│   └── 04_Logica_montecarlo500.ipynb                    # Declaración de fx y lógicas para simulación final
│
├── 3_models/                               # Modelos serializados
│   ├── random_forest_modeloN.joblib        # Modelos intermedios del tuning.
│   ├── random_forest_modelofinalOK.joblib  # Pipeline Random Forest final (Despliegue).
│   └── model_config.yaml                   # Configuración de hiperparámetros.
│
├── 4_app_streamlit/        # Despliegue de la Aplicación Web
│   ├── app.py              # Código de la aplicación Streamlit.
│   ├── main.py             # Lógica de carga y estructura de datos.
│   ├── utils.py            # Funciones auxiliares (H2H Caching, simulación).
│   └── requirements.txt    # Dependencias del proyecto.
│
├── 5_docs/
│    ├── PRESENTACION JUEVES.pdf    
│    ├── PRESENTACION VIERNES.pdf
│    ├── simulacion_determinista.mp4
│    └── simulacion_montecarlo.mp4 
│
└─── README.md

```

---


## Tecnologías y Librerías
- **Python 3.11**
- **Pandas** 
- **NumPy** 
- **Matplotlib / Seaborn**
- **scikit-learn: Modelado y tuning (Random Forest).**
- **joblib: Serialización del modelo Pipeline.**

---

## Instrucciones de Despliegue

La aplicación ha sido desplegada exitosamente en Streamlit Community Cloud, asegurando la accesibilidad y el rendimiento de la arquitectura de caching optimizada.

Acceso Directo a la Aplicación Web  
Puedes acceder a las dos versiones del modelo a través de los siguientes enlaces:

Simulación Determinista (Camino Más Probable): https://prediccionusopen2025determinista.streamlit.app/

Simulación Monte Carlo - 1 run (Análisis de Riesgo): https://prediccionusopen2025montecarlo.streamlit.app/

La simulación Monte Carlo de 500 runs no tiene app por cumplimiento de tiempos estipulados para la presentación en clase.

El despliegue utiliza @st.cache_resource para asegurar que la carga pesada del DataFrame y la pre-computación de los datos (H2H Caching) se ejecuten solo una vez en el servidor. Esto es clave para garantizar la alta velocidad de la interfaz.


Para ejecutar la aplicación interactiva y simular el torneo (usando la arquitectura de caching optimizada):  
- Clonar el respositorio
- Instalar Dependencias:  

```python
pip install -r 4_app_streamlit/requirements.txt

```

- Ejecutar la Aplicación Streamlit:  

```python
streamlit run 4_app_streamlit/app.py

```
---

## 📚 Autor
Proyecto realizado por **Ignacio Díaz**
