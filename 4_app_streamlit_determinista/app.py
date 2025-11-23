import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Importar funciones y datos estructurales ---
# Importamos la función de simulación determinista
from main import load_data_and_caches, cuadro_usopen2025 
from utils import simular_torneo_presentacion # <-- CAMBIO CLAVE

# --- 2. Caching de Recursos Pesados ---
@st.cache_resource
def get_resources():
    return load_data_and_caches()

# 3. Asignación Global (se ejecuta solo una vez gracias a @st.cache_resource)
modelofinalOK, diccionario_jugadores_simulacion, h2h_cache_final = get_resources() 

# ----------------------------------------------------
# 4. CONFIGURACIÓN INICIAL Y ESTILOS CSS
# ----------------------------------------------------

# Aplicar estilos CSS para centrado, fondo y detalles de color
st.set_page_config(
    page_title="Simulación US Open 2025 Determinista", # Título actualizado
    layout="wide"
)

# Inyección de CSS para centrado de títulos, color de detalles (azul/amarillo) y botón grande
st.markdown("""
<style>
/* Estilos Globales */
.stApp {
    background-color: #F8F8FF; /* Fondo blanco roto */
    color: #153E77; /* Texto principal en azul oscuro */
}
/* Centrar Títulos */
h1 {
    text-align: center;
    color: #153E77; /* Azul US Open */
}
/* Centrar Subtítulo (h3) */
h3 {
    text-align: center;
    color: #153E77;
}
/* Estilo del Contenedor de la Imagen (para centrar) */
[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}
/* Estilo del Botón de Simulación (Azul US Open) */
div.stButton > button {
    width: 100%; /* Estirarlo para centrar mejor en la columna */
    background-color: #153E77; /* Azul Oscuro (Color principal) */
    color: white;
    border: 3px solid #FFCD00; /* Borde amarillo (detalles) */
    font-size: 1.2em;
    padding: 15px;
    border-radius: 10px;
    transition: background-color 0.3s;
}
div.stButton > button:hover {
    background-color: #2F69B9;
    color: #FFCD00; /* Letras amarillas al pasar el mouse */
}
/* Estilo para la Caja de Éxito (Campeón) */
[data-testid="stSuccess"] {
    background-color: #153E77 !important; /* Azul Oscuro para el fondo */
    color: white !important;
    border-left: 8px solid #FFCD00 !important; /* Detalle amarillo */
}
/* Ocultar el índice del DataFrame */
.row-header {
    display: none; 
}
</style>
""", unsafe_allow_html=True)


# --------------------------
# LOGO CENTRADO Y TÍTULOS
# --------------------------
# Usamos columnas para centrar la imagen (3 columnas iguales para centrar el ancho)
col_img_1, col_img_2, col_img_3 = st.columns([1, 1, 1])
with col_img_2:
    st.image("US_OPEN_Logo.png", width=300)

# El título principal está centrado por el CSS h1
st.title("🎾 Predicción DETERMINISTA del US Open 2025") # Título actualizado
st.markdown(
    """
    ### ¿Quién será el campeón del último Grand Slam del año según la mayor probabilidad?
    ---
    """, unsafe_allow_html=True 
)

# --------------------------
# MOSTRAR CUADRO INICIAL (Más Amigable)
# --------------------------
st.subheader("Cuadro oficial de enfrentamientos")

# Usamos st.expander para una presentación limpia
with st.expander("Ver Cuadro Completo de 128 Jugadores"):
    df_cuadro = pd.DataFrame(cuadro_usopen2025)
    
    # Presentación: Ocultamos el índice por CSS y usamos st.dataframe
    st.dataframe(
        df_cuadro, 
        use_container_width=True,
    )

st.markdown("---")

# --------------------------
# BOTÓN CENTRADO Y GRANDE
# --------------------------
st.subheader("🏆 Simulación del Torneo")

# Usamos columnas para centrar el botón de forma efectiva
col_a, col_b, col_c = st.columns([1, 3, 1])
with col_b:
    if st.button("🔥 PREDECIR CAMPEÓN (Determinista) 🔥", type="primary"): # Texto del botón actualizado
        
        with st.spinner("Ejecutando Simulación Determinista..."): # Texto del spinner actualizado
            # 1. Ejecutar la simulación determinista
            df_progress = simular_torneo_presentacion( # <-- CAMBIO CLAVE
                modelo=modelofinalOK,
                cuadro_inicial=cuadro_usopen2025,
                jugadores_data=diccionario_jugadores_simulacion,
                h2h_cache=h2h_cache_final
            )

        # 2. PROCESAR Y MOSTRAR RESULTADOS
        
        # Filtrar rondas finales
        df_final = df_progress[
            df_progress['Ronda'].isin(['R4 (16)', 'QF (8)', 'SF (4)', 'Final (2)'])
        ].copy()

        df_final["Probabilidad_P1"] = (df_final["Probabilidad_P1"] * 100).round(2).astype(str) + "%"

        # 3. Mostrar tabla de progresión
        st.subheader("📊 Resultados Proyectados (Determinista - desde octavos de final)") # Título actualizado
        st.dataframe(df_final, use_container_width=True, hide_index=True)

        # 4. Anuncio del Campeón (Centrado, Grande y con Colores del US Open)
        campeon = df_progress["Ganador"].iloc[-1]
        
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: 50px; background-color: #153E77; padding: 20px; border-radius: 15px; border: 4px solid #FFCD00;'>
                <div style='font-size: 2em; color: white; font-weight: bold;'>
                    ¡EL CAMPEÓN ES...!
                </div>
                <h1 style='font-size: 4.5em; color: #FFCD00; margin-top: 10px; margin-bottom: 0px;'>
                    {campeon}
                </h1>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.info("Clickeá para obtener la predicción usando el resultado más probable en cada partido (determinista).") # Texto info actualizado