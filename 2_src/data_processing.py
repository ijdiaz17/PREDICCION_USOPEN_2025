import pandas as pd

##################################################################################
df = pd.read_csv('0_data/0_raw/atp_tennis.csv')
print("✅Has leído el csv 'atp_tennis.csv' correctamente✅")
##################################################################################

df = df.iloc[:66105]

##################################################################################
jugadores = pd.read_csv("0_data/0_raw/atp_players_till_2022.csv")
print("✅Has leído el csv 'atp_players_till_2022.csv' correctamente✅")
##################################################################################

jugadores

# Crear columna en jugadores con formato: "Apellido I."

#print("Columnas disponibles:", jugadores.columns.tolist())

if 'name_last' in jugadores.columns and 'name_first' in jugadores.columns:
    def format_player_name(name, lastname):
        """
        Convierte 'John Michael' + 'García-López' -> 'García-López J.'
        Convierte 'Anna' + 'De La Cruz' -> 'De La Cruz A.'
        """
        if pd.isna(name) or pd.isna(lastname):
            return None
        
        # Limpiar espacios
        name_clean = str(name).strip()
        lastname_clean = str(lastname).strip()
        
        # Tomar la primera letra del PRIMER nombre (puede ser múltiple)
        first_initial = name_clean.split()[0][0].upper() if name_clean else ''
        
        # Construir: "Apellido I."
        return f"{lastname_clean} {first_initial}."
    
    # Aplicar función
    jugadores['Player_Name'] = jugadores.apply(
        lambda row: format_player_name(row['name_first'], row['name_last']), 
        axis=1
    )
    
    '''
    print("\n=== Nueva columna 'Player_Name' en jugadores ===")
    print(jugadores[['name_first', 'name_last', 'Player_Name']].head(15))
    print()
    '''

    # Comparar con df
    '''
    print("=== Ejemplo de Player_1 en df ===")
    print(df['Player_1'].head(15).values)
    print()
    '''

    # Verificar coincidencias
    #print("=== Verificación de coincidencias ===")
    coincidencias = df['Player_1'].isin(jugadores['Player_Name']).sum()
    total = len(df)
    #print(f"Coincidencias: {coincidencias}/{total} ({coincidencias/total*100:.2f}%)")
    
else:
    print("\nColumnas no coinciden. Por favor, verifica los nombres de las columnas:")
    print(jugadores.columns.tolist())
    print("\nSample data:")
    print(jugadores.head(3))

# Eliminar solo las filas donde 'dob' es NaN
#print(f"Jugadores antes: {len(jugadores)}")
#print(f"Filas con dob NaN: {jugadores['dob'].isna().sum()}")
jugadores = jugadores.dropna(subset=['dob'])
#print(f"Jugadores después de eliminar NaN en dob: {len(jugadores)}")
#print()

# Crear columna 'año' con los primeros 4 dígitos de 'dob'
jugadores['año'] = jugadores['dob'].astype(str).str[:4].astype(int)

'''
print("=== Nueva columna 'año' en jugadores ===")
print(jugadores[['dob', 'año']].head(10))
print()
print(f"Rango de años: {jugadores['año'].min()} - {jugadores['año'].max()}")
'''

jugadores["Player_Name"] = jugadores["Player_Name"].str.title()
jugadores
# hacer el merge
# Merge con df (tanto Player_1 como Player_2)

# Crear un diccionario con la edad de cada jugador
edad_dict = dict(zip(jugadores['Player_Name'], jugadores['año']))  # o la columna de edad

# Añadir edad a Player_1
df['Age_Player1'] = df['Player_1'].map(edad_dict)

# Añadir edad a Player_2
df['Age_Player2'] = df['Player_2'].map(edad_dict)

#print("Valores NaN en Age_1:", df['Age_Player1'].isna().sum())
#print("Valores NaN en Age_2:", df['Age_Player2'].isna().sum())
# Reemplazos manuales según tu lista de nombres correctos
# Se hace por coincidencia parcial del apellido para capturar variantes y luego
# se vuelve a mapear las edades.
replacements = {
    'Ferrero J.': 'Ferrero J.C.',
    'Garcia Lopez G.': 'Garcia-Lopez G.',
    'Tsonga J.': 'Tsonga J.W.',
    'Del Potro J.': 'Del Potro J.M.',
    'Chela J.': 'Chela J.I.',
    'Ramos A.': 'Ramos-Vinolas A.',
    'Mathieu P.': 'Mathieu P.H.',
    'Struff J.': 'Struff J.L.',
    'Auger Aliassime': 'Auger-Aliassime F.',
    '\tLu Y.': 'Lu Y.H.',
    'Lu Y.': 'Lu Y.H.',
    'Lee H.': 'Lee H.T.',
    'Gimeno Traver D': 'Gimeno-Traver D.',
    'Gambill J.': 'Gambill J.M.',
    'Bautista': 'Bautista R.',
    'Herbert P.': 'Herbert P.H.',
    'Roger Vasselin': 'Roger-Vasselin E.',
    'Ramirez Hidalgo R.': 'Ramirez-Hidalgo R.',
    'Bogomolov': 'Bogomolov A.',
    'Kuznetsov A.': 'Kuznetsov An.',
}
jugadores['Player_Name'] = jugadores['Player_Name'].astype(str).str.strip()
jugadores['Player_Name'] = jugadores['Player_Name'].replace(replacements)

edad_dict = dict(zip(jugadores['Player_Name'], jugadores['año']))
df['Age_Player1'] = df['Player_1'].map(edad_dict)
df['Age_Player2'] = df['Player_2'].map(edad_dict)
# Reemplazar NaN en Age_Player1 cuando Player_1 sea 'Auger-Aliassime F.'
mask = (df['Player_1'] == 'Auger-Aliassime F.') & (df['Age_Player1'].isna())
df.loc[mask, 'Age_Player1'] = 2000.0
#print(f"Filas actualizadas (Player_1 = Auger-Aliassime F.): {mask.sum()}")

# Reemplazar NaN en Age_Player1 cuando Player_1 sea 'Auger-Aliassime F.'
mask = (df['Player_2'] == 'Auger-Aliassime F.') & (df['Age_Player2'].isna())
df.loc[mask, 'Age_Player2'] = 2000.0
#print(f"Filas actualizadas (Player_1 = Auger-Aliassime F.): {mask.sum()}")

#print("Valores NaN en Age_1:", df['Age_Player1'].isna().sum())
#print("Valores NaN en Age_2:", df['Age_Player2'].isna().sum())
df = df.dropna(subset=['Age_Player1'])
df = df.dropna(subset=['Age_Player2'])
# # Filtrar solo finales para contar títulos de torneo
# df_finals = df[df['Round'] == 'The Final']
# title_counts = df_finals['Winner'].value_counts().reset_index()
# title_counts.columns = ['Player', 'Tournament_Titles']

# # Merge y Diferencia
# df = df.merge(title_counts, left_on='Player_1', right_on='Player', how='left').rename(columns={'Tournament_Titles': 'P1_Tournament_Titles'}).drop(columns=['Player'])
# df = df.merge(title_counts, left_on='Player_2', right_on='Player', how='left').rename(columns={'Tournament_Titles': 'P2_Tournament_Titles'}).drop(columns=['Player'])
df.head()

##################################################################################
df.to_csv('0_data/1_processed/partidos_limpios.csv', index=False)
print("💾Has exportado 'partidos_limpios.csv' en '0_data/1_processed' correctamente💾")
##################################################################################

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

import warnings

# Configuraciones
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

##################################################################################
df = pd.read_csv("0_data/1_processed/partidos_limpios.csv")
print("✅Has leído el csv 'partidos_limpios.csv' correctamente✅")
##################################################################################

df
df["Rank_Diff"] = df["Rank_1"] - df["Rank_2"]
df
df["Player_1"] = df["Player_1"].str.strip()
# Target Variable (Y): 1 si Jugador 1 gana, 0 si Jugador 2 gana
df['Target'] = (df['Winner'] == df['Player_1']).astype(int)
df.head()
# --- 1. Mapeo Ordinal para 'Series' (Importancia del Torneo) ---
series_mapping = {
    'ATP250': 1,
    'International': 1,    
    'International Gold': 2,
    'ATP500': 2,
    'Masters': 3,
    'Masters 1000': 3,
    'Masters Cup': 4,
    'Grand Slam': 5
}

df['Series_Ordinal'] = df['Series'].map(series_mapping)

# --- 2. Mapeo Ordinal para 'Round' (Etapa del Torneo) ---
# Escala: 1 (Round Robin) -> 2 (1st Round) -> ... -> 8 (The Final)
round_mapping = {
    'Round Robin': 1,
    '1st Round': 1,
    '2nd Round': 2,
    '3rd Round': 3,
    '4th Round': 4,
    'Quarterfinals': 5,
    'Semifinals': 6,
    'The Final': 7
}

# Aplicar las transformaciones
df['Round_Ordinal'] = df['Round'].map(round_mapping)

# 2. Formato de Fecha (Crucial para ordenamiento cronológico)
df['Date'] = pd.to_datetime(df['Date'])
# 3. Codificación Binaria/OHE: Surface, Court, Best of ---
# Surface: One-Hot Encoding (OHE)
df = pd.concat([df, pd.get_dummies(df['Surface'], prefix='Surface', dtype=int)], axis=1)

# Court: Binaria (1 para Indoor, 0 para Outdoor)
df['Court_Indoor'] = (df['Court'] == 'Indoor').astype(int)

# Best of: Binaria (1 para 5 sets, 0 para 3 sets)
df['Best_of_5'] = (df['Best of'] == 5).astype(int)
df["Age_Diff"] = df["Age_Player1"] - df["Age_Player2"]
df["Target"].value_counts(normalize=True)

# --- 4. FEATURE DINÁMICA: FORMA RECIENTE (WINS LAST 5 DIFF) ---

# Crear el log de partidos desde la perspectiva del 'Player' (para el rolling)
p1_log = df[['Date', 'Player_1', 'Target']].rename(columns={'Player_1': 'Player', 'Target': 'Win_Flag'})
p2_log = df[['Date', 'Player_2', 'Target']].rename(columns={'Player_2': 'Player', 'Target': 'P1_Win_Flag'})
p2_log['Win_Flag'] = 1 - p2_log['P1_Win_Flag']
p2_log.drop(columns=['P1_Win_Flag'], inplace=True)

player_match_log = pd.concat([p1_log, p2_log], ignore_index=True)
player_match_log.sort_values(by=['Player', 'Date'], inplace=True)

# Calcular Victorias en la Ventana Móvil de 5 Partidos
player_match_log['P_Wins_Last_5'] = (
    player_match_log
    .groupby('Player')['Win_Flag']
    .rolling(window=5, min_periods=1)
    .sum()
    .reset_index(level=0, drop=True)
    .shift(1) # Desplazar 1 para asegurar la retrospectividad
    .fillna(0)
)

# IMPORTANTE: Eliminar duplicados antes del merge
# Mantener la primera ocurrencia de cada (Date, Player)
player_match_log = player_match_log.drop_duplicates(subset=['Date', 'Player'], keep='first').reset_index(drop=True)

# Merge la forma reciente de vuelta al DF original
df = df.merge(player_match_log[['Date', 'Player', 'P_Wins_Last_5']], left_on=['Date', 'Player_1'], right_on=['Date', 'Player'], how='left').rename(columns={'P_Wins_Last_5': 'P1_Wins_Last_5'}).drop(columns=['Player'])
df = df.merge(player_match_log[['Date', 'Player', 'P_Wins_Last_5']], left_on=['Date', 'Player_2'], right_on=['Date', 'Player'], how='left').rename(columns={'P_Wins_Last_5': 'P2_Wins_Last_5'}).drop(columns=['Player'])

# Crear la Feature de Diferencia Final
df['Wins_Last_5_Diff'] = df['P1_Wins_Last_5'].fillna(0) - df['P2_Wins_Last_5'].fillna(0)
df.head()
def calculate_h2h(df):
    """
    Calcula el historial Head-to-Head (H2H) para cada partido.
    El resultado es: (Victorias de Player_1 vs Player_2) - (Victorias de Player_2 vs Player_1)
    """
    h2h_df = df[['Date', 'Player_1', 'Player_2', 'Winner']].copy()
    
    # Inicializar la columna de H2H
    df['H2H_Advantage'] = 0.0
    
    # Crear un diccionario para almacenar los resultados históricos (PlayerA vs PlayerB)
    # La llave será una tupla ordenada (min(A, B), max(A, B))
    historical_h2h = {} 

    for index, row in h2h_df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        winner = row['Winner']

        # Crear la llave canónica para el par de jugadores (asegura que A vs B sea igual a B vs A)
        players_key = tuple(sorted((p1, p2)))

        # Obtener el historial actual o inicializarlo
        # El historial es: [Victorias_P1, Victorias_P2]
        h2h_record = historical_h2h.get(players_key, {p1: 0, p2: 0})
        
        # 1. Aplicar la Feature (antes de actualizar el historial)
        # La ventaja H2H se calcula desde la perspectiva de Player_1:
        # H2H_Advantage = Victorias_P1 - Victorias_P2
        
        # Primero, obtener los conteos de victorias del diccionario
        wins_p1 = h2h_record.get(p1, 0)
        wins_p2 = h2h_record.get(p2, 0)

        # Si el key está en orden P1, P2, usarlo directamente
        if p1 == players_key[0]:
            df.loc[index, 'H2H_Advantage'] = wins_p1 - wins_p2
        # Si el key está en orden P2, P1, hay que invertir el signo
        else:
            df.loc[index, 'H2H_Advantage'] = wins_p2 - wins_p1

        # 2. Actualizar el historial para el próximo partido
        if winner == p1:
            h2h_record[p1] = h2h_record.get(p1, 0) + 1
        elif winner == p2:
            h2h_record[p2] = h2h_record.get(p2, 0) + 1
            
        # Guardar el nuevo historial
        historical_h2h[players_key] = h2h_record

    return df

# Aplica la función
df = calculate_h2h(df.copy())
def calculate_recent_form(df, n_matches=10):
    """
    Calcula el porcentaje de victorias en los últimos N partidos para cada jugador.
    """
    
    # Crear un DataFrame para registrar el historial de victorias y partidos jugados de cada jugador
    player_history = {} 
    
    # Inicializar las columnas de forma reciente
    df['P1_Recent_Win_Pct'] = 0.0
    df['P2_Recent_Win_Pct'] = 0.0

    for index, row in df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        winner = row['Winner']

        # --- Calcular la Feature (antes de actualizar el historial) ---
        
        # Función auxiliar para obtener el % de victorias en los últimos N partidos
        def get_win_pct(player):
            if player not in player_history:
                return 0.0 # Si no hay historial, el porcentaje es 0
            
            # Historial es una lista de 1 (victoria) o 0 (derrota), limitada a los últimos N
            history = player_history[player]
            if not history:
                 return 0.0
            
            # Suma de victorias / número de partidos jugados
            return sum(history) / len(history)

        # Aplicar el porcentaje de victorias reciente
        df.loc[index, 'P1_Recent_Win_Pct'] = get_win_pct(p1)
        df.loc[index, 'P2_Recent_Win_Pct'] = get_win_pct(p2)
        
        # --- Actualizar el Historial ---

        # P1: agregar resultado y limitar a N
        p1_result = 1 if winner == p1 else 0
        if p1 not in player_history:
            player_history[p1] = []
        player_history[p1].append(p1_result)
        if len(player_history[p1]) > n_matches:
            player_history[p1].pop(0) # Eliminar el más antiguo

        # P2: agregar resultado y limitar a N
        p2_result = 1 if winner == p2 else 0
        if p2 not in player_history:
            player_history[p2] = []
        player_history[p2].append(p2_result)
        if len(player_history[p2]) > n_matches:
            player_history[p2].pop(0) # Eliminar el más antiguo

    return df

# Aplica la función (usando 10 partidos como N por defecto)
df = calculate_recent_form(df.copy(), n_matches=10)

# Opcional: crea una feature de diferencia
df['Recent_Form_10_Diff'] = df['P1_Recent_Win_Pct'] - df['P2_Recent_Win_Pct']
def calculate_elo(df, initial_elo=1500, k_factor=32, d_factor=400):
    """
    Calcula el Elo Rating para todos los jugadores a lo largo del tiempo.
    El cálculo se hace estrictamente por orden cronológico.
    """
    
    # 1. Preparación de datos (reasegurar orden y tipo de fecha)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # 2. Inicializar scores y listas de resultados
    player_elos = {} # Diccionario para almacenar el Elo actual de cada jugador
    
    # Inicializar columnas para las nuevas features
    df['P1_Elo_Before'] = 0.0
    df['P2_Elo_Before'] = 0.0

    # 3. Iterar cronológicamente y calcular/actualizar el Elo
    for index, row in df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        winner = row['Winner']

        # Asegurar que ambos jugadores tengan un Elo inicial
        if p1 not in player_elos:
            player_elos[p1] = initial_elo
        if p2 not in player_elos:
            player_elos[p2] = initial_elo

        # Obtener los Elos actuales
        r_p1 = player_elos[p1]
        r_p2 = player_elos[p2]
        
        # 4. Aplicar la Feature (Elos ANTES del partido)
        df.loc[index, 'P1_Elo_Before'] = r_p1
        df.loc[index, 'P2_Elo_Before'] = r_p2
        
        # 5. Calcular la Puntuación Esperada (E)
        # E = 1 / (1 + 10^((R_oponente - R_propio) / D))
        e_p1 = 1 / (1 + 10**((r_p2 - r_p1) / d_factor))
        e_p2 = 1 / (1 + 10**((r_p1 - r_p2) / d_factor))

        # 6. Definir la Puntuación Real (S)
        # S = 1 (Victoria) o 0 (Derrota)
        s_p1 = 1.0 if winner == p1 else 0.0
        s_p2 = 1.0 if winner == p2 else 0.0
        
        # 7. Actualizar el Elo (R' = R + K * (S - E))
        player_elos[p1] = r_p1 + k_factor * (s_p1 - e_p1)
        player_elos[p2] = r_p2 + k_factor * (s_p2 - e_p2)

    # 8. Crear la feature de diferencia (la más predictiva)
    df['Elo_Diff'] = df['P1_Elo_Before'] - df['P2_Elo_Before']
    
    return df

# --- Ejecución ---

# Aplica la función Elo
df = calculate_elo(df)
def calculate_elo_surface(df, initial_elo=1500, k_factor=32, d_factor=400):
    """
    Calcula el Elo Rating para cada jugador a lo largo del tiempo,
    manteniendo scores separados para cada superficie (Surface).

    El resultado es: (P1 Elo en la superficie) - (P2 Elo en la superficie).
    """
    
    # 1. Preparación de datos (asegura el orden cronológico estricto)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # 2. Inicializar scores: Diccionario anidado {Superficie: {Jugador: Elo}}
    player_elos = {
        'Hard': {},
        'Clay': {},
        'Grass': {},
        'Carpet': {} 
    }
    
    # Inicializar columnas para las nuevas features
    df['P1_Elo_Surface'] = 0.0
    df['P2_Elo_Surface'] = 0.0

    # 3. Iterar cronológicamente y calcular/actualizar el Elo
    for index, row in df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        winner = row['Winner']
        surface = row['Surface'] # Leer la superficie del partido actual
        
        # Obtener la referencia al diccionario de Elo de la superficie actual
        elos_on_surface = player_elos[surface]
        
        # Asegurar que ambos jugadores tengan un Elo inicial en esta superficie
        if p1 not in elos_on_surface:
            elos_on_surface[p1] = initial_elo
        if p2 not in elos_on_surface:
            elos_on_surface[p2] = initial_elo

        # Obtener los Elos actuales (ANTES del partido)
        r_p1 = elos_on_surface[p1]
        r_p2 = elos_on_surface[p2]
        
        # 4. Aplicar la Feature (Elos ANTES del partido)
        df.loc[index, 'P1_Elo_Surface'] = r_p1
        df.loc[index, 'P2_Elo_Surface'] = r_p2
        
        # 5. Calcular la Puntuación Esperada (E)
        e_p1 = 1 / (1 + 10**((r_p2 - r_p1) / d_factor))
        e_p2 = 1 / (1 + 10**((r_p1 - r_p2) / d_factor))

        # 6. Definir la Puntuación Real (S)
        s_p1 = 1.0 if winner == p1 else 0.0
        s_p2 = 1.0 if winner == p2 else 0.0
        
        # 7. Actualizar el Elo y guardarlo en el diccionario de la superficie
        elos_on_surface[p1] = r_p1 + k_factor * (s_p1 - e_p1)
        elos_on_surface[p2] = r_p2 + k_factor * (s_p2 - e_p2)

    # 8. Crear la feature de diferencia
    df['Elo_Surface_Diff'] = df['P1_Elo_Surface'] - df['P2_Elo_Surface']
    
    return df

# --- Ejecución ---
# Nota: La función 'calculate_elo' ya debe haber ordenado el DF por fecha.
# Si la usas de forma independiente, asegúrate de ordenar el DF antes.
df = calculate_elo_surface(df.copy())
df.head()
def calculate_dynamic_titles(df, player_col, win_col, new_col_name):
    """
    Calcula el total de títulos ganados por un jugador hasta el partido anterior (retrospectivo).
    """
    
    # 1. Crear un DF temporal solo con los datos necesarios y el índice
    df_temp = df[[player_col, 'Date', win_col]].copy()
    
    # 2. Ordenar por jugador y fecha, y resetear índice
    df_temp = df_temp.sort_values(by=[player_col, 'Date']).reset_index()
    
    # 3. Calcular la suma acumulativa (Títulos en cada partido)
    df_temp['Titles_Acum'] = df_temp.groupby(player_col)[win_col].cumsum()
    
    # 4. Aplicar SHIFT(1): El jugador solo tiene los títulos GANADOS HASTA ANTES de este partido.
    df_temp[new_col_name] = df_temp.groupby(player_col)['Titles_Acum'].shift(1).fillna(0)
    
    # 5. Fusionar de nuevo al DF original usando el índice original ('index')
    # Nos aseguramos de que solo la columna de resultado se fusione
    df = df.merge(
        df_temp[['index', new_col_name]],
        on='index', 
        how='left'
    )
    
    # 6. Devolver el DF
    return df
# Asegurarse de que el DF tenga una columna 'index' antes de llamar
df = df.reset_index(names='index') 

# --- A. Corregir Títulos de Torneo (Todos) ---

# 1. Marcar si una fila representa una victoria en una final
df['Title_Win'] = (df['Round_Ordinal'] == 7).astype(int) 
df['P1_Title_Won'] = ((df['Winner'] == df['Player_1']) & (df['Title_Win'] == 1)).astype(int)
df['P2_Title_Won'] = ((df['Winner'] == df['Player_2']) & (df['Title_Win'] == 1)).astype(int)

# 2. Aplicar la lógica dinámica
df = calculate_dynamic_titles(df, 'Player_1', 'P1_Title_Won', 'P1_Tournament_Titles_Dynamic')
df = calculate_dynamic_titles(df, 'Player_2', 'P2_Title_Won', 'P2_Tournament_Titles_Dynamic')

df['Tournament_Titles_Diff_Dynamic'] = df['P1_Tournament_Titles_Dynamic'] - df['P2_Tournament_Titles_Dynamic']

# --- B. Corregir Títulos Grandes (Grand Slams y Masters Cup) ---

# 1. Marcar si una fila representa una victoria en una final GRANDE (Series_Ordinal >= 4)
df['Grand_Title_Win'] = ((df['Round_Ordinal'] == 7) & (df['Series_Ordinal'] >= 4)).astype(int)
df['P1_Grand_Title_Won'] = ((df['Winner'] == df['Player_1']) & (df['Grand_Title_Win'] == 1)).astype(int)
df['P2_Grand_Title_Won'] = ((df['Winner'] == df['Player_2']) & (df['Grand_Title_Win'] == 1)).astype(int)

# 2. Aplicar la lógica dinámica
df = calculate_dynamic_titles(df, 'Player_1', 'P1_Grand_Title_Won', 'P1_Grand_Titles_Dynamic')
df = calculate_dynamic_titles(df, 'Player_2', 'P2_Grand_Title_Won', 'P2_Grand_Titles_Dynamic')

df['Grand_Titles_Diff_Dynamic'] = df['P1_Grand_Titles_Dynamic'] - df['P2_Grand_Titles_Dynamic']

# 3. Eliminar columnas auxiliares y estáticas viejas
df = df.drop(columns=[
    'P1_Tournament_Titles', 'P2_Tournament_Titles', 'Tournament_Titles_Diff', 
    'P1_Grand_Titles', 'P2_Grand_Titles', 'Grand_Titles_Diff'
], errors='ignore')
df[df["Tournament"]=="US Open"].tail()

##################################################################################
df.to_csv('0_data/1_processed/partidos_final.csv', index=False)
print("💾Has exportado 'partidos_final.csv' en '0_data/1_processed' correctamente💾")
##################################################################################
