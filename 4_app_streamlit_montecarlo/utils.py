import numpy as np
import pandas as pd
import joblib
from itertools import combinations
from collections import defaultdict
import random



lista_jugadores_usopen = ['Shevchenko A.', 'Davidovich Fokina A.',
                          'Rinderknech A.', 'Carballes Baena R.',
                          'Jarry N.', 'Mensik J.',
                          'Shelton B.', 'Buse I.',
                          'Nava E.', 'Fritz T.',
                          'Blanchet U.', 'Marozsan F.', 
                          'Dostanic S.', 'Spizzirri E.',
                          'Darderi L.', 'Hijikata R.',
                          'Coric B.', 'Lehecka J.',
                          'Mannarino A.', 'Griekspoor T.',
                          'Nardi L.', 'Machac T.',
                          'Svajda Z.', 'Piros Z.',
                          'Quinn E.', 'Kym J.',
                          'Carreno Busta P.', 'Llamas Ruiz P.',
                          'Ugo Carabelli C.', 'Etcheverry T.',
                          'Thompson J.', 'Moutet C.',
                          'Navone M.', 'Giron M.',
                          
                          'Nakashima B.', 'De Jong J.',
                          'Tien L.', 'Djokovic N.',
                          'Bonzi B.', 'Medvedev D.',
                          'Baez S.', 'Harris L.',
                          'Damm M.', 'Blanch D.',
                          'Kovacevic A.', 'Wong C.',
                          'Walton A.', 'Humbert U.',
                          'Garin C.', 'Bergs Z.', 
                          'Norrie C.', 'Korda S.',
                          'Kecmanovic M.', 'Fonseca J.',
                          'Draper J.', 'Gomez F.',
                          'Galan D.', 'Collignon R.',
                          'Comesana F.', 'Michelsen A.',
                          'Passaro F.', 'Cobolli F.',
                          'Bellucci M.', 'Shang J.',
                          'Vukic A.', 'Brooksby J.',
                          'Rune H.', 'Van De Zandschulp B.',
                          'Nishioka Y.', 'Tiafoe F.',
                          'Munar J.', 'Faria J.',
                          'Dellien H.', 'Majchrzak K.',
                          'Struff J.L.', 'Mcdonald M.',
                          'Duckworth J.', 'Boyer T.',
                          'Khachanov K.', 'Basavareddy N.',
                          'Prizmic D.', 'Rublev A.',
                          
                          'Ruud C.', 'Ofner S.',
                          'Dzumhur D.', 'Diallo G.',
                          'Alcaraz C.', 'Opelka R.',
                          'Fucsovics M.', 'Shapovalov D.',
                          'Cerundolo F.', 'Arnaldi M.', 
                          'Muller A.', 'Musetti L.',
                          'Goffin D.', 'Halys Q.',
                          'Cilic M.', 'Bublik A.',
                          'Riedi L.', 'Martinez P.',
                          'Kopriva V.', 'Sinner J.',
                          'Royer V.', 'Fearnley J.',
                          'Ruusuvuori E.', 'Popyrin A.',
                          'Auger-Aliassime F.', 'Harris B.',
                          'Gaston H.', 'Mochizuki S.',
                          'Altmaier D.', 'Medjedovic H.',
                          'Sonego L.', 'Schoolkate T.',
                          'Tsitsipas S.', 'Muller A.',
                          'Kecmanovic M.', 'De Minaur A.',
                          'Borges N.', 'Holt B.',
                          'Monfils G.', 'Safiullin R.',
                          
                          'Fearnley J.', 'Bautista Agut R.',
                          'Moller E.', 'Paul T.',
                          'Zverev A.', 'Tabilo A.']




# --- CONSTANTES DEL MODELO FINAL ---
X_TRAIN_COLUMNS_FINAL = [
    'Elo_Diff', 'Elo_Surface_Diff', 'H2H_Advantage', 
    'Recent_Form_10_Diff', 'Age_Diff', 'Tournament_Titles_Diff_Dynamic', 
    'Grand_Titles_Diff_Dynamic', 'Round_Ordinal', 'Series_Ordinal', 
    'Court_Indoor', 'Best_of_5', 'Surface_Clay', 'Surface_Grass', 
    'Surface_Hard'
]

# --- CONSTANTES DEL TORNEO (US OPEN) ---
SERIES_ORDINAL_GS = 5
COURT_INDOOR = 0
BEST_OF_5 = 1
SURFACE_CLAY = 0
SURFACE_GRASS = 0
SURFACE_HARD = 1

def create_h2h_cache(df_historico: pd.DataFrame, lista_jugadores: list) -> dict:
    """
    Crea un diccionario de cache para el Head-to-Head de todos los pares de jugadores.
    La clave es la tupla ordenada: ('Jugador A', 'Jugador B').
    """
    h2h_cache = {}
    
    # Iterar sobre todos los pares ÚNICOS de jugadores del cuadro
    for player1, player2 in combinations(lista_jugadores, 2):
        
        # Buscar en el DF histórico (SOLO ESTE PASO ES LENTO, Y SOLO SE HACE UNA VEZ)
        h2h_matches = df_historico[
            ((df_historico['Player_1'] == player1) & (df_historico['Player_2'] == player2)) |
            ((df_historico['Player_1'] == player2) & (df_historico['Player_2'] == player1))
        ].copy()

        if h2h_matches.empty:
            continue 
        
        # Contar Victorias
        p1_wins = (h2h_matches['Winner'] == player1).sum()
        p2_wins = (h2h_matches['Winner'] == player2).sum()
        
        # Almacenar el resultado con la clave canónica (tupla ordenada)
        key = tuple(sorted((player1, player2)))
        h2h_cache[key] = {'P1_Wins': p1_wins, 'P2_Wins': p2_wins}
        
    return h2h_cache
def lookup_h2h_advantage_CACHED(h2h_cache, player1, player2):
    """
    Busca la ventaja H2H en el cache pre-calculado, lo cual es instantáneo.
    Retorna (Victorias P1 - Victorias P2).
    """
    # 1. Crear la clave canónica (ordenada)
    key = tuple(sorted((player1, player2)))
    
    # 2. Buscar en el cache
    record = h2h_cache.get(key)
    
    if record is None:
        return 0 # No hay historial entre ellos
    
    # 3. Determinar la ventaja desde la perspectiva de P1 (P1 - P2)
    # Debemos verificar qué jugador corresponde a 'P1_Wins' en el registro
    if player1 == key[0]:
        return record['P1_Wins'] - record['P2_Wins']
    else:
        # El jugador 1 (P1) es el segundo elemento de la tupla.
        return record['P2_Wins'] - record['P1_Wins']
def simular_partido_mc(proba_A_gana, nombre_A, nombre_B):
    """
    Decide el ganador de una simulación basado en la probabilidad (Monte Carlo).
    """
    r = random.random()
    
    if r < proba_A_gana:
        return nombre_A
    else:
        return nombre_B
def predecir_match_features_final(modelo, jugador_A_data, jugador_B_data, round_ordinal, h2h_cache):
    """
    Calcula el vector de 14 features de diferencia para el modelo y retorna la probabilidad de victoria de A.
    """
    
    data_vector = {
        # Habilidad Dinámica
        'Elo_Diff': [jugador_A_data['Elo_Absoluto'] - jugador_B_data['Elo_Absoluto']],
        'Elo_Surface_Diff': [jugador_A_data['Elo_Surface_Hard'] - jugador_B_data['Elo_Surface_Hard']],
        
        # Historial Directo (Lookup en el DF histórico)
        'H2H_Advantage': [lookup_h2h_advantage_CACHED(h2h_cache, jugador_A_data['Nombre'], jugador_B_data['Nombre'])],

        # Forma Reciente (Win Pct)
        'Recent_Form_10_Diff': [jugador_A_data['Forma_WinPct_Absoluto'] - jugador_B_data['Forma_WinPct_Absoluto']],
        
        # EXPERIENCIA ESTÁTICA
        'Age_Diff': [jugador_A_data['Age_Absoluta'] - jugador_B_data['Age_Absoluta']],
        'Tournament_Titles_Diff_Dynamic': [jugador_A_data['Titulos_Absoluto'] - jugador_B_data['Titulos_Absoluto']],
        'Grand_Titles_Diff_Dynamic': [jugador_A_data['Grand_Titles_Absoluto'] - jugador_B_data['Grand_Titles_Absoluto']],
        
        # CONTEXTO DEL TORNEO (CONSTANTES Y VARIABLES DE RONDA)
        'Round_Ordinal': [round_ordinal], 
        'Series_Ordinal': [SERIES_ORDINAL_GS], 
        'Court_Indoor': [COURT_INDOOR],
        'Best_of_5': [BEST_OF_5],
        'Surface_Clay': [SURFACE_CLAY],
        'Surface_Grass': [SURFACE_GRASS],
        'Surface_Hard': [SURFACE_HARD],
    }
    
    X_new = pd.DataFrame(data_vector, columns=X_TRAIN_COLUMNS_FINAL) 
    
    # Retorna la probabilidad de que gane P1 (Jugador A)
    return modelo.predict_proba(X_new)[0, 1]
def buscar_ultima_data_absoluta(df_historico: pd.DataFrame, nombre_jugador: str) -> dict:
    """
    Busca la última aparición de un jugador en el DF histórico y extrae sus 
    métricas absolutas (pre-torneo) necesarias para la simulación.
    """
    
    # Asegurarse de que el DF esté ordenado por fecha para tomar la última línea
    df = df_historico.sort_values(by='Date', ascending=False)
    
    # 1. Filtrar las filas donde el jugador aparece como P1 o P2
    df_player = df[
        (df['Player_1'] == nombre_jugador) | (df['Player_2'] == nombre_jugador)
    ].copy()
    
    if df_player.empty:
        # Esto ocurre si el jugador nunca ha jugado un partido en el histórico.
        return None 

    # 2. Tomar la última fila (la más reciente)
    ultima_fila = df_player.iloc[0]
    
    # 3. Determinar si el jugador fue P1 o P2 en ese último partido
    es_p1 = (ultima_fila['Player_1'] == nombre_jugador)
    
    # 4. Extracción de Métricas Absolutas
    
    # ** A) Métricas de Habilidad Absoluta (Elo) **
    # Tuvimos que inferir las columnas, asumimos que existen P1_Elo_Before/P2_Elo_Before
    # o que podemos inferirlas a partir de Elo_Diff
    
    # Usaremos una aproximación robusta: asumiremos que el Elo ABSOLUTO de ese 
    # último partido es el valor que el modelo ha aprendido.
    
    # NOTA: Debes tener la columna P1_Elo_Before/P2_Elo_Before en tu DF.
    # Si no la tienes, esto fallará y tendrás que volver a la etapa de Elo para guardarla.
    
    if 'P1_Elo_Before' not in df.columns:
        raise KeyError("La columna 'P1_Elo_Before' es requerida. Vuelve a calcular Elo en tu cuaderno 2 y guarda los scores absolutos.")

    # ELO General
    elo_abs = ultima_fila['P1_Elo_Before'] if es_p1 else ultima_fila['P2_Elo_Before']
    
    # ELO Surface Hard (Asumimos que el US Open es Hard)
    elo_hard = ultima_fila['P1_Elo_Surface'] if es_p1 else ultima_fila['P2_Elo_Surface']

    # ** B) Métricas de Forma Absoluta **
    # Asumimos que la forma se puede extraer del lado P1 o P2.
    # Nota: Wins_Last_5_Absoluto se eliminó del modelo final, por lo que solo usamos Recent_Form_10.
    
    # La forma se debe calcular *antes* de la predicción, pero si tu DF tiene P1_Recent_Win_Pct, úsalo.
    # Asumiremos que tenemos las columnas absolutas (ej. P1_Recent_Win_Pct)
    
    # NOTA: Debes tener la columna P1_Recent_Win_Pct en tu DF.
    if 'P1_Recent_Win_Pct' not in df.columns:
         raise KeyError("La columna 'P1_Recent_Win_Pct' es requerida. Vuelve a calcular Forma en tu cuaderno 2 y guarda los scores absolutos.")

    form_win_pct_10 = ultima_fila['P1_Recent_Win_Pct'] if es_p1 else ultima_fila['P2_Recent_Win_Pct']

    # ** C) Métricas Estáticas/Dinámicas (Títulos/Edad) **
    
    titulos_totales = ultima_fila['P1_Tournament_Titles_Dynamic'] if es_p1 else ultima_fila['P2_Tournament_Titles_Dynamic']
    titulos_grandes = ultima_fila['P1_Grand_Titles_Dynamic'] if es_p1 else ultima_fila['P2_Grand_Titles_Dynamic']
    edad_abs = ultima_fila['Age_Player1'] if es_p1 else ultima_fila['Age_Player2']

    # 5. Retornar el diccionario con las métricas absolutas
    return {
        "Nombre": nombre_jugador,
        "Elo_Absoluto": elo_abs,
        "Elo_Surface_Hard": elo_hard,
        "Forma_WinPct_Absoluto": form_win_pct_10,
        "Titulos_Absoluto": titulos_totales,
        "Grand_Titles_Absoluto": titulos_grandes,
        "Age_Absoluta": edad_abs,
    }
def crear_diccionario_jugadores(df_historico, lista_jugadores: list) -> dict:
    """
    Crea el diccionario final de jugadores buscando las últimas métricas absolutas.
    """
    diccionario_final = {}
    
    for jugador in lista_jugadores:
        data = buscar_ultima_data_absoluta(df_historico, jugador)
        if data:
            diccionario_final[jugador] = data
        else:
            print(f"⚠️ Advertencia: Jugador {jugador} no encontrado en el historial. Ignorando.")
            # Si un jugador no está en el historial (ej. novato), puedes darle un ELO base (ej. 1500)
            
    return diccionario_final
def simular_torneo(modelo, cuadro_inicial, jugadores_data, h2h_cache, num_simulaciones=500):
    
    # Contenedor para contar victorias en la Final
    conteo_campeones = defaultdict(int)
    
    # Bucle principal de Monte Carlo
    for sim in range(num_simulaciones):
        
        # El cuadro comienza con los partidos de la Ronda 1
        cuadro_actual = [p.copy() for p in cuadro_inicial] 
        ronda_num = 1
        
        # Bucle de avance por rondas (R1 -> R2 -> ... -> Final)
        while len(cuadro_actual) >= 1:
            partidos_siguiente_ronda = []
            
            # 1. Simular todos los partidos de la ronda actual
            for partido in cuadro_actual:
                p1_name = partido["P1"]
                p2_name = partido["P2"]
                
                # Obtener la data absoluta de los jugadores (habilidad, edad, títulos)
                p1_data = jugadores_data.get(p1_name)
                p2_data = jugadores_data.get(p2_name)
                
                # 🚨 Control de jugadores no encontrados (ej. si no están en el DF histórico)
                if p1_data is None or p2_data is None:
                    # Asumimos que el jugador con data gana, o se le da 50%
                    ganador = p1_name if p2_data is None else p2_name
                else:
                    # 2. Calcular probabilidad de victoria de P1
                    proba_p1 = predecir_match_features_final(
                        modelo, p1_data, p2_data, ronda_num, h2h_cache
                    )
                    
                    # 3. Decidir el ganador por Monte Carlo
                    ganador = simular_partido_mc(proba_p1, p1_name, p2_name)
                
                partidos_siguiente_ronda.append(ganador)
            
            # 4. Condición de Parada: Campeón encontrado
            if len(partidos_siguiente_ronda) == 1:
                campeon = partidos_siguiente_ronda[0]
                conteo_campeones[campeon] += 1
                break
            
            # 5. Construir la siguiente ronda
            cuadro_actual = []
            for i in range(0, len(partidos_siguiente_ronda), 2):
                cuadro_actual.append(
                    {"P1": partidos_siguiente_ronda[i], "P2": partidos_siguiente_ronda[i+1]}
                )
            
            ronda_num += 1

    # 6. Resultados finales: convertir conteo a probabilidad (0.0 a 1.0)
    resultados = {
        jugador: conteo / num_simulaciones 
        for jugador, conteo in conteo_campeones.items()
    }
    return dict(sorted(resultados.items(), key=lambda item: item[1], reverse=True))
def simular_torneo_run_aleatorio(modelo, cuadro_inicial, jugadores_data, h2h_cache):
    """
    Simula el avance de un torneo solo una vez, introduciendo aleatoriedad 
    basada en la probabilidad del modelo (Un solo Run de Monte Carlo).
    """
    
    registro_partidos = []
    cuadro_actual = [p.copy() for p in cuadro_inicial] 
    ronda_num = 1
    
    ronda_nombres = {
        1: "R1 (128)", 2: "R2 (64)", 3: "R3 (32)", 4: "R4 (16)",
        5: "QF (8)", 6: "SF (4)", 7: "Final (2)"
    }
    
    # Bucle de avance por rondas
    while len(cuadro_actual) >= 1:
        partidos_siguiente_ronda = []
        ronda_key = ronda_nombres.get(ronda_num, f"Ronda {ronda_num}")
        
        for partido in cuadro_actual:
            p1_name = partido["P1"]
            p2_name = partido["P2"]
            
            p1_data = jugadores_data.get(p1_name)
            p2_data = jugadores_data.get(p2_name)
            
            # --- Predicción de Probabilidad ---
            if p1_data is None or p2_data is None:
                ganador = p1_name if p2_data is None else p2_name
                probabilidad = 0.50
            else:
                probabilidad = predecir_match_features_final(
                    modelo, p1_data, p2_data, ronda_num, h2h_cache
                )
                
                # 🚨 DECISIÓN ALEATORIA: Usa Monte Carlo
                ganador = simular_partido_mc(probabilidad, p1_name, p2_name)
            
            registro_partidos.append({
                'Ronda': ronda_key,
                'P1': p1_name,
                'P2': p2_name,
                'Ganador': ganador,
                'Probabilidad_P1': probabilidad
            })
            
            partidos_siguiente_ronda.append(ganador)
        
        if len(partidos_siguiente_ronda) == 1:
            break
        
        # Construir la siguiente ronda
        cuadro_actual = []
        for i in range(0, len(partidos_siguiente_ronda), 2):
            cuadro_actual.append(
                {"P1": partidos_siguiente_ronda[i], "P2": partidos_siguiente_ronda[i+1]}
            )
        
        ronda_num += 1

    return pd.DataFrame(registro_partidos)