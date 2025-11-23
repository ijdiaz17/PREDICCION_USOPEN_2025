import pandas as pd
import joblib
import os

# --- 1. Importar funciones auxiliares y listas fijas ---
# Asumo que las funciones auxiliares (crear_diccionario_jugadores, create_h2h_cache) están en utils.py
from utils import crear_diccionario_jugadores, create_h2h_cache, lista_jugadores_usopen

# --- 2. Definición de Variables Estructurales ---
# Se necesita para que app.py lo importe directamente
cuadro_usopen2025 = [

# Partido 1 (Ganador va contra Ganador P2)
{"P1": "Kopriva V.", "P2": "Sinner J."},

# Partido 2
{"P1": "Popyrin A.", "P2": "Ruusuvuori E."},

 # Partido 3  (Ganador va contra Ganador P4)
{"P1": "Royer V.", "P2": "Fearnley J."},

# Partido 4
{"P1": "Fucsovics M.", "P2": "Shapovalov D."},

# Partido 5 (Ganador va contra Ganador P6)
{"P1": "Cilic M.", "P2": "Bublik A."},

# Partido 6 (Ganador va contra Ganador P7)
{"P1": "Sonego L.", "P2": "Schoolkate T."},

# Partido 7 (Ganador va contra Ganador P8)
{"P1": "Borges N.", "P2": "Holt B."},
 
# Partido 8
{"P1": "Moller E.", "P2": "Paul T."},
 
#
 
# Partido 9 (Ganador va contra Ganador P10)
{"P1": "Muller A.", "P2": "Musetti L."},
 
# Partido 10
{"P1": "Goffin D.", "P2": "Halys Q."},
 
 # Partido 11 (Ganador va contra Ganador P12)
{"P1": "Vukic A.", "P2": "Brooksby J."},
 
# Partido 12
{"P1": "Passaro F.", "P2": "Cobolli F."},

# Partido 13 (Ganador va contra Ganador P14)
{"P1": "Dzumhur D.", "P2": "Diallo G."},

# Partido 14
{"P1": "Munar J.", "P2": "Faria J."},

# Partido 15 (Ganador va contra Ganador P16)
{"P1": "Garin C.", "P2": "Bergs Z."},

# Partido 16
{"P1": "Draper J.", "P2": "Gomez F."},
 
#

# Partido 17 (Ganador va contra Ganador P18)
{"P1": "Zverev A.", "P2": "Tabilo A."},

# Partido 18
{"P1": "Fearnley J.", "P2": "Bautista Agut R."},

# Partido 19 (Ganador va contra Ganador P20)
{"P1": "Monfils G.", "P2": "Safiullin R."},

# Partido 20
{"P1": "Auger-Aliassime F.", "P2": "Harris B."},

# Partido 21 (Ganador va contra Ganador P22)
{"P1": "Walton A.", "P2": "Humbert U."},

# Partido 22
{"P1": "Kovacevic A.", "P2": "Wong C."},

# Partido 23 (Ganador va contra Ganador P24)
{"P1": "Duckworth J.", "P2": "Boyer T."},

# Partido 24
{"P1": "Prizmic D.", "P2": "Rublev A."},

#

# Partido 25 (Ganador va contra Ganador P26)
{"P1": "Khachanov K.", "P2": "Basavareddy N."},

# Partido 26
{"P1": "Dellien H.", "P2": "Majchrzak K."},

# Partido 27 (Ganador va contra Ganador P28)
{"P1": "Riedi L.", "P2": "Martinez P."},

# Partido 28
{"P1": "Cerundolo F.", "P2": "Arnaldi M."},
 
# Partido 29 (Ganador va contra Ganador P30)
{"P1": "Tsitsipas S.", "P2": "Muller A."},

# Partido 30
{"P1": "Altmaier D.", "P2": "Medjedovic H."},

# Partido 31 (Ganador va contra Ganador P32)
{"P1": "Gaston H.", "P2": "Mochizuki S."},

# Partido 32
{"P1": "Kecmanovic M.", "P2": "De Minaur A."},
 
#

# Partido 33 (Ganador va contra Ganador P34)
{"P1": "Tien L.", "P2": "Djokovic N."},

# Partido 34
{"P1": "Svajda Z.", "P2": "Piros Z."},

# Partido 35 (Ganador va contra Ganador P36)
{"P1": "Norrie C.", "P2": "Korda S."},
 
# Partido 36
{"P1": "Comesana F.", "P2": "Michelsen A."},

# Partido 37 (Ganador va contra Ganador P38)
{"P1": "Nishioka Y.", "P2": "Tiafoe F."},

# Partido 38
{"P1": "Damm M.", "P2": "Blanch D."},

# Partido 39 (Ganador va contra Ganador P40)
{"P1": "Struff J.L.", "P2": "Mcdonald M."},

# Partido 40
{"P1": "Rune H.", "P2": "Van De Zandschulp B."},

#

#Partido 41 (Ganador va contra Ganador P42)
{"P1": "Jarry N.", "P2": "Mensik J."},

# Partido 42
{"P1": "Blanchet U.", "P2": "Marozsan F."},
 
# Partido 43 (Ganador va contra Ganador P44)
{"P1": "Kecmanovic M.", "P2": "Fonseca J."},
 
# Partido 44
{"P1": "Nardi L.", "P2": "Machac T."},

# Partido 45 (Ganador va contra Ganador P46)
{"P1": "Nakashima B.", "P2": "De Jong J."},

# Partido 46
{"P1": "Quinn E.", "P2": "Kym J."},

# Partido 47 (Ganador va contra Ganador P48)
{"P1": "Baez S.", "P2": "Harris L."},

# Partido 48
{"P1": "Nava E.", "P2": "Fritz T."},

#

# Partido 49 (Ganador va contra Ganador P50)
{"P1": "Shelton B.", "P2": "Buse I."},

# Partido 50
{"P1": "Carreno Busta P.", "P2": "Llamas Ruiz P."},
 
# Partido 51 (Ganador va contra Ganador P52)
{"P1": "Thompson J.", "P2": "Moutet C."},

# Partido 52
{"P1": "Mannarino A.", "P2": "Griekspoor T."},

# Partido 53 (Ganador va contra Ganador P54)
{"P1": "Coric B.", "P2": "Lehecka J."},

# Partido 54
{"P1": "Ugo Carabelli C.", "P2": "Etcheverry T."},

# Partido 55 (Ganador va contra Ganador P56)
{"P1": "Galan D.", "P2": "Collignon R."},
 
# Partido 56
{"P1": "Ruud C.", "P2": "Ofner S."},
 
#

# Partido 57 (Ganador va contra Ganador P58)
{"P1": "Bonzi B.", "P2": "Medvedev D."},

# Partido 58
{"P1": "Navone M.", "P2": "Giron M."},

# Partido 59 (Ganador va contra Ganador P60)
{"P1": "Rinderknech A.", "P2": "Carballes Baena R."},

# Partido 60
{"P1": "Shevchenko A.", "P2": "Davidovich Fokina A."},

# Partido 61 (Ganador va contra Ganador P62)
{"P1": "Darderi L.", "P2": "Hijikata R."},
 
# Partido 62
{"P1": "Dostanic S.", "P2": "Spizzirri E."},
 
# Partido 63 (Ganador va contra Ganador P64)
{"P1": "Bellucci M.", "P2": "Shang J."},

# Partido 64
{"P1": "Alcaraz C.", "P2": "Opelka R."},

]

# --- 3. La Función de Carga (Núcleo de la eficiencia) ---
def load_data_and_caches():
    
    # 🚨 Lógica de Rutas Robusta (Desde 2_src/main.py)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "..", "0_data", "1_processed", "partidos_final.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "..", "3_models", "random_forest_modelofinalOK.joblib")

    # OPERACIONES PESADAS
    df = pd.read_csv(CSV_PATH)
    modelofinalOK = joblib.load(MODEL_PATH)

    # CREACIÓN DE CACHES (Lento, pero necesario)
    diccionario_jugadores_simulacion = crear_diccionario_jugadores(df, lista_jugadores_usopen)
    h2h_cache_final = create_h2h_cache(
        df_historico=df, 
        lista_jugadores=lista_jugadores_usopen
    )

    # Devolvemos todos los objetos cargados (el DF histórico ya no es necesario en app.py)
    return modelofinalOK, diccionario_jugadores_simulacion, h2h_cache_final

# ⚠️ Nota: No hay código de ejecución global después de la función.