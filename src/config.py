"""
CONFIGURACION UNICA DEL PROYECTO — Stuff+ calibrado por altitud (ISAC / Diablos).

YA ESTA AJUSTADO AL DICCIONARIO OFICIAL (stuff_plus_data_dictionary.csv).
El dia del hackathon normalmente solo se revisa:
  1. ALTITUDE_M  -> las etiquetas EXACTAS de la columna altitude_category del
                    dataset real y los metros que representa cada una
                    (el pipeline se detiene y te lista las que falten)
  2. HOME_ALTITUDE -> cual de esas etiquetas es la del Harp Helu
  3. ALTITUDE_RHO -> la densidad del aire de cada etiqueta (si una etiqueta no
                    esta, se usa la atmosfera estandar y el pipeline lo avisa)
  4. PITCH_CALL_MAP / HIT_TYPE_MAP / PLAY_RESULT_MAP -> solo si aparecen etiquetas
                    nuevas (el pipeline avisa cuantos pitcheos no pudo clasificar)
  5. EQUIPO / INTEGRANTES / INSTITUCION -> portada del documento

Todo lo demas lee los nombres canonicos y no hay que modificarlo.
"""
import os
from pathlib import Path

try:
    import pandas as pd
except ImportError:      # sin pandas config se importa igual: el aviso claro de que falta lo da run_all.py
    pd = None

# El codigo se probo con pandas 3, que siempre trabaja con copy-on-write. En
# pandas 2 se enciende aqui (todos los modulos importan config) para que una
# tabla sacada de otra se comporte igual en las dos versiones. En pandas 3 la
# opcion ya no se toca: esta en desuso y solo emite una advertencia.
if pd is not None and pd.__version__.split(".")[0] == "2":
    pd.options.mode.copy_on_write = True

ROOT = Path(__file__).resolve().parent.parent
# Un archivo (.csv, .csv.gz o .parquet) o una CARPETA: si es carpeta se leen y
# juntan todos los archivos de datos que haya dentro (p. ej. partes 1..N)
DATA_FILE = ROOT / "data"
OUT = ROOT / "out"
MODELS = ROOT / "models"

# Modo rapido para probar que todo corre (30% de los pitchers, menos arboles):
#   Windows:  set RAPIDO=1   ·   Mac/Linux:  export RAPIDO=1
RAPIDO = os.environ.get("RAPIDO") == "1"

# ── 0. CORRIDA DE ENTREGA ────────────────────────────────────────────────────
# ESTRICTO: corrida de entrega (set ESTRICTO=1): los avisos graves detienen el
# pipeline.
ESTRICTO = os.environ.get("ESTRICTO") == "1"

# ── 1. MAPEO DE COLUMNAS: nombre en el dataset ISAC -> nombre canonico ───────
COLUMN_MAP = {
    # identificadores (solo para agrupar / validar, NUNCA como variable)
    "PitchUID": "pitch_id", "game_anon_id": "game_id", "pitcher_anon_id": "pitcher_id",
    "batter_anon_id": "batter_id", "catcher_anon_id": "catcher_id", "year": "season",
    # contexto
    "PitcherThrows": "p_throws", "BatterSide": "stand", "altitude_category": "altitude_category",
    "Inning": "inning", "Top/Bottom": "half", "Outs": "outs", "Balls": "balls", "Strikes": "strikes",
    # tipo de pitcheo
    "AutoPitchType": "pitch_type",
    # FISICA (Stuff+)
    "RelSpeed": "velocity", "EffectiveVelo": "eff_velo", "SpinRate": "spin_rate",
    "SpinAxis": "spin_axis", "RelHeight": "release_height", "RelSide": "release_side",
    "Extension": "extension", "InducedVertBreak": "ivb", "HorzBreak": "hb",
    "VertRelAngle": "vra", "HorzRelAngle": "hra",
    # llegada (dependen del aire: se usan para la fisica, no para Stuff+)
    "ZoneSpeed": "plate_speed", "SpeedDrop": "speed_drop", "ZoneTime": "zone_time",
    "VertApprAngle": "vaa", "HorzApprAngle": "haa",
    # aceleracion vertical del ajuste de trayectoria: solo para el control de
    # sustentacion de physics.py (opcional)
    "az0": "az0",
    # ubicacion (Location+ / Pitching+, NO Stuff+)
    "PlateLocHeight": "plate_z", "PlateLocSide": "plate_x",
    # resultados (solo objetivos)
    "PitchCall": "pitch_call", "KorBB": "k_or_bb", "play_result": "play_result",
    "hit_type": "hit_type", "ExitSpeed": "exit_velo", "Angle": "launch_angle",
    "Direction": "direction", "Distance": "hit_distance",
    "RunsScored": "runs_scored", "OutsOnPlay": "outs_on_play",
    "single": "single", "double": "double", "triple": "triple", "home_run": "home_run",
    "base_on_balls": "base_on_balls", "strikeout": "strikeout", "is_hit_by_pitch": "is_hit_by_pitch",
    # banderas oficiales: van con nombre PROPIO (flag_*) para no chocar con las
    # etiquetas que derivamos de PitchCall; features.py las compara y avisa
    "is_swing": "flag_swing", "is_whiff": "flag_whiff", "is_called_strike": "flag_called_strike",
    "in_strike_zone": "flag_in_zone", "swung_outside_strike_zone": "flag_chase",
    # si algun dia viene un run value ya calculado, se usa; si no, se calcula
    "run_value": "run_value",
}
# Columnas del diccionario oficial que ningun paso usa: NO se cargan. Ocupaban
# memoria en todos los pasos y, en un archivo entregado en partes, un texto
# suelto en una de ellas tumbaba el guardado a parquet. ("count" se vuelve a
# armar con Balls y Strikes.)
COLUMNAS_SIN_USO = {
    "Tilt", "VertBreak", "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "pfxx", "pfxz",
    "is_contact", "is_swinging_strike", "is_ball_in_play", "is_batted", "is_hit",
    "outside_strike_zone", "count",
} | {f"PitchTrajectory{eje}c{k}" for eje in "XYZ" for k in range(3)}

# Minimo indispensable. Si falta alguna, el pipeline se detiene y dice cual.
REQUIRED = ["pitch_id", "pitcher_id", "season", "p_throws", "stand",
            "balls", "strikes", "pitch_type", "pitch_call",
            "velocity", "spin_rate", "ivb", "hb", "extension",
            "release_height", "release_side", "plate_x", "plate_z"]

# ── 2. ETIQUETAS DE RESULTADO ────────────────────────────────────────────────
# Las llaves se comparan en minusculas y sin espacios/guiones, asi que
# "Ground ball", "GroundBall" y "ground_ball" caen en la misma.
PITCH_CALL_MAP = {
    "ballcalled": "ball", "ballindirt": "ball", "strikecalled": "called_strike",
    "strikeswinging": "whiff", "foulball": "foul", "foulballnotfieldable": "foul",
    "foulballfieldable": "foul", "inplay": "in_play", "hitbypitch": "hbp",
}
# No son calidad del pitcheo (base intencional, bola automatica por reloj, y
# "Undefined": el archivo oficial trae 0.13% de pitcheos sin cantar): se
# descartan sin aviso y se cuentan en el mensaje de limpieza.
PITCH_CALL_DESCARTAR = {"ballintentional", "automaticball", "undefined"}
# Los toques ("bunt") se reconocen para sacarlos del modelado: no son un rodado.
HIT_TYPE_MAP = {"groundball": "gb", "groundballs": "gb", "gb": "gb",
                "linedrive": "ld", "ld": "ld", "flyball": "fb", "fb": "fb",
                "popup": "pu", "popfly": "pu", "pu": "pu", "bunt": "bunt"}
# play_result de un batazo -> evento con peso en carreras (seccion 4). El
# diccionario oficial NO enumera sus valores ("Dataset-specific result labels"):
# aqui van las formas usuales de escribir cada resultado. Un batazo con una
# etiqueta que no este aqui (y sin bandera de hit) se valua como out Y SE
# AVISA: un error escrito de otra forma pasaba por out sin decir nada.
# Sacrificio y jugada de seleccion van como out (ver la seccion 4).
PLAY_RESULT_MAP = {
    "single": "Single", "double": "Double", "triple": "Triple", "homerun": "HomeRun",
    "error": "Error", "reachedonerror": "Error", "fielderror": "Error", "fieldingerror": "Error",
    "out": "Out", "fieldout": "Out", "forceout": "Out", "doubleplay": "Out", "tripleplay": "Out",
    "groundedintodoubleplay": "Out", "fielderschoice": "Out", "fielderschoiceout": "Out",
    "sacrifice": "Out", "sacrificefly": "Out", "sacrificebunt": "Out", "sacfly": "Out",
    "sacbunt": "Out", "sacflydoubleplay": "Out", "sacbuntdoubleplay": "Out", "undefined": "Out",
}

# ── 3. ALTITUD ───────────────────────────────────────────────────────────────
# ISAC no trae estadio, trae altitude_category, con 3 etiquetas en el archivo
# oficial (stuff_model_df, 635,002 pitcheos, 2024-2026). Aqui: etiqueta -> metros
# representativos de esa categoria (promedio por juego de sus parques).
# Que parques caen en cada una no lo dice el diccionario: se dedujo de los juegos
# por temporada de cada categoria (47.4 / 24.0 / 28.5 % de los juegos del
# dataset, contra 45.4 / 25.1 / 29.5 % con este reparto en el calendario publico
# LMB 2024-2026, src/clima_lmb.py), y de su huella fisica, que no depende del reparto: la velocidad que
# pierde la bola de la mano al plato (7.04 / 6.29 / 5.96 mph) va con la densidad
# de abajo.
#   No Altitude:      parques de menos de 700 m (sureste, Tijuana, Laredo, Monterrey, Monclova)
#   Medium Altitude:  Torreon, Chihuahua, Saltillo, Oaxaca, Zapopan (1,100 a 1,700 m)
#   Extreme Altitude: Leon, Queretaro, Aguascalientes, Durango, Puebla y el Harp Helu (1,800 m o mas)
ALTITUDE_M = {
    "No Altitude":      158,
    "Medium Altitude":  1462,
    "Extreme Altitude": 1981,
}
HOME_ALTITUDE = "Extreme Altitude"    # la categoria del Harp Helu
HOME_PARK = HOME_ALTITUDE             # (alias de HOME_ALTITUDE)
HOME_STADIUM = "harp_helu"            # cuando el dataset trae el estadio exacto (clave de STADIUMS)

# Bandas para comparar "altitud vs nivel del mar" (features.banda_de):
#   con estadio exacto: alta = mas de BANDA_ALTA_M, mar = hasta BANDA_BAJA_M
#   con categorias:     alta = SOLO la categoria HOME_ALTITUDE,
#                       mar = categorias cuyos metros en ALTITUDE_M <= BANDA_BAJA_M,
#                       el resto queda en "media"
BANDA_ALTA_M, BANDA_BAJA_M = 1700, 700

# Elevacion de cada estadio en el modelo digital del terreno (Copernicus GLO-90).
# El pipeline solo la usa si el dataset trae una columna stadium con estas claves
# (el dataset oficial no la trae); de aqui salen tambien CDMX_M y las altitudes
# de parques que cita el dashboard.
STADIUMS = {
    "harp_helu": ("Alfredo Harp Helú · CDMX", 2236), "hermanos_serdan": ("Hermanos Serdán · Puebla", 2209),
    "francisco_villa": ("Francisco Villa · Durango", 1883), "romo_chavez": ("Romo Chávez · Aguascalientes", 1880),
    "conspiradores": ("Conspiradores · Querétaro", 1821), "domingo_santana": ("Domingo Santana · León", 1801),
    "francisco_madero": ("Francisco I. Madero · Saltillo", 1552), "panamericano": ("Panamericano · Zapopan", 1665),
    "eduardo_vasconcelos": ("Eduardo Vasconcelos · Oaxaca", 1573),
    "monumental_chihuahua": ("Monumental · Chihuahua", 1408), "revolucion": ("Revolución · Torreón", 1129),
    "monclova": ("Monclova", 612), "monterrey": ("Estadio de Monterrey", 518),
    "la_junta": ("Parque La Junta · Nuevo Laredo", 127), "unitrade": ("UniTrade · Laredo TX", 140),
    "chevron": ("Chevron · Tijuana", 39), "beto_avila_cancun": ("Beto Ávila · Cancún", 4),
    "beto_avila_veracruz": ("Universitario Beto Ávila · Veracruz", 11),
    "nelson_barrera": ("Nelson Barrera · Campeche", 9), "centenario_27": ("Centenario 27 · Villahermosa", 19),
    "kukulcan": ("Kukulcán · Mérida", 11), "kanasin": ("Kanasín · Yucatán", 11),
}
SEA_LEVEL_M = 0
CDMX_M = STADIUMS[HOME_STADIUM][1]    # 2,236 m: una sola fuente para simulador, dashboard y documento

# ── 3b. AIRE REAL DE LA LMB (de donde salen las densidades de abajo) ─────────
# Calculado una vez con src/clima_lmb.py, solo con fuentes publicas: los juegos
# LMB 2024-2026 (estadio, fecha y hora) salen del calendario PUBLICO de la liga
# en MLB Stats API (sportId 23), y la temperatura, la humedad y la presion a la
# hora de cada juego, del REANALISIS ERA5 (Open-Meteo) en las coordenadas del
# estadio. Densidad relativa = rho / 1.225 kg/m3 (aire estandar a nivel del mar).
#
# El dataset oficial no trae estadio ni fecha del juego: cada pitcheo usa la
# densidad PROMEDIO de su categoria de altitud (ALTITUDE_RHO).
ALTITUDE_RHO = {"No Altitude": 0.9293, "Medium Altitude": 0.8099, "Extreme Altitude": 0.7724}
CLIMA_FACTOR = 0.9395  # densidad real / atmosfera estandar, promedio de los juegos (noches templadas-calientes)
RHO_CDMX = 0.7530      # Harp Helu, promedio a la hora del juego (158 juegos; 22.8 C, 50% HR, 788.5 hPa)
RHO_MAR = 0.9435       # parques LMB de menos de 200 m, promedio (1,024 juegos; 27.8 C, 73% HR)
RHO_REF = 1.0          # densidad de referencia interna para guardar la recta del pitcher
CLIMA_TIPICO = {"cdmx": (22.8, 50.2), "mar": (27.8, 73.1)}   # (temp C, humedad %) para el simulador

# ── 4. VALOR EN CARRERAS (ISAC no trae run value: se calcula) ────────────────
# Se estima con el mismo dataset (features.estimar_valores_carrera, guarda
# out/valores_carrera.json): pesos lineales con la expectativa de carreras
# (RE24) de una cadena de Markov de bases y outs calibrada con las medias
# entradas completas del dataset, y el valor de cada cuenta por la cadena de
# cuentas de la liga. Positivo = bateador.
# Valor de un pitcheo = valor(cuenta despues) - valor(cuenta antes), o
# peso(evento) - valor(cuenta antes) si el pitcheo termina el turno.
# El sacrificio y la jugada de seleccion se valuan como out (ver PLAY_RESULT_MAP).

# ── 4b. EQUIPO (aparece en la portada del documento tecnico) ────────────────
EQUIPO = "Trez"
INTEGRANTES = ["Luis Ángel Córdova Gil", "José Ángel López Mercado"]
INSTITUCION = "Instituto Politécnico Nacional (ESCOM)"

# ── 5. PARAMETROS DE MODELADO ────────────────────────────────────────────────
RANDOM_STATE = 2026
BACKEND = "sklearn"              # "sklearn" (probado) | "lightgbm" (opcional, hay que instalarlo)
ITERS_MAX = 80 if RAPIDO else 400   # tope de arboles; el numero final se elige con validacion agrupada por pitcher
N_FOLDS = 3 if RAPIDO else 5        # cross-fit por pitcher
MAX_FILAS_FOLD = 300_000         # filas de entrenamiento por fold del cross-fit (tiempo de corrida)
BETA_LIMITES = (0.6, 2.0)        # limites de cordura para una elasticidad usada en el contrafactual
MIN_PITCHERS_BETA = 30           # pitchers minimos para usar la elasticidad propia de un tipo de pitcheo
MIN_PITCHES_PITCHER = 150   # para reportar Stuff+ de un pitcher
MIN_PITCHES_ARSENAL = 50    # para reportar un pitch type de un pitcher
PLUS_SCALE = 10.0           # Stuff+ = 100 + 10 * z
N_THREADS = -1              # en tu PC: todos los nucleos


def norm_label(x):
    """'Ground ball' / 'GroundBall' / 'ground_ball' -> 'groundball'."""
    return "".join(ch for ch in str(x).lower() if ch.isalnum())


def comando(paso):
    """El comando que vuelve a correr la tuberia desde `paso`, escrito para la carpeta del proyecto. Los mensajes
    de error lo citan: un archivo suelto (python score.py) no corre desde ahi ni queda en out/run_info.json."""
    return f"python run_all.py {'--rapido ' if RAPIDO else ''}{'--estricto ' if ESTRICTO else ''}--desde {paso}"


# Las etiquetas del dataset se comparan ya normalizadas (norm_label). Aqui se
# normalizan tambien las LLAVES de los mapas, para que una etiqueta agregada tal
# como la imprime el aviso ("Fly Out", "ReachedOnError") coincida; y un valor de
# PLAY_RESULT_MAP que no sea un evento conocido se dice de una vez (si no, el
# batazo se valuaba como out sin ningun aviso). Lo mismo con los valores de los
# otros dos mapas: antes el error salia hasta el paso de modelos.
PITCH_CALL_MAP = {norm_label(k): v for k, v in PITCH_CALL_MAP.items()}
PITCH_CALL_DESCARTAR = {norm_label(k) for k in PITCH_CALL_DESCARTAR}
HIT_TYPE_MAP = {norm_label(k): v for k, v in HIT_TYPE_MAP.items()}
PLAY_RESULT_MAP = {norm_label(k): v for k, v in PLAY_RESULT_MAP.items()}
EVENTOS_BATAZO = ("Single", "Double", "Triple", "HomeRun", "Error", "Out")
for _nombre, _mapa, _validos in (
        ("PITCH_CALL_MAP", PITCH_CALL_MAP, ("ball", "called_strike", "whiff", "foul", "in_play", "hbp")),
        ("HIT_TYPE_MAP", HIT_TYPE_MAP, ("gb", "ld", "fb", "pu", "bunt")),
        ("PLAY_RESULT_MAP", PLAY_RESULT_MAP, EVENTOS_BATAZO)):
    _malos = {k: v for k, v in _mapa.items() if v not in _validos}
    if _malos:
        raise SystemExit(f"config.{_nombre}: valores no validos {_malos}. Cada etiqueta debe ir a uno de "
                         f"{_validos}, escrito tal cual (con sus mayusculas y minusculas).")
