"""
FEATURES — de pitcheos crudos a la matriz de modelado.

Cuatro familias, igual que pide el reto:
  1. FISICA en el release: velocidad, IVB, HB, spin, punto de salida, extension
     (+ derivadas: movimiento total y movimiento por cada 1000 rpm)
  2. ARSENAL: diferenciales contra la recta primaria del pitcher (medida en el
     MISMO aire que el pitcheo), tunel, uso del pitcheo, tamano del repertorio
  3. UBICACION Y CONTEXTO: coordenadas en el plato, zona, cuenta, lado del
     bateador y outs. El dataset de ISAC no trae orden de pitcheos ni
     corredores: no hay variables de secuencia ni estados base-out.
  4. AMBIENTE: densidad relativa del aire y banda de altitud. NO son variables
     de ningun modelo: sirven para llevar el movimiento de un aire a otro
     (transportar) y para valuar los batazos.

Todo en marco "diestro": lo horizontal de los zurdos se espejea (HB, lado de
salida, plato, angulos y eje) para que un slider de zurdo y uno de derecho se
comparen en la misma escala. En ese marco, positivo = lado del brazo.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# El interruptor ESTRICTO y las rutas (OUT, DATA_FILE, ...) se leen como
# config.X en el momento de usarse, para que una prueba pueda cambiarlos.
import config
from config import (COLUMN_MAP, COLUMNAS_SIN_USO, REQUIRED, PITCH_CALL_MAP, PITCH_CALL_DESCARTAR,
                    HIT_TYPE_MAP, PLAY_RESULT_MAP,
                    STADIUMS, ALTITUDE_M, ALTITUDE_RHO, CLIMA_FACTOR, HOME_ALTITUDE, HOME_STADIUM,
                    BANDA_ALTA_M, BANDA_BAJA_M, RHO_REF,
                    RAPIDO, RANDOM_STATE, norm_label)

# ── tipos de pitcheo -> familias estandar ────────────────────────────────────
PT_MAP = {
    "Four-Seam": "FF", "Fastball": "FF", "FourSeamFastBall": "FF",
    "Sinker": "SI", "TwoSeamFastBall": "SI", "Cutter": "FC",
    "Slider": "SL", "Sweeper": "SL", "Curveball": "CU", "Knuckle Curve": "CU",
    "Changeup": "CH", "ChangeUp": "CH", "Splitter": "FS", "Knuckleball": "KN",
}
PT_MAP = {norm_label(k): v for k, v in PT_MAP.items()}   # tolerante a mayusculas/guiones
# "Other" es un valor del diccionario oficial (la etiqueta automatica del dataset,
# AutoPitchType, no clasifico el pitcheo) y "Undefined" otra forma de escribir un
# pitcheo sin clasificar: se descartan y se cuentan, pero no son etiquetas
# desconocidas ni piden agregarse a PT_MAP.
PT_DESCARTAR = {"other", "undefined"}
TIPOS = ("FF", "SI", "FC", "SL", "CU", "CH", "FS")
FASTBALLS = ("FF", "SI", "FC")
# Un tipo cuenta en el repertorio si es >= 5% de los pitcheos del pitcher: la
# etiqueta automatica (AutoPitchType) clasifica mal pitcheos sueltos y, contando
# toda etiqueta, el repertorio de un pitcher sale inflado.
USO_MIN_REPERTORIO = 0.05

# Clases de resultado por pitcheo (mutuamente excluyentes). El valor de cada
# clase en carreras se estima con el mismo dataset (model.outcome_values).
OUTCOMES = ["ball", "called_strike", "whiff", "foul", "hbp",
            "ip_gb", "ip_ld", "ip_fb_soft", "ip_fb_hard", "ip_pu"]


def _ascii(x):
    """Texto seguro para la consola de Windows (las etiquetas del dataset pueden traer acentos)."""
    return str(x).encode("ascii", "replace").decode()


def _aviso_grave(*lineas):
    """Aviso que no debe pasar desapercibido; con ESTRICTO=1 detiene la corrida."""
    print("\n" + "!" * 78)
    for linea in lineas:
        print("AVISO: " + linea)
    print("!" * 78 + "\n")
    if config.ESTRICTO:
        raise SystemExit("ESTRICTO=1: corrida detenida por el aviso anterior.")


def _sin_reconocer(df, m, base, col, que, arreglo, limite, que_pasa="se descartan"):
    """Etiquetas del dataset que el pipeline no entiende: se listan, se dice
    `que_pasa` con sus filas (quien llama es quien las descarta o las deja) y,
    si pasan de `limite` % de `base`, el aviso es grave."""
    if not m.any():
        return
    pct = 100 * m.sum() / base.sum()
    print(f"{que} sin reconocer: {int(m.sum()):,} ({pct:.2f}%) -> {que_pasa}. Etiquetas:",
          _ascii(df.loc[m, col].value_counts(dropna=False).head(8).to_dict()) if col in df else "(sin columna)")
    if pct > limite:
        _aviso_grave(f"{que}: {pct:.1f}% trae una etiqueta que no se reconoce ({que_pasa}).", arreglo)


# ── aire ─────────────────────────────────────────────────────────────────────
def air_density_ratio(alt_m):
    """Densidad del aire relativa a nivel del mar (atmosfera estandar ICAO).
    2,236 m (Harp Helu) -> 0.80, es decir ~20% menos aire que en el mar."""
    alt_m = np.asarray(alt_m, dtype=float)
    return (1.0 - 2.25577e-5 * alt_m) ** 4.2559


def densidad_aire(alt_m, temp_c=None, hr_pct=None):
    """Densidad relativa (1 = 1.225 kg/m3). Con temperatura y humedad usa la
    formula del aire humedo con la presion de esa altitud; sin ellas, la
    atmosfera estandar corregida por el clima tipico de los juegos LMB."""
    alt_m = np.asarray(alt_m, dtype=float)
    if temp_c is None:
        return air_density_ratio(alt_m) * CLIMA_FACTOR
    p = 1013.25 * (1.0 - 2.25577e-5 * alt_m) ** 5.25588                  # hPa, estandar
    # la presion REAL en temporada LMB es mayor que la estandar en altura:
    # correccion lineal con la altitud (+1.87% a 2,240 m), ajustada a la
    # presion media del reanalisis ERA5 a la hora de los juegos del calendario
    # publico de la LMB (src/clima_lmb.py)
    p = p * (1.0 + 0.0187 * alt_m / 2240.0)
    t = np.asarray(temp_c, float)
    pv = np.asarray(50.0 if hr_pct is None else hr_pct, float) / 100 * 6.1078 * 10 ** (7.5 * t / (t + 237.3))
    T = t + 273.15
    return ((p - pv) * 100 / (287.05 * T) + pv * 100 / (461.495 * T)) / 1.225


def modo_estadio(df):
    """True si df['stadium'] trae el estadio exacto; False si es la categoria de
    altitud haciendo de estadio (el caso del dataset oficial)."""
    return "stadium" in df and df["stadium"].isin(list(STADIUMS)).mean() > .5


def banda_de(df):
    """Banda de altitud de cada pitcheo: 'alta', 'media' o 'mar'.
    Con estadio exacto se corta por metros (BANDA_ALTA_M / BANDA_BAJA_M). Con
    categorias, 'alta' es SOLO la categoria del Harp Helu (HOME_ALTITUDE), 'mar'
    las categorias de hasta BANDA_BAJA_M metros en ALTITUDE_M y el resto 'media'."""
    if modo_estadio(df):
        alt = df["altitude_m"].to_numpy(float)
        b = np.where(alt > BANDA_ALTA_M, "alta", np.where(alt <= BANDA_BAJA_M, "mar", "media"))
    else:
        cat = df["altitude_category"].astype(str)
        al_mar = [k for k, m in ALTITUDE_M.items() if m <= BANDA_BAJA_M and k != HOME_ALTITUDE]
        b = np.where(cat == HOME_ALTITUDE, "alta", np.where(cat.isin(al_mar), "mar", "media"))
    return pd.Series(b, index=df.index)


# ── carga ────────────────────────────────────────────────────────────────────
EXTENSIONES = (".csv", ".csv.gz", ".parquet")


def _leer_archivo(p):
    """Un archivo de datos, sin las columnas que ningun paso usa (config.COLUMNAS_SIN_USO)."""
    if p.name.lower().endswith(".parquet"):
        d = pd.read_parquet(p)
        d = d.drop(columns=[c for c in d.columns if c in COLUMNAS_SIN_USO])
        # un parquet puede traer columnas categoricas (category de pandas, factor de R):
        # se pasan a valores sueltos, porque .map sobre una categorica devuelve otra
        # categorica y "ip_" + categorica truena al armar la clase de resultado
        for c in [c for c in d.columns if isinstance(d[c].dtype, pd.CategoricalDtype)]:
            d[c] = d[c].astype(object)
        return d
    try:
        return pd.read_csv(p, low_memory=False, usecols=lambda c: c not in COLUMNAS_SIN_USO)
    except UnicodeDecodeError:
        # csv guardado desde Excel en Windows (no es UTF-8): se lee con la codificacion de Windows
        print(f"AVISO: {_ascii(p.name)} no esta en UTF-8: se lee como latin-1 (el csv que guarda Excel en Windows)")
        return pd.read_csv(p, low_memory=False, usecols=lambda c: c not in COLUMNAS_SIN_USO, encoding="latin-1")


def load_raw(path=None):
    path = Path(path or config.DATA_FILE)
    if not path.exists():
        raise SystemExit(f"No existe {path}: pon ahi el archivo de datos (csv, csv.gz o parquet)")
    archivos = (sorted(p for p in path.iterdir() if p.name.lower().endswith(EXTENSIONES))
                if path.is_dir() else [path])
    if not archivos:
        hay = sorted(p.name for p in path.iterdir())
        raise SystemExit(f"No hay archivos de datos ({', '.join(EXTENSIONES)}) en {path}. "
                         + (f"Ahi hay: {_ascii(hay[:10])}. Si el dataset viene comprimido (.zip) o en Excel, "
                            "descomprimelo o guardalo como csv." if hay else "La carpeta esta vacia."))
    df = pd.concat([_leer_archivo(p) for p in archivos], ignore_index=True)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    faltan = [c for c in REQUIRED if c not in df.columns]
    if faltan:
        oficial = {v: k for k, v in COLUMN_MAP.items()}
        otras = [c for c in df.columns if c not in oficial]
        raise SystemExit(f"Faltan columnas obligatorias: {faltan} (en el diccionario oficial: "
                         f"{[oficial.get(c, c) for c in faltan]}). Columnas del archivo que no se "
                         f"reconocieron: {_ascii(otras[:25])}. Si alguna es una de las que faltan con otro "
                         "nombre, agregala a COLUMN_MAP en config.py")
    if "stadium" not in df and "altitude_category" not in df:
        raise SystemExit("El dataset no trae ni estadio ni altitude_category: no hay como medir la altitud")
    dup = df["pitch_id"].notna() & df["pitch_id"].duplicated()
    if dup.mean() > 0.01:
        raise SystemExit(f"{int(dup.sum()):,} filas ({100 * dup.mean():.1f}%) repiten un PitchUID: parece que "
                         f"el archivo se junto dos veces. Deja en {path} una sola copia de cada parte.")
    if dup.any():
        df = df[~dup].reset_index(drop=True)
        print(f"Duplicados: se quitaron {int(dup.sum()):,} filas con PitchUID repetido (queda la primera)")
    if RAPIDO:
        # el valor en carreras se estima ANTES del recorte: quitar pitchers rompe
        # las medias entradas con cambio de pitcher y sesga las carreras
        estimar_valores_carrera(df)
        # prueba rapida: 30% de los PITCHERS (no de los pitcheos, para no
        # romper arsenales ni la validacion por pitcher)
        ids = pd.Series(df["pitcher_id"].dropna().unique())
        keep = set(ids.sample(frac=0.3, random_state=RANDOM_STATE))
        df = df[df["pitcher_id"].isin(keep)].reset_index(drop=True)
        print(f"MODO RAPIDO: {len(df):,} pitcheos de {len(keep):,} pitchers")
    return df


def _norm(s):
    """norm_label sobre una columna (por valores unicos: rapido en tablas grandes)."""
    return s.map({v: norm_label(v) for v in s.dropna().unique()})


# Textos que cuentan como bandera encendida / apagada (en minusculas). El
# diccionario dice "0/1 or TRUE/FALSE"; el resto son variantes de Excel.
BANDERA_SI = ("1", "1.0", "true", "t", "yes", "y", "si", "verdadero")
BANDERA_NO = ("0", "0.0", "false", "f", "no", "n", "falso")
BANDERAS = ("flag_swing", "flag_whiff", "flag_called_strike", "flag_in_zone", "flag_chase",
            "single", "double", "triple", "home_run")


def _bandera(df, col):
    """Bandera del dataset como 1.0 / 0.0 (acepta 0/1, TRUE/FALSE y variantes);
    NaN si no viene o si el texto no se entiende (ver _banderas_ilegibles)."""
    if col not in df:
        return pd.Series(np.nan, index=df.index)
    txt = df[col].astype(str).str.strip().str.lower()
    si, no = (txt.isin([t.strip().lower() for t in lista]) for lista in (BANDERA_SI, BANDERA_NO))
    return pd.Series(np.where(si, 1.0, np.where(no, 0.0, np.nan)), index=df.index)


def _flag(df, col):
    return _bandera(df, col) == 1


def _banderas_ilegibles(df):
    """Banderas que SI vienen en el archivo pero con textos que no son 0/1 ni
    TRUE/FALSE. Sin este aviso se daban por ausentes sin decir nada: la zona
    caia a la caja fija y los hits se quedaban sin bandera."""
    oficial = {v: k for k, v in COLUMN_MAP.items()}
    for col in BANDERAS:
        if col in df:
            viene = df[col].notna()
            _sin_reconocer(df, viene & _bandera(df, col).isna(), viene, col, f"Bandera {oficial.get(col, col)}",
                           "Las banderas deben venir como 0/1 o TRUE/FALSE; agrega el texto que traen a "
                           "BANDERA_SI / BANDERA_NO (features.py) y vuelve a correr.", 1,
                           "se toman como dato faltante")


def es_jonron(df):
    """True en los pitcheos que acabaron en jonron: bandera home_run o etiqueta
    de play_result (config.PLAY_RESULT_MAP). Todo False si no viene ninguna."""
    hr = _flag(df, "home_run")
    if "play_result" in df:
        hr = hr | (_norm(df["play_result"]).map(PLAY_RESULT_MAP) == "HomeRun")
    return hr


# ── valor en carreras, estimado con el propio dataset ───────────────────────
# El dataset no trae run value. Los pesos de cada evento (pesos lineales, Tango
# et al. 2007) salen de la expectativa de carreras de una cadena de Markov de
# bases y outs calibrada con el dataset; el valor de cada cuenta sale de la
# cadena de cuentas de la liga con esos pesos.
EVENTOS = ("Single", "Double", "Triple", "HomeRun", "Walk", "HitByPitch", "Error", "Out", "Strikeout")
VALORES_CARRERA = {}      # lo llena estimar_valores_carrera; build() lo guarda en out/valores_carrera.json
COLUMNAS_VALOR = ("game_id", "inning", "half", "runs_scored", "outs_on_play")


def _eventos_de_turno(df, call, b, s):
    """Evento con que acaba el turno en cada pitcheo ("" si el pitcheo no lo acaba): ponche (KorBB o strike con
    2 strikes), base por bolas (KorBB o bola con 3 bolas, incluye la intencional), golpe, y en un batazo el
    evento de las banderas de hit o de play_result; un batazo sin hit ni error vale como out (sacrificio y
    jugada de seleccion incluidos, como en run_value_por_cuenta)."""
    kbb = _norm(df["k_or_bb"]) if "k_or_bb" in df else pd.Series("", index=df.index, dtype=object)
    pr = _norm(df["play_result"]).map(PLAY_RESULT_MAP) if "play_result" in df else pd.Series(np.nan, index=df.index)
    ev = pd.Series("", index=df.index, dtype=object)
    ip = call == "in_play"
    ev[ip] = "Out"
    for flag, nombre in (("single", "Single"), ("double", "Double"), ("triple", "Triple"), ("home_run", "HomeRun")):
        ev[ip & (_flag(df, flag) | (pr == nombre))] = nombre
    ev[ip & (pr == "Error")] = "Error"
    ev[call == "hbp"] = "HitByPitch"
    ev[(kbb == "walk") | ((call == "ball") & (b == 3))] = "Walk"
    ev[(kbb == "strikeout") | (call.isin(["called_strike", "whiff"]) & (s == 2))] = "Strikeout"
    return ev


# ── cadena de Markov de bases y outs (24 estados) calibrada con el dataset ──
# Estado = outs * 8 + bases (bit 1 = primera, 2 = segunda, 4 = tercera). Como el dataset no trae corredores, el
# avance de los corredores se modela con 6 probabilidades (THETA) que se ajustan para que la cadena reproduzca lo que
# si trae: cuantas carreras anota en promedio cada tipo de jugada, la tasa de doble play y las carreras por media
# entrada. Con la cadena ajustada sale la expectativa de carreras de cada estado (RE24) y, de ahi, los pesos.
THETA = ("anota_de_2a_con_sencillo", "de_1a_a_3a_con_sencillo", "anota_de_1a_con_doble",
         "anota_de_3a_con_out", "doble_play", "avanza_con_out")
THETA0 = np.array([.6, .25, .4, .5, .12, .25])


def _forzado(bases):
    """Base por bolas / golpe: solo avanzan los corredores forzados."""
    if not bases & 1:
        return bases | 1, 0
    if not bases & 2:
        return bases | 3, 0
    if not bases & 4:
        return 7, 0
    return 7, 1


def _transiciones(e, st, th):
    """Lista de (probabilidad, estado nuevo o None si acaba la media entrada, carreras) del evento e en st."""
    s2, s1, d1, sf, dp, av = th
    o, bs = divmod(st, 8)
    r1, r2, r3 = bs & 1, (bs >> 1) & 1, (bs >> 2) & 1
    sal = []
    def poner(p, outs, bases, carreras):
        if p > 0:
            sal.append((p, None if outs >= 3 else outs * 8 + bases, 0 if outs >= 3 else carreras))
    if e == "Strikeout":
        poner(1.0, o + 1, bs, 0)
    elif e in ("Walk", "HitByPitch"):
        nb, c = _forzado(bs); poner(1.0, o, nb, c)
    elif e in ("Single", "Error"):
        for a2, pa2 in (((1, s2), (0, 1 - s2)) if r2 else ((0, 1.0),)):          # a2 = el de 2a anota
            tercera = r2 and not a2
            for a1, pa1 in (((1, s1), (0, 1 - s1)) if (r1 and not tercera) else ((0, 1.0),)):   # a1 = el de 1a llega a 3a
                nb = 1 | (2 if (r1 and not a1) else 0) | (4 if (tercera or a1) else 0)
                poner(pa2 * pa1, o, nb, r3 + a2)
    elif e == "Double":
        for a1, pa1 in (((1, d1), (0, 1 - d1)) if r1 else ((0, 1.0),)):
            poner(pa1, o, 2 | (4 if (r1 and not a1) else 0), r3 + r2 + a1)
    elif e == "Triple":
        poner(1.0, o, 4, r1 + r2 + r3)
    elif e == "HomeRun":
        poner(1.0, o, 0, r1 + r2 + r3 + 1)
    elif e == "Out":
        if o == 2:
            poner(1.0, 3, 0, 0)
        else:
            pdp = dp if r1 else 0.0
            if pdp:
                poner(pdp, o + 2, bs & ~1, 0)                                      # sale el de 1a y el bateador
            for ar3, p3 in (((1, sf), (0, 1 - sf)) if r3 else ((0, 1.0),)):
                for av_, pa in (((1, av), (0, 1 - av)) if (r1 or r2) else ((0, 1.0),)):
                    nb = (bs & ~4) if ar3 else bs
                    if av_:                                                          # avanzan una base si hay lugar
                        if (nb & 2) and not (nb & 4): nb = (nb & ~2) | 4
                        if (nb & 1) and not (nb & 2): nb = (nb & ~1) | 2
                    poner((1 - pdp) * p3 * pa, o + 1, nb, ar3)
    return sal


def _cadena(p, th):
    """RE de cada estado, visitas esperadas por media entrada y, por evento, lo que vale y cuantas carreras anota."""
    Q, r = np.zeros((24, 24)), np.zeros(24)
    T = {e: [_transiciones(e, st, th) for st in range(24)] for e in p}
    for e, pe in p.items():
        for st in range(24):
            for q, nuevo, c in T[e][st]:
                r[st] += pe * q * c
                if nuevo is not None:
                    Q[st, nuevo] += pe * q
    M = np.linalg.inv(np.eye(24) - Q)
    RE, visitas = M @ r, M[0]
    peso, carreras = {}, {}
    for e in p:
        v = c_ = 0.0
        for st in range(24):
            for q, nuevo, c in T[e][st]:
                v += visitas[st] * q * (c + (RE[nuevo] if nuevo is not None else 0.0) - RE[st])
                c_ += visitas[st] * q * c
        peso[e], carreras[e] = v / visitas.sum(), c_ / visitas.sum()
    return RE, visitas, peso, carreras


def ajustar_cadena(p, objetivo):
    """objetivo: carreras promedio por jugada de cada evento, tasa de doble play entre los outs en juego y carreras
    por media entrada. Devuelve (theta, RE, pesos, ajuste)."""
    from scipy.optimize import least_squares
    # el error no entra: sus transiciones son las del sencillo y es raro (le daba al ajuste tanto peso como el
    # sencillo). Cada evento pesa segun su frecuencia (raiz de p / p del mas frecuente).
    claves = [e for e in ("Single", "Double", "Triple", "HomeRun", "Walk", "Out") if e in objetivo["carreras"]]
    pmax = max(p.get(e, 0.0) for e in claves)
    peso_res = {e: np.sqrt(p.get(e, 0.0) / pmax) for e in claves}
    def desvios(th):
        RE, vis, peso, car = _cadena(p, th)
        d = {e: (car[e] - objetivo["carreras"][e]) / max(.05, objetivo["carreras"][e]) for e in claves}
        d["doble_play"] = (_tasa_dp(p, th, vis) - objetivo["dp"]) / max(.02, objetivo["dp"])
        d["por_media_entrada"] = (RE[0] - objetivo["por_media_entrada"]) / objetivo["por_media_entrada"]
        return d
    def res(th):
        d = desvios(th)
        return np.array([peso_res.get(k, 1.0) * v for k, v in d.items()])
    sol = least_squares(res, THETA0, bounds=(0, 1), xtol=1e-10, ftol=1e-10)
    RE, vis, peso, car = _cadena(p, sol.x)
    d = desvios(sol.x)
    frec = [k for k in d if k != "Triple"]
    return sol.x, RE, peso, {"carreras_modelo": car, "dp_modelo": _tasa_dp(p, sol.x, vis), "re00": float(RE[0]),
                             "desvios": {k: float(v) for k, v in d.items()},
                             "residuo_max": float(np.abs(res(sol.x)).max()),
                             "desvio_max_frecuentes": float(max(abs(d[k]) for k in frec))}


def _tasa_dp(p, th, vis):
    """Fraccion de los outs en juego que son doble play, segun la cadena."""
    dp = th[4]
    con_1a = sum(vis[st] for st in range(16) if (st % 8) & 1)
    return dp * con_1a / max(1e-12, vis.sum())


def _momentos(n, r, n_dp, n_me, r_tot):
    """Frecuencia de cada evento por turno y lo que la cadena debe reproducir (sumas de juegos)."""
    p = {e: float(x) for e, x in zip(EVENTOS, n / n.sum()) if x > 0}
    i = {e: k for k, e in enumerate(EVENTOS)}
    obj = {"carreras": {e: float(r[i[e]] / n[i[e]]) for e in ("Single", "Double", "Triple", "HomeRun", "Walk", "Error", "Out")
                        if n[i[e]] > 0},
           "dp": float(n_dp / max(1, n[i["Out"]])), "por_media_entrada": float(r_tot / max(1, n_me))}
    return p, obj


def estimar_valores_carrera(raw):
    """Pesos lineales y valor de cada cuenta con el propio dataset (se llama con la tabla cruda, antes de la
    limpieza, para que cada media entrada conserve todos sus turnos).
      1. Media entrada = juego + entrada + mitad. Se usan las completas (3 outs: OutsOnPlay mas los ponches, que
         OutsOnPlay no cuenta) de las entradas 1 a 9 (en extra inning el corredor automatico pone carreras sin
         evento).
      2. Cadena de Markov de bases y outs (24 estados): la frecuencia de cada evento por turno sale del dataset;
         como el dataset no trae corredores, su avance (THETA) se ajusta para que la cadena reproduzca las carreras
         que anota en promedio cada tipo de jugada, la tasa de doble play y las carreras por media entrada.
      3. Peso de cada evento = lo que cambia la expectativa de carreras (RE24) mas lo que anota, promediado sobre
         los estados en que ocurre (positivo = bateador; el promedio por turno es 0). IC 95% por remuestreo de juegos.
      4. Valor de cada cuenta: lo que vale en promedio el turno desde esa cuenta, por recursion hacia atras con
         la frecuencia de cada resultado en cada cuenta de la liga (un foul con 2 strikes deja la cuenta igual).
    Contraste: una regresion de las carreras de cada media entrada sobre sus eventos (no ve corredores y por eso
    da valores menores) se guarda en out/valores_carrera.json."""
    faltan = [c for c in COLUMNAS_VALOR if c not in raw]
    if faltan:
        oficial = {v: k for k, v in COLUMN_MAP.items()}
        raise SystemExit(f"Para calcular el valor en carreras hacen falta {[oficial.get(c, c) for c in faltan]} "
                         "(el dataset no trae run value y se estima con sus propias medias entradas).")
    crudo = _norm(raw["pitch_call"])
    call = crudo.map(PITCH_CALL_MAP)
    # la bola automatica (reloj) mueve la cuenta; la intencional solo cuenta para la base por bolas del turno
    call = call.where(crudo != "automaticball", "ball")
    call_ev = call.where(crudo != "ballintentional", "ball")
    b = pd.to_numeric(raw["balls"], errors="coerce").clip(0, 3)
    s = pd.to_numeric(raw["strikes"], errors="coerce").clip(0, 2)
    ev = _eventos_de_turno(raw, call_ev, b, s)
    oop = pd.to_numeric(raw["outs_on_play"], errors="coerce").fillna(0).clip(0, 3)
    runs = pd.to_numeric(raw["runs_scored"], errors="coerce").fillna(0).clip(0, 4)
    inning = pd.to_numeric(raw["inning"], errors="coerce")
    juego = raw["game_id"].astype(str)
    clave = juego + "|" + inning.astype(str) + "|" + raw["half"].astype(str)
    outs_me = (oop + ((ev == "Strikeout") & (oop == 0)).astype(int)).groupby(clave).transform("sum")
    completa = (outs_me == 3) & inning.between(1, 9)
    fin_ = completa & (ev != "")
    n_me = int(clave[completa].nunique())
    if n_me < 2000:
        raise SystemExit(f"Solo {n_me:,} medias entradas completas: no alcanzan para estimar el valor en carreras. "
                         "Revisa que el dataset traiga game_anon_id, Inning, Top/Bottom, RunsScored y OutsOnPlay.")
    # sumas por juego (para el remuestreo)
    t = pd.DataFrame({"g": juego[fin_], "e": ev[fin_], "r": runs[fin_], "dp": ((ev == "Out") & (oop >= 2))[fin_]})
    Gn = pd.crosstab(t["g"], t["e"]).reindex(columns=list(EVENTOS), fill_value=0)
    Gr = t.pivot_table(index="g", columns="e", values="r", aggfunc="sum").reindex(index=Gn.index, columns=list(EVENTOS)).fillna(0)
    Gdp = t.groupby("g")["dp"].sum().reindex(Gn.index).fillna(0)
    Gme = clave[completa].groupby(juego[completa]).nunique().reindex(Gn.index).fillna(0)
    Gn, Gr, Gdp, Gme = Gn.to_numpy(float), Gr.to_numpy(float), Gdp.to_numpy(float), Gme.to_numpy(float)
    def ajuste(idx):
        p, obj = _momentos(Gn[idx].sum(0), Gr[idx].sum(0), Gdp[idx].sum(), Gme[idx].sum(), Gr[idx].sum())
        return p, obj, ajustar_cadena(p, obj)
    todo = np.arange(len(Gn))
    p, obj, (th, RE, pw, aj) = ajuste(todo)
    W = {e: float(pw.get(e, 0.0)) for e in EVENTOS}
    rng, boot = np.random.default_rng(2026), []
    for _ in range(100):
        pb = ajuste(rng.integers(0, len(Gn), len(Gn)))[2][2]
        boot.append([pb.get(e, np.nan) for e in EVENTOS])
    lo, hi = np.nanpercentile(np.array(boot), [2.5, 97.5], axis=0)
    # contraste: regresion sin constante de las carreras de cada media entrada sobre sus eventos
    me = pd.DataFrame({"k": clave[completa], "r": runs[completa]})
    for e in EVENTOS:
        me[e] = (ev[completa] == e).astype(int)
    me = me.groupby("k").sum()
    X, y = me[list(EVENTOS)].to_numpy(float), me["r"].to_numpy(float)
    br = np.linalg.lstsq(X, y, rcond=None)[0]
    pi = np.array([p.get(e, 0.0) for e in EVENTOS])
    reg = {e: float(v) for e, v in zip(EVENTOS, br - pi @ br)}
    # valor de cada cuenta por la cadena de cuentas
    ok = call.notna() & b.notna() & s.notna() & (crudo != "ballintentional")
    cat = pd.Series(np.where(call == "in_play", "ip_" + ev, call), index=raw.index)
    tabla = pd.crosstab([b[ok].astype(int), s[ok].astype(int)], cat[ok])
    V = np.zeros((5, 4))
    for ss in (2, 1, 0):
        for bb in (3, 2, 1, 0):
            f = tabla.loc[(bb, ss)] if (bb, ss) in tabla.index else pd.Series(dtype=float)
            n = float(f.sum())
            pr_ = (lambda c: float(f.get(c, 0.0)) / n) if n > 0 else (lambda c: 0.0)
            resto = (pr_("ball") * (W["Walk"] if bb == 3 else V[bb + 1, ss])
                     + (pr_("called_strike") + pr_("whiff")) * (W["Strikeout"] if ss == 2 else V[bb, ss + 1])
                     + pr_("hbp") * W["HitByPitch"]
                     + sum(pr_("ip_" + e) * W[e] for e in EVENTOS if e not in ("Walk", "HitByPitch", "Strikeout")))
            V[bb, ss] = resto / max(1e-9, 1 - pr_("foul")) if ss == 2 else resto + pr_("foul") * V[bb, ss + 1]
    # centrado: se resta a pesos y cuentas lo que vale la cuenta 0-0 (la cadena de cuentas usa la frecuencia de
    # resultados de TODOS los pitcheos y la de Markov la de las medias entradas completas). El valor de cada
    # pitcheo no cambia: es una diferencia de pesos y cuentas.
    v00 = float(V[0, 0])
    W = {e: x - v00 for e, x in W.items()}
    cuentas = {f"{bb}-{ss}": float(V[bb, ss]) - v00 for bb in range(4) for ss in range(3)}
    me_19 = int(clave[inning.between(1, 9)].nunique())
    carreras_me = float(runs[completa].sum() / max(1, n_me))
    VALORES_CARRERA.clear()
    VALORES_CARRERA.update({
        "metodo": "cadena de Markov de bases y outs calibrada con el dataset (RE24) + cadena de cuentas",
        "medias_entradas": n_me, "medias_entradas_1a9": me_19, "fraccion_completas": n_me / max(1, me_19),
        "juegos": int(len(Gn)), "turnos": int(Gn.sum()),
        "carreras_por_media_entrada": carreras_me,
        "carreras_en_jugadas_por_media_entrada": obj["por_media_entrada"], "re24": [float(x) for x in RE],
        "cuenta_00_antes_de_centrar": v00,
        "parametros_ajuste": {k: float(v) for k, v in zip(THETA, th)},
        "nota_parametros": "parametros de avance de corredores que reproducen los promedios del dataset; no son tasas medidas",
        "ajuste": {"objetivo": obj, **aj},
        "pesos": W, "pesos_ic95": {e: [float(a_ - v00), float(c_ - v00)] for e, a_, c_ in zip(EVENTOS, lo, hi)},
        "nota_ic95": "solo el ruido de muestreo de juegos (remuestreo), no los supuestos de la cadena",
        "pesos_regresion": reg, "frecuencia_eventos": p, "cuentas": cuentas})
    print(f"Valor en carreras estimado con el dataset: {n_me:,} medias entradas completas ({100 * n_me / max(1, me_19):.1f}% "
          f"de las de las entradas 1 a 9) de {len(Gn):,} juegos ({obj['por_media_entrada']:.3f} carreras en jugadas por "
          f"media entrada; la cadena de bases y outs da {RE[0]:.3f}; desvio max {100 * aj['desvio_max_frecuentes']:.1f}%). "
          "Pesos (positivo = bateador): " + ", ".join(f"{e} {W[e]:+.3f}" for e in EVENTOS)
          + f" | cuenta 0-0 {cuentas['0-0']:+.3f}, 0-2 {cuentas['0-2']:+.3f}, 3-0 {cuentas['3-0']:+.3f}")
    orden_ok = W["HomeRun"] > W["Triple"] > W["Double"] > W["Single"] > 0 > W["Out"] and W["Walk"] > 0 > W["Strikeout"]
    if not orden_ok or abs(v00) > 0.03 or aj["desvio_max_frecuentes"] > 0.25:
        _aviso_grave("el valor en carreras no salio razonable: orden de los pesos (jonron > triple > doble > sencillo > 0 > "
                     f"out, base por bolas > 0 > ponche), cuenta 0-0 = {v00:+.3f} antes de centrar (debe ser cerca de 0) o "
                     f"ajuste de la cadena (desvio maximo {aj['desvio_max_frecuentes']:.2f}).",
                     "Revisa RunsScored, OutsOnPlay, KorBB y play_result en el dataset.")
    return VALORES_CARRERA


def run_value_por_cuenta(df):
    """Valor en carreras de cada pitcheo (positivo = bateador) con los pesos
    lineales y valores de cuenta estimados con el propio dataset
    (estimar_valores_carrera).
      - pitcheo que NO termina el turno: valor(cuenta nueva) - valor(cuenta)
      - pitcheo que termina el turno:   peso(evento) - valor(cuenta)
    No necesita el orden de los pitcheos: la cuenta siguiente sale del resultado.
    Batazos: hit (banderas o play_result), error, y todo lo demas es out; que
    evento es cada etiqueta de play_result lo dice config.PLAY_RESULT_MAP. El
    sacrificio y la jugada de seleccion se valuan como out: solo ocurren con
    corredores en base y un peso propio le quitaria al rodado parte de su valor.
    Un batazo con una etiqueta que no esta en el mapa (y sin bandera de hit)
    tambien se valua como out, pero se avisa: el diccionario no enumera
    play_result y un error escrito de otra forma valdria como out."""
    if not VALORES_CARRERA:
        raise SystemExit("Falta estimar el valor en carreras (estimar_valores_carrera) antes de valuar los pitcheos")
    b, s = df["balls"].astype(int), df["strikes"].astype(int)
    cv = lambda bb, ss: (bb.astype(str) + "-" + ss.astype(str)).map(VALORES_CARRERA["cuentas"]).astype(float)
    antes = cv(b, s)
    call = df["call"]
    pr = _norm(df["play_result"]) if "play_result" in df else pd.Series(np.nan, index=df.index, dtype=object)
    segun_mapa = pr.map(PLAY_RESULT_MAP)
    W = VALORES_CARRERA["pesos"]
    rv = pd.Series(np.nan, index=df.index)
    ip = call == "in_play"
    evento = pd.Series("Out", index=df.index)
    for flag, ev in (("single", "Single"), ("double", "Double"), ("triple", "Triple"), ("home_run", "HomeRun")):
        evento[_flag(df, flag) | (segun_mapa == ev)] = ev
    evento[segun_mapa == "Error"] = "Error"
    print("Batazos por evento:", evento[ip].value_counts().to_dict(), "| play_result de los valuados como out:",
          _ascii(pr[ip & (evento == "Out")].value_counts(dropna=False).head(6).to_dict()))
    if "play_result" not in df:
        print("AVISO: el dataset no trae play_result: todo batazo sin bandera de hit se valua como out "
              "(un error tambien)")
    _sin_reconocer(df, ip & (evento == "Out") & pr.notna() & segun_mapa.isna(), ip, "play_result",
                   "Resultado del batazo (play_result)",
                   "Agrega esas etiquetas a config.PLAY_RESULT_MAP (Single, Double, Triple, HomeRun, Error u Out) "
                   "y vuelve a correr.", 0.5,
                   "se valuan como out")
    hits = evento[ip].isin(["Single", "Double", "Triple", "HomeRun"]).mean() if ip.any() else np.nan
    if hits < 0.15:
        _aviso_grave(f"solo {100 * hits:.0f}% de los batazos se reconocio como hit (lo normal es ~33%).",
                     "Revisa las banderas single / double / triple / home_run o las etiquetas de play_result.")
    rv[ip] = evento[ip].map(W) - antes[ip]
    rv[call == "hbp"] = W["HitByPitch"] - antes[call == "hbp"]
    bola = call == "ball"
    bb = bola & (b == 3)
    rv[bb] = W["Walk"] - antes[bb]
    rv[bola & ~bb] = cv(b + 1, s)[bola & ~bb] - antes[bola & ~bb]
    strike = call.isin(["called_strike", "whiff"])
    k = strike & (s == 2)
    rv[k] = W["Strikeout"] - antes[k]
    rv[strike & ~k] = cv(b, s + 1)[strike & ~k] - antes[strike & ~k]
    foul = call == "foul"
    rv[foul & (s == 2)] = 0.0
    rv[foul & (s < 2)] = cv(b, s + 1)[foul & (s < 2)] - antes[foul & (s < 2)]
    return rv


# ── recta de referencia y transporte entre densidades ───────────────────────
def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def _beta_de(tipos, beta, var):
    """Elasticidad de `var` (ivb / hb_n) para el tipo de pitcheo de cada fila.
    Sin medida se usa 1.0: fuerza de Magnus proporcional a la densidad."""
    tabla = {pt: b[var] for pt, b in (beta or {}).items() if np.isfinite(b.get(var, np.nan))}
    return pd.to_numeric(tipos.map(tabla), errors="coerce").fillna(1.0).to_numpy(float)


def _movimiento(d):
    """Movimiento total y por cada 1000 rpm (igual al construir y al transportar)."""
    d["mov_total"] = np.hypot(d["ivb"], d["hb_n"])
    d["mov_per_krpm"] = d["mov_total"] / (d["spin_rate"].where(d["spin_rate"] > 0) / 1000.0)


def _recta_en_su_aire(d, beta):
    """fb_ivb / fb_hb = la recta del pitcher en el aire de cada fila, y los
    diferenciales del pitcheo contra ella."""
    r = d["air_density"].to_numpy(float) / RHO_REF
    d["fb_ivb"] = d["fb_ivb0"] * r ** _beta_de(d["fb_type"], beta, "ivb")
    d["fb_hb"] = d["fb_hb0"] * r ** _beta_de(d["fb_type"], beta, "hb_n")
    d["ivb_diff_vs_fb"] = d["ivb"] - d["fb_ivb"]
    d["hb_diff_vs_fb"] = d["hb_n"] - d["fb_hb"]


def referencia_recta(df, beta=None):
    """fb_ivb0 / fb_hb0: quiebre medio de la recta primaria del pitcher-temporada llevado a RHO_REF;
    fb_ivb / fb_hb: esa recta en el aire de CADA pitcheo (df['air_density']);
    ivb_diff_vs_fb / hb_diff_vs_fb: pitcheo menos la recta en su mismo aire.

    Por que: con la recta promediada sobre todos los parques, el diferencial de
    un pitcheo delataba DONDE se lanzo (en altitud todo quiebra menos que la
    recta "promedio" del pitcher). Comparado contra la recta en su mismo aire,
    el diferencial mide solo la separacion que ve el bateador.
    beta: {pt: {"ivb": b, "hb_n": b}} de cargar_elasticidades(); lo que falte = 1.0.
    Es idempotente: solo lee pt, fb_type, ivb, hb_n y air_density."""
    return _poner_recta(df.copy(), beta)


def _poner_recta(d, beta):
    """referencia_recta escribiendo sobre `d` (para tablas grandes recien creadas)."""
    es_recta = (d["pt"] == d["fb_type"]).to_numpy()
    a_ref = RHO_REF / d["air_density"].to_numpy(float)
    grupo = [d["pitcher_id"], d["season"]]
    for var, col in (("ivb", "fb_ivb0"), ("hb_n", "fb_hb0")):
        en_ref = (d[var] * a_ref ** _beta_de(d["fb_type"], beta, var)).where(es_recta)
        d[col] = en_ref.groupby(grupo).transform("mean")
    _recta_en_su_aire(d, beta)
    return d


def transportar(df, rho, beta=None):
    """EL CONTRAFACTUAL DE ALTITUD: los mismos pitcheos llevados a la densidad
    `rho` (un numero, o un valor por fila en el orden de df).

        ivb_destino = ivb * (rho / rho_origen) ^ beta_ivb[tipo]     (igual hb_n)

    Tambien se mueve lo que depende del movimiento: la recta de referencia del
    pitcher (con la beta de SU tipo de recta), los diferenciales, el movimiento
    total y por rpm. El eje de giro gira lo mismo que la direccion del
    movimiento, porque el sistema de medicion infiere el SpinAxis de la
    trayectoria: por eso axis_dev NO cambia. Recalcular axis_dev contra un eje sin transportar
    fabricaba un cambio que no existe en los datos reales.

    NO se tocan velocidad, spin, punto de salida, extension, velo_diff_vs_fb,
    tunnel_dist ni eff_velo. LIMITACION: en aire delgado la bola pierde menos
    velocidad en el camino (~1 mph en CDMX); ese beneficio no se modela.

    df debe traer fb_ivb0 / fb_hb0 calculados con la MISMA beta (lo hace
    cargar_features). Devuelve una copia; conviene pasar solo las columnas que
    se vayan a usar."""
    faltan = [c for c in ("pt", "fb_type", "ivb", "hb_n", "spin_rate", "air_density", "fb_ivb0", "fb_hb0")
              if c not in df]
    if faltan:
        raise KeyError(f"transportar: faltan columnas {faltan} (carga los datos con features.cargar_features)")
    d = df.copy()
    rho = np.asarray(rho, dtype=float) * np.ones(len(d))
    ratio = rho / d["air_density"].to_numpy(float)
    ang0 = np.degrees(np.arctan2(d["ivb"], d["hb_n"]))
    d["ivb"] = d["ivb"] * ratio ** _beta_de(d["pt"], beta, "ivb")
    d["hb_n"] = d["hb_n"] * ratio ** _beta_de(d["pt"], beta, "hb_n")
    d["air_density"] = rho
    _recta_en_su_aire(d, beta)
    _movimiento(d)
    ang1 = np.degrees(np.arctan2(d["ivb"], d["hb_n"]))
    if "spin_axis_n" in d:
        d["spin_axis_n"] = (d["spin_axis_n"] - _wrap180(ang1 - ang0)) % 360.0
    if "mov_angle" in d:
        d["mov_angle"] = ang1
    return d


def cargar_elasticidades(path=None):
    """{pt: {"ivb": b, "hb_n": b}} con la columna beta_usada de
    out/altitude_elasticities.csv (physics.py decide ahi que elasticidad es
    confiable). Si no existe devuelve {} y todo se transporta con beta = 1."""
    path = Path(path) if path else config.OUT / "altitude_elasticities.csv"
    e = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if "beta_usada" not in e:
        print(f"AVISO: no hay {path.name} con la columna beta_usada (corre  {config.comando('physics')}) "
              "-> el movimiento se transporta con elasticidad 1.0")
        return {}
    e = e[e["var"].isin(["ivb", "hb_n"]) & e["pt"].isin(TIPOS)]
    beta = {}
    for pt, var, b in zip(e["pt"], e["var"], pd.to_numeric(e["beta_usada"], errors="coerce")):
        if np.isfinite(b):
            beta.setdefault(pt, {})[var] = float(b)
    return beta


def cargar_features(solo_con_rv=True):
    """La tabla de modelado lista para usar: out/features.parquet con la recta
    de referencia recalculada con las elasticidades medidas. model.py y
    score.py cargan los datos SOLO por aqui (asi entrenamiento, scores y
    simulador ven los mismos diferenciales)."""
    ruta = config.OUT / "features.parquet"
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}: corre primero  {config.comando('features')}")
    df = _poner_recta(pd.read_parquet(ruta), cargar_elasticidades())
    con_rv = np.isfinite(df["run_value"])
    if solo_con_rv and not con_rv.all():
        df = df[con_rv]
    return df.reset_index(drop=True)


# ── construccion de la tabla de modelado ─────────────────────────────────────
RESUMEN_LIMPIEZA = {}     # lo llena _limpiar; main() lo guarda en out/limpieza.json
MAX_SIN_RASTREO = 15      # % de batazos sin rastreo arriba del cual el aviso es grave (con --estricto detiene)


def _limpiar(df):
    """Identificadores, tipo de pitcheo, resultado y rangos fisicos."""
    # IDs: se trabajan como texto (ISAC los da como "pitcher_00123"); un id que
    # llegue como numero se pasa a texto sin decimales.
    pid = df["pitcher_id"].where(df["pitcher_id"].notna()).astype(object)
    pid = pid.map(lambda v: v if pd.isna(v) else (str(int(v)) if isinstance(v, float) else str(v)))
    if pid.isna().any() and "pitcher_name" in df:
        # 1) si el MISMO nombre tiene id en otras filas, se usa ese id: si no,
        #    un pitcher quedaria duplicado y se filtraria entre los folds de la
        #    validacion por pitcher
        por_nombre = (df.assign(_p=pid).dropna(subset=["_p"])
                        .groupby("pitcher_name")["_p"].agg(lambda x: x.mode().iat[0]))
        pid = pid.fillna(df["pitcher_name"].map(por_nombre))
        # 2) solo los que de plano no tienen id reciben uno derivado del nombre
        cod = "nom_" + pd.Series(pd.factorize(df["pitcher_name"].fillna("?"))[0], index=df.index).astype(str)
        pid = pid.fillna(cod)
    if pid.isna().any():
        print(f"Sin pitcher_id: {int(pid.isna().sum()):,} pitcheos descartados")
    df = df[pid.notna()].copy()                     # copia propia: la tabla cruda no se toca
    df["pitcher_id"] = pid[pid.notna()]
    if "pitcher_name" not in df:
        df["pitcher_name"] = df["pitcher_id"]       # ISAC: anonimo

    n0 = len(df)
    todos = pd.Series(True, index=df.index)
    tipo_crudo = _norm(df["pitch_type"])
    df["pt"] = tipo_crudo.map(PT_MAP)
    if "pitch_type_tagged" in df:
        df["pt"] = df["pt"].fillna(_norm(df["pitch_type_tagged"]).map(PT_MAP))
    sin_tipo = df["pt"].isna() & tipo_crudo.isin(PT_DESCARTAR)      # 'Other': previsto en el diccionario
    tipo_raro = df["pt"].isna() & ~sin_tipo
    _sin_reconocer(df, tipo_raro, todos, "pitch_type", "Tipo de pitcheo",
                   "Agrega esas etiquetas a PT_MAP (arriba en features.py) y vuelve a correr.", 3)
    crudo = _norm(df["pitch_call"])
    df["call"] = crudo.map(PITCH_CALL_MAP)
    descartar = crudo.isin(PITCH_CALL_DESCARTAR)
    sin_call = df["call"].isna() & ~descartar
    _sin_reconocer(df, sin_call, todos, "pitch_call", "PitchCall",
                   "Agrega esas etiquetas a config.PITCH_CALL_MAP y vuelve a correr.", 1)
    # tipo de batazo: gb / ld / fb / pu. Los toques y los batazos sin tipo se
    # sacan del modelado (antes caian todos en "rodado" sin avisar).
    en_juego = df["call"] == "in_play"
    df["bb_type"] = (_norm(df["hit_type"]).map(HIT_TYPE_MAP) if "hit_type" in df
                     else pd.Series(np.nan, index=df.index, dtype=object))
    df.loc[~en_juego, "bb_type"] = np.nan
    toque = en_juego & (df["bb_type"] == "bunt")
    # Un batazo que el radar no rastreo no trae tipo, ni angulo, ni velocidad de salida: no es una etiqueta
    # que le falte a HIT_TYPE_MAP (no hay nada que agregar). En el archivo oficial son 5.6% de los batazos, casi
    # todos de 2024. Se descartan aparte y se guarda como acabaron (out/limpieza.json, lo cita el documento).
    medido = pd.Series(False, index=df.index)
    for c in ("launch_angle", "exit_velo"):
        if c in df:
            medido |= pd.to_numeric(df[c], errors="coerce").notna()
    sin_etiqueta = (_norm(df["hit_type"]).fillna("").isin(["", "nan", "none", "null", "undefined"])
                    if "hit_type" in df else pd.Series(True, index=df.index))
    sin_rastreo = en_juego & df["bb_type"].isna() & sin_etiqueta & ~medido
    sin_tipo_bat = en_juego & df["bb_type"].isna() & ~sin_rastreo
    _sin_reconocer(df, sin_tipo_bat, en_juego, "hit_type", "Batazos (hit_type)",
                   "Agrega esas etiquetas a config.HIT_TYPE_MAP (gb / ld / fb / pu / bunt) y vuelve a correr.", 1)
    evento = (_norm(df["play_result"]).map(PLAY_RESULT_MAP).fillna("otro") if "play_result" in df
              else pd.Series("otro", index=df.index))
    rastreado = en_juego & ~sin_rastreo & ~sin_tipo_bat & ~toque
    RESUMEN_LIMPIEZA.clear()
    RESUMEN_LIMPIEZA.update({
        "batazos": int(en_juego.sum()), "sin_rastreo": int(sin_rastreo.sum()),
        "sin_rastreo_por_evento": {str(k): int(v) for k, v in evento[sin_rastreo].value_counts().items()},
        "rastreados": int(rastreado.sum()),
        "rastreados_por_evento": {str(k): int(v) for k, v in evento[rastreado].value_counts().items()}})
    if sin_rastreo.any():
        pct = 100 * sin_rastreo.sum() / en_juego.sum()
        print(f"Batazos sin rastreo (sin hit_type, angulo ni velocidad de salida): {int(sin_rastreo.sum()):,} "
              f"({pct:.2f}% de los batazos) -> se descartan: no hay como saber su tipo. Como acabaron: "
              + _ascii(RESUMEN_LIMPIEZA["sin_rastreo_por_evento"]))
        if pct > MAX_SIN_RASTREO:
            _aviso_grave(f"{pct:.1f}% de los batazos no trae rastreo (ni hit_type, ni angulo, ni velocidad de salida).",
                         f"Pasa de {MAX_SIN_RASTREO}% (en el archivo oficial es 5.6%): revisa que COLUMN_MAP lea "
                         "hit_type, Angle y ExitSpeed en config.py.")
    if (sin_tipo_bat | sin_rastreo).sum() > 0.5 * en_juego.sum():
        # sin tipo de batazo no hay clases de batazo: seguir daria un modelo de 5 resultados en vez de 10
        raise SystemExit("Mas de la mitad de los batazos no trae un tipo de batazo reconocible (columna hit_type "
                         "del diccionario): sin el no se pueden armar las clases de resultado. Revisa que la "
                         "columna exista (COLUMN_MAP) y sus etiquetas (HIT_TYPE_MAP) en config.py")
    quitar, partes = pd.Series(False, index=df.index), []
    for motivo, m in (("sin tipo de pitcheo ('Other') o knuckleball", sin_tipo | (df["pt"] == "KN")),
                      ("con tipo de pitcheo sin reconocer", tipo_raro),
                      ("bola intencional/automatica o PitchCall 'Undefined'", descartar),
                      ("sin PitchCall reconocido", sin_call), ("toques", toque),
                      ("batazos sin tipo reconocido", sin_tipo_bat), ("batazos sin rastreo", sin_rastreo)):
        partes.append(f"{int((m & ~quitar).sum()):,} {motivo}")
        quitar |= m
    df = df[~quitar].copy()
    print(f"Limpieza: {n0:,} -> {len(df):,} pitcheos (fuera: {', '.join(partes)})")

    for c in ("velocity", "spin_rate", "spin_axis", "ivb", "hb", "extension", "release_height",
              "release_side", "plate_x", "plate_z", "eff_velo", "vra", "hra", "vaa", "haa",
              "plate_speed", "speed_drop", "zone_time", "az0", "exit_velo", "launch_angle", "direction",
              "hit_distance", "season", "balls", "strikes"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    n1 = len(df)
    df = df.dropna(subset=["velocity", "ivb", "hb", "plate_x", "plate_z", "balls", "strikes", "season"])
    df = df[df["velocity"].between(55, 106) & df["ivb"].between(-35, 35) & df["hb"].between(-35, 35)]
    mano = df["p_throws"].astype(str).str.strip().str[:1].str.upper()
    mano = mano.where(mano.isin(["R", "L"]))
    # un pitcher lanza con la misma mano en todos sus pitcheos: si el dato falta en
    # algunos se toma de los demas (antes esos pitcheos se quedaban sin prediccion)
    propia = mano.groupby(df["pitcher_id"]).transform(lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
    rescatados = int((mano.isna() & propia.notna()).sum())
    mano = mano.fillna(propia)
    df = df[mano.notna()].copy()
    print(f"Fisica: {n1:,} -> {len(df):,} pitcheos (fuera: sin velocidad, movimiento, ubicacion o cuenta, "
          "valores imposibles o pitcher sin mano)"
          + (f" | mano del pitcher tomada de sus otros pitcheos: {rescatados:,}" if rescatados else ""))
    if df.empty:
        raise SystemExit("No quedo ningun pitcheo despues de la limpieza: revisa los avisos de arriba, "
                         "COLUMN_MAP y los mapas de etiquetas de config.py")
    df["season"] = df["season"].astype(int)
    df["lefty"] = (mano[df.index] == "L").astype(int)
    return df


def _marco_diestro(df):
    """Espejea lo horizontal de los zurdos y calcula el movimiento derivado."""
    sgn = np.where(df["lefty"] == 1, -1.0, 1.0)
    df["hb_n"] = df["hb"] * sgn
    df["rel_side_n"] = df["release_side"] * sgn
    df["plate_x_n"] = df["plate_x"] * sgn
    for c in ("haa", "hra"):
        if c in df:
            df[c + "_n"] = df[c] * sgn
    _movimiento(df)
    # Diagnosticos, NO variables de ningun modelo. Convencion verificada con
    # los datos: la direccion de movimiento que implica el eje es 270 - eje. El
    # sistema de medicion infiere SpinAxis de la trayectoria, asi que el eje
    # sigue al movimiento y no agrega informacion a IVB/HB.
    df["mov_angle"] = np.degrees(np.arctan2(df["ivb"], df["hb_n"]))
    if "spin_axis" in df:
        df["spin_axis_n"] = np.where(df["lefty"] == 1, 360.0 - df["spin_axis"], df["spin_axis"])
        df["axis_dev"] = _wrap180(df["mov_angle"] - (270.0 - df["spin_axis_n"]))
    return df


def _ambiente(df):
    """Altitud, densidad del aire, banda y casa. Con el dataset oficial todo
    sale de altitude_category (no trae estadio)."""
    if modo_estadio(df):                              # datos que traen el estadio exacto (claves de STADIUMS)
        alt = df["stadium"].map({k: v[1] for k, v in STADIUMS.items()})
        if alt.isna().any():
            print(f"Estadio fuera de config.STADIUMS: {int(alt.isna().sum()):,} pitcheos descartados")
        df = df[alt.notna()].copy()
        df["altitude_m"] = alt[alt.notna()].astype(float)
        df["air_density"] = densidad_aire(df["altitude_m"])
    else:
        if "altitude_category" not in df:
            raise SystemExit("El dataset no trae altitude_category ni estadios de config.STADIUMS")
        nulos = df["altitude_category"].isna()
        if nulos.any():
            print(f"Sin altitude_category: {int(nulos.sum()):,} pitcheos descartados")
            df = df[~nulos].copy()
        df["altitude_category"] = df["altitude_category"].astype(str).str.strip()
        faltan = sorted(set(df["altitude_category"]) - set(ALTITUDE_M))
        if faltan:
            raise SystemExit(f"Categorias de altitud sin metros en config.ALTITUDE_M: {_ascii(faltan)}. "
                             "Agregalas (etiqueta exacta -> metros), revisa HOME_ALTITUDE y ALTITUDE_RHO, "
                             "y vuelve a correr.")
        df["stadium"] = df["altitude_category"]       # la categoria hace de "estadio"
        df["altitude_m"] = df["stadium"].map(ALTITUDE_M).astype(float)
        # densidad promedio MEDIDA de la categoria; si la etiqueta no tiene una,
        # atmosfera estandar corregida por el clima tipico de la LMB
        rho = df["stadium"].map(ALTITUDE_RHO)
        if rho.isna().any():
            print(f"AVISO: categorias sin densidad en config.ALTITUDE_RHO: "
                  f"{_ascii(sorted(set(df.loc[rho.isna(), 'stadium'])))} -> atmosfera estandar x CLIMA_FACTOR")
        df["air_density"] = rho.fillna(pd.Series(densidad_aire(df["altitude_m"]), index=df.index)).astype(float)

    estadio = modo_estadio(df)
    df["banda"] = banda_de(df)
    df["es_casa"] = ((df["stadium"] == HOME_STADIUM) if estadio
                     else (df["altitude_category"] == HOME_ALTITUDE)).astype(int)
    if not df["es_casa"].any():
        raise SystemExit(f"Ningun pitcheo es de casa ({HOME_STADIUM if estadio else HOME_ALTITUDE}). "
                         f"En los datos hay: {_ascii(sorted(df['stadium'].unique()))}. "
                         "Corrige HOME_ALTITUDE / HOME_STADIUM en config.py")
    print(f"MODO {'ESTADIO' if estadio else 'CATEGORIA'}: {df['stadium'].nunique()} "
          f"{'estadios' if estadio else 'categorias de altitud'}. Bandas (densidad media):")
    t = df.groupby(["banda", "stadium"])["air_density"].agg(["size", "mean"]).reset_index()
    for b in ("alta", "media", "mar"):
        s = t[t["banda"] == b]
        print(f"  {b:5s} {int(s['size'].sum()):>9,} pitcheos: "
              + ", ".join(f"{_ascii(e)} ({r:.3f})" for e, r in zip(s["stadium"], s["mean"])))
    if estadio:
        print(f"  casa = {HOME_STADIUM}: {int(df['es_casa'].sum()):,} pitcheos")
    else:
        print(f"  casa = categoria {_ascii(HOME_ALTITUDE)}: {int(df['es_casa'].sum()):,} pitcheos. OJO: junta "
              "TODOS los parques de esa categoria, no solo el Harp Helu")
    if not (df["banda"] == "mar").any():
        _aviso_grave("ningun pitcheo cae en la banda 'mar': no hay con que comparar la altitud.",
                     "Revisa los metros de cada categoria en config.ALTITUDE_M (mar = hasta "
                     f"{BANDA_BAJA_M} m).")
    return df


def _arsenal(df):
    """Uso, repertorio y recta primaria (FF, SI o FC mas usada) por pitcher-temporada."""
    llave = ["pitcher_id", "season"]
    uso = df.groupby(llave + ["pt"]).size().rename("n").reset_index()
    uso["usage_pct"] = uso["n"] / uso.groupby(llave)["n"].transform("sum")
    uso["n_pitch_types"] = (uso["usage_pct"] >= USO_MIN_REPERTORIO).groupby(
        [uso["pitcher_id"], uso["season"]]).transform("sum")
    primaria = (uso[uso["pt"].isin(FASTBALLS)].sort_values(["n", "pt"], ascending=[False, True])
                .drop_duplicates(llave)[llave + ["pt"]].rename(columns={"pt": "fb_type"}))
    df = (df.merge(uso[llave + ["pt", "usage_pct", "n_pitch_types"]], on=llave + ["pt"], how="left")
            .merge(primaria, on=llave, how="left"))
    es_recta, grupo = df["pt"] == df["fb_type"], [df["pitcher_id"], df["season"]]
    for nueva, col in (("fb_velo", "velocity"), ("fb_rel_h", "release_height"),
                       ("fb_rel_s", "rel_side_n"), ("fb_ext", "extension")):
        df[nueva] = df[col].where(es_recta).groupby(grupo).transform("mean")
    df["velo_diff_vs_fb"] = df["velocity"] - df["fb_velo"]
    # tunel: que tan lejos sale la mano, EN PROMEDIO, de donde sale la recta
    # (pulgadas). Es un rasgo del repertorio (pitcher-temporada-tipo), no de cada
    # pitcheo: con el punto de salida de cada pitcheo la variable media la
    # dispersion de la mano, que es comando y no "stuff" (y el simulador, que
    # recibe promedios, no podia reproducirla). La recta primaria vale 0.
    tipo = [df["pitcher_id"], df["season"], df["pt"]]
    df["tunnel_dist"] = 12 * np.hypot(df["release_height"].groupby(tipo).transform("mean") - df["fb_rel_h"],
                                      df["rel_side_n"].groupby(tipo).transform("mean") - df["fb_rel_s"])
    df["is_fastball"] = df["pt"].isin(FASTBALLS).astype(int)
    # quiebre de la recta y diferenciales: en el aire de cada pitcheo. Aqui con
    # elasticidad 1.0; cargar_features los recalcula con las medidas.
    return _poner_recta(df, None)


def chequeo_etiquetas(df, caja, zona_usada):
    """Etiquetas que derivamos vs banderas oficiales del dataset (una fila por
    par). PitchCall es la fuente de verdad de swing / whiff / strike cantado
    (el diccionario enumera sus valores); la bandera oficial solo define la
    zona cuando viene completa."""
    pares = [("y_swing", df["y_swing"], "flag_swing", "PitchCall"),
             ("y_whiff", df["y_whiff"], "flag_whiff", "PitchCall"),
             ("y_called_strike", df["y_called_strike"], "flag_called_strike", "PitchCall"),
             ("in_zone_caja", caja, "flag_in_zone", zona_usada),
             ("y_chase", df["y_chase"], "flag_chase", "swing de PitchCall y zona: " + zona_usada)]
    filas, difieren = [], {}
    for nombre, der, bandera, usada in pares:
        b = _bandera(df, bandera)
        ok = b.notna()
        fila = {"derivada": nombre, "bandera": bandera, "n": int(ok.sum()), "tasa_derivada": np.nan,
                "tasa_bandera": np.nan, "acuerdo": np.nan, "usada": usada}
        if ok.any():
            fila.update(tasa_derivada=float(der[ok].mean()), tasa_bandera=float(b[ok].mean()),
                        acuerdo=float((der[ok] == b[ok]).mean()))
            difieren[nombre] = df.loc[ok & (der != b), "pitch_call"].value_counts().head(5).to_dict()
        filas.append(fila)
        print(f"  {nombre:15s} vs {bandera:18s}: " + (
            f"n {fila['n']:>9,} | tasa {fila['tasa_derivada']:.3f} vs {fila['tasa_bandera']:.3f} | "
            f"acuerdo {100 * fila['acuerdo']:.1f}% | se usa {usada}" if ok.any()
            else ("la bandera viene pero no se pudo leer" if bandera in df else "el dataset no trae la bandera")
            + f" | se usa {usada}"))
    t = pd.DataFrame(filas)
    malos = t[t["derivada"].isin(["y_swing", "y_whiff"]) & (t["acuerdo"] < 0.98)]
    # caso conocido y no grave: la bandera de swing del archivo no cuenta los
    # fouls (un foul ES un swing). Si TODO el desacuerdo de y_swing son fouls que
    # la bandera deja en 0, se avisa sin detener la corrida de entrega.
    b = _bandera(df, "flag_swing")
    dif = b.notna() & (df["y_swing"] != b)
    solo_fouls = bool(dif.any()) and float(((df["call"] == "foul") & (b == 0))[dif].mean()) >= 0.995
    t["nota"] = ""
    if solo_fouls:
        t.loc[t["derivada"].isin(["y_swing", "y_chase"]), "nota"] = "la bandera no cuenta los fouls como swing"
        if (malos["derivada"] == "y_swing").any():
            print(f"AVISO: la bandera oficial de swing no cuenta los fouls ({int(dif.sum()):,} pitcheos); un foul es "
                  "un swing, asi que se usa PitchCall. Todo el desacuerdo viene de ahi.")
        malos = malos[malos["derivada"] != "y_swing"]
    if len(malos):
        _aviso_grave(
            "las banderas oficiales NO coinciden con PitchCall en: " + ", ".join(
                f"{r.bandera} ({100 * r.acuerdo:.1f}%)" for r in malos.itertuples()),
            "Se usa PitchCall para swing / whiff / strike cantado (es la columna cuyos valores enumera",
            "el diccionario). PitchCall de las filas que difieren: " + "; ".join(
                f"{d} {_ascii(difieren[d])}" for d in malos["derivada"]),
            "Si alguna de esas etiquetas esta mal clasificada, corrige config.PITCH_CALL_MAP.")
    return t


def _ubicacion_y_resultado(df):
    """Zona, cuenta, contexto, clase de resultado, etiquetas y run value."""
    df["balls"], df["strikes"] = df["balls"].clip(0, 3), df["strikes"].clip(0, 2)
    df["count"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    _banderas_ilegibles(df)
    # zona: la bandera oficial si viene completa (>= 99%); si no, una caja fija
    caja = ((df["plate_x"].abs() <= 0.83) & df["plate_z"].between(1.5, 3.5)).astype(int)
    oficial = _bandera(df, "flag_in_zone")
    usa_oficial = oficial.notna().mean() >= 0.99
    zona_usada = "bandera oficial in_strike_zone" if usa_oficial else "caja 0.83 x 1.5-3.5 pies"
    df["in_zone"] = oficial.fillna(caja).astype(int) if usa_oficial else caja
    df["zone_dist"] = np.hypot(df["plate_x_n"], df["plate_z"] - 2.5)
    # lado del bateador: si no viene, el dato queda vacio (no se inventa "diestro")
    bat = df["stand"].astype(str).str.strip().str[:1].str.upper()
    _sin_reconocer(df, ~bat.isin(["R", "L"]), pd.Series(True, index=df.index), "stand",
                   "Lado del bateador (BatterSide)", "Debe empezar con R o L (Right / Left, como en el diccionario).",
                   5, "stand_l y same_side quedan vacios")
    df["stand_l"] = np.where(bat.isin(["R", "L"]), (bat == "L").astype(float), np.nan)
    df["same_side"] = np.where(bat.isin(["R", "L"]), (df["lefty"] == df["stand_l"]).astype(float), np.nan)
    # ISAC no trae corredores: base_out son solo los outs. Si el dataset trajera
    # on_1b/on_2b/on_3b se armarian los 24 estados.
    df["outs"] = pd.to_numeric(df["outs"], errors="coerce").fillna(0).clip(0, 2) if "outs" in df else 0
    df["base_out"] = df["outs"]
    if all(b in df for b in ("on_1b", "on_2b", "on_3b")):
        df["base_out"] = (_flag(df, "on_1b") + 2 * _flag(df, "on_2b") + 4 * _flag(df, "on_3b")
                          + 8 * df["outs"])

    # clase de resultado. Un elevado sin velocidad de salida medida queda como
    # ip_fb_soft.
    en_juego = df["call"] == "in_play"
    ev = df["exit_velo"] if "exit_velo" in df else pd.Series(np.nan, index=df.index)
    elevado = en_juego & (df["bb_type"] == "fb")
    df["outcome"] = df["call"]
    df.loc[en_juego, "outcome"] = "ip_" + df.loc[en_juego, "bb_type"]
    df.loc[elevado, "outcome"] = np.where(ev[elevado] >= 95, "ip_fb_hard", "ip_fb_soft")

    # etiquetas; los denominadores correctos de cada tasa los arma score.py
    swing = df["call"].isin(["whiff", "foul", "in_play"])
    df["y_swing"] = swing.astype(int)
    df["y_whiff"] = (df["call"] == "whiff").astype(int)
    df["y_called_strike"] = (df["call"] == "called_strike").astype(int)
    df["y_ooz"] = (df["in_zone"] == 0).astype(int)                # pitcheo fuera de zona
    df["y_chase"] = (swing & (df["in_zone"] == 0)).astype(int)    # swing a pitcheo fuera de zona
    df["y_bip"] = en_juego.astype(int)                            # batazo en juego
    df["y_ev"] = (en_juego & ev.notna()).astype(int)              # batazo con velocidad de salida medida
    df["y_gb"] = (df["outcome"] == "ip_gb").astype(int)
    df["y_hard"] = (en_juego & (ev >= 95)).astype(int)
    df["y_weak"] = (en_juego & (ev < 85)).astype(int)
    print(f"Zona: se usa {zona_usada}. Etiquetas derivadas vs banderas oficiales:")
    config.OUT.mkdir(exist_ok=True)
    chequeo_etiquetas(df, caja, zona_usada).to_csv(config.OUT / "chequeo_etiquetas.csv", index=False)

    # valor en carreras: el del dataset si viene; si no (ISAC), se calcula
    dado = pd.to_numeric(df["run_value"], errors="coerce") if "run_value" in df else None
    if dado is None or dado.notna().mean() < .5:
        df["run_value"] = run_value_por_cuenta(df)
        print("run_value calculado con los pesos lineales estimados con el dataset (no venia en el dataset)")
    else:
        df["run_value"] = dado
    return df


def _texto_uniforme(df):
    """Columnas con numeros y texto mezclados: pasa cuando el archivo viene en
    partes y una columna que no se convierte a numero (un identificador, una
    bandera) se leyo como numero en una parte y como texto en otra. Parquet no
    puede guardarlas: se pasan completas a texto y se avisa cuales."""
    mezcladas = [c for c in df.columns[(df.dtypes == object).to_numpy()]
                 if pd.api.types.infer_dtype(df[c], skipna=True).startswith("mixed")]
    for c in mezcladas:
        df[c] = df[c].map(lambda v: v if pd.isna(v) else str(v))
    if mezcladas:
        print(f"AVISO: columnas con numeros y texto mezclados entre las partes del archivo: {_ascii(mezcladas)} "
              "-> se guardan como texto")
    return df


def build(df):
    """De las columnas canonicas crudas a una fila de modelado por pitcheo."""
    if not VALORES_CARRERA:              # en --rapido ya se estimo en load_raw, antes del recorte
        estimar_valores_carrera(df)      # con la tabla completa, antes de descartar pitcheos
    df = _limpiar(df)
    df = _marco_diestro(df)
    df = _ambiente(df)
    df = _arsenal(df)
    df = _ubicacion_y_resultado(df)
    return _texto_uniforme(df.reset_index(drop=True))


# ── conjuntos de variables por sub-modelo ────────────────────────────────────
# Stuff+ = solo fisica del pitcheo en el release y su relacion con la recta del
# pitcher. Lo que se QUITO de Stuff+ y por que:
#   air_density   identifica la categoria de altitud del parque; la altitud
#                 entra por el transporte del movimiento y por el valor de los
#                 batazos, no como variable
#   eff_velo      depende del aire (la bola frena menos en altitud), quedaba
#                 congelada en el contrafactual y no existe en el simulador
#   usage_pct, n_pitch_types   son decisiones del pitcher, no fisica; se quedan
#                 en Pitching+
#   spin_axis_n, axis_dev   el eje (SpinAxis) se infiere del movimiento: es
#                 redundante con IVB/HB y creaba un artefacto en el contrafactual
#                 de altitud
# VertRelAngle / HorzRelAngle NO entran a Stuff+ aunque el diccionario los
# permita: junto con el punto de salida, la velocidad y el quiebre reconstruyen
# por donde cruza el plato -> Stuff+ mediria comando a escondidas. Si entran a
# Pitching+, que si ve la ubicacion.
STUFF_FEATS = ["velocity", "ivb", "hb_n", "spin_rate", "release_height", "rel_side_n", "extension",
               "mov_total", "mov_per_krpm", "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb",
               "tunnel_dist", "is_fastball", "lefty"]
LOCATION_FEATS = ["plate_x_n", "plate_z", "zone_dist", "in_zone", "balls", "strikes", "same_side"]
CONTEXTO_FEATS = ["usage_pct", "n_pitch_types", "stand_l", "base_out", "vaa", "haa_n", "vra", "hra_n"]
PITCHING_FEATS = STUFF_FEATS + LOCATION_FEATS + CONTEXTO_FEATS
CATEGORICAL = ["pt"]


if __name__ == "__main__":
    import time
    t = time.time()
    df = build(load_raw())
    config.OUT.mkdir(exist_ok=True)
    df.to_parquet(config.OUT / "features.parquet", index=False)
    (config.OUT / "limpieza.json").write_text(json.dumps(RESUMEN_LIMPIEZA, ensure_ascii=True, indent=1),
                                              encoding="utf-8")
    (config.OUT / "valores_carrera.json").write_text(json.dumps(VALORES_CARRERA, ensure_ascii=True, indent=1),
                                                     encoding="utf-8")
    modo = ("estadio", "estadios") if modo_estadio(df) else ("categoria", "categorias de altitud")
    print(f"{len(df):,} pitcheos | {df['pitcher_id'].nunique():,} pitchers | modo {modo[0]} "
          f"({df['stadium'].nunique()} {modo[1]}) | {time.time() - t:.0f}s")
    print("Resultados:", df["outcome"].value_counts(normalize=True).round(3).to_dict())
    print("Bandas:", df["banda"].value_counts().to_dict(), "| casa:", int(df["es_casa"].sum()),
          "| sin run_value:", int(df["run_value"].isna().sum()))
