"""
SIMULADOR — de las caracteristicas fisicas de un pitcheo a su Stuff+ en el
Harp Helu y al nivel del mar, lado a lado.

El pitcheo se captura "como se midio" (por ejemplo el bullpen de un prospecto
en Monterrey) y hay que decir DONDE se midio: el mismo quiebre vale distinto
segun el aire. De ahi:
  1. fila_pitcheo arma la fila con las variables del modelo Stuff, con la misma
     funcion que usan las tablas (features.transportar).
  2. evaluar la lleva a la densidad destino con las elasticidades medidas en la
     LMB (out/altitude_elasticities.csv), la califica con models/stuff.joblib
     y valua los batazos en ese aire (model.xrv).
  3. Stuff+ = 100 + 10 * z contra el pitcheo promedio de la liga lanzado en ESE
     mismo aire (rejilla de models/scale.json). Es la escala POR PITCHEO.

MARCO: todo lo horizontal se captura con "lado del brazo positivo", igual para
diestros y zurdos. Es el marco en que se entreno el modelo (features.py espejea
a los zurdos), asi que aqui no se cambia ningun signo: un zurdo solo lleva
lefty = 1.

TUNEL. tunnel_dist es un rasgo del REPERTORIO, no de cada pitcheo (igual que
en features._arsenal): la distancia, en pulgadas, entre el punto de salida
PROMEDIO del tipo y el de la recta primaria del pitcher; 0 cuando el pitcheo ES
la recta primaria. Aqui sale de esos dos promedios si se dan; sin el punto de
salida de la recta se usa el tunel mediano del tipo entre quienes no lo tienen
de recta primaria (tunel_tipico). Todas las repeticiones de un promedio
tecleado llevan el mismo tunel.

UN PITCHEO vs EL PROMEDIO DE UN PITCHEO. fila_pitcheo + evaluar califican UN
pitcheo concreto. Pero lo que un coach teclea es el PROMEDIO de un tipo de
pitcheo, y un pitcheo lanzado exactamente en su promedio no existe: los reales
se dispersan alrededor (velocidad, quiebre, spin, punto de salida) y el modelo
no es lineal. Calificar el promedio como si fuera un pitcheo se parece poco al
Stuff+ medio real del mismo arsenal. Por eso el dashboard usa
calificar_promedio: el promedio tecleado mas la dispersion con que los pitchers
de la liga repiten ese tipo de pitcheo (N_NUBE repeticiones), y promedia. La
seccion 3 de la prueba mide los dos caminos con los datos que haya.

OJO: el simulador usa el modelo ajustado con TODOS los datos; las tablas usan
los modelos de la validacion cruzada (a cada pitcher lo califica uno que no lo
vio). Por eso un arsenal tecleado se compara contra los arsenales de la liga
calificados por este mismo camino (liga_tecleada), no contra las tablas; que
tanto coinciden los dos caminos lo mide contra_el_ranking, y el dashboard lo
dice con ese numero.

    python simulator.py     prueba de regresion: pitcheos reales pasados por
                            fila_pitcheo + evaluar contra el mismo modelo sobre
                            las filas reales, el camino del dashboard (promedios
                            tecleados, arsenales, liga) y barridos de cordura.
                            PASA / FALLA
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

# Las rutas (OUT, MODELS) se leen como config.X al usarse, para que una prueba
# pueda cambiarlas.
import config
import features
from config import (RHO_CDMX, RHO_MAR, RHO_REF, PLUS_SCALE, RANDOM_STATE, MIN_PITCHES_PITCHER,
                    MIN_PITCHES_ARSENAL)
from features import TIPOS, FASTBALLS
from model import proba, xrv, I, IP, SWING

# La rejilla de scale.json cubre densidades de 0.72 a 1.00. Un destino apenas
# fuera (el Harp Helu en una tarde de 33 C) usa la media del extremo; mas lejos
# ya no hay "pitcheo promedio de ese aire" contra que comparar.
MARGEN_REJILLA = 0.03
# variable del modelo <- dato que captura el usuario
FISICA = {"velocity": "velo", "ivb": "ivb", "hb_n": "hb_brazo", "spin_rate": "spin",
          "release_height": "rel_h", "rel_side_n": "rel_s_brazo", "extension": "ext"}
# Repeticiones con que se representa un promedio tecleado. El promedio de las repeticiones tiene un error de
# simulacion (segun que desviaciones tocaron) que calificar_promedio devuelve junto al numero.
N_NUBE = 300
# Los arsenales de la liga de referencia son miles: cada uno lleva menos repeticiones (es lo que tarda el
# dashboard al abrir) y su error, independiente de un arsenal a otro, apenas ensancha la referencia.
N_NUBE_LIGA = 60
_CACHE = {}


class SimuladorNoListo(Exception):
    """Faltan, o son de otra version, los archivos que deja el pipeline."""


def _rutas():
    return {"modelo": config.MODELS / "stuff.joblib", "valores": config.MODELS / "outcome_values.json",
            "escala": config.MODELS / "scale.json", "defaults": config.MODELS / "sim_defaults.json",
            "beta": config.OUT / "altitude_elasticities.csv"}


def faltantes():
    """Archivos que el simulador necesita y todavia no existen (carpeta/nombre)."""
    return [f"{p.parent.name}/{p.name}" for p in _rutas().values() if not p.exists()]


def recursos():
    """Modelo Stuff de todos los datos, valor de cada resultado, escala,
    medianas por tipo y elasticidades. Se vuelven a leer solos cuando el
    pipeline los reescribe: el dashboard puede quedarse abierto."""
    if faltantes():
        raise SimuladorNoListo("Faltan " + ", ".join(faltantes()) + ". Corre primero:  python run_all.py")
    rutas = _rutas()
    firma = tuple((str(p), p.stat().st_mtime_ns) for p in rutas.values())
    if _CACHE.get("firma") != firma:
        lee = lambda k: json.loads(rutas[k].read_text(encoding="utf-8"))
        escala, valores, defaults = lee("escala"), lee("valores"), lee("defaults")
        if "rejilla" not in escala or "densidad" not in valores:
            raise SimuladorNoListo("models/scale.json u outcome_values.json son de una version anterior del "
                                   "codigo. Corre:  python run_all.py --desde valores")
        _CACHE.clear()
        _CACHE.update(firma=firma, modelo=joblib.load(rutas["modelo"]), valores=valores, escala=escala,
                      defaults={p: d for p, d in defaults.items() if p in TIPOS},
                      beta=features.cargar_elasticidades(rutas["beta"]),
                      rejilla=np.array(sorted((n["rho"], n["mu"]) for n in escala["rejilla"])))
    return _CACHE


def _de_salida(nombre, pide):
    """Un archivo de out/: (ruta, firma que cambia si el pipeline lo reescribe,
    mensaje para cuando le faltan columnas porque es de otra version)."""
    ruta = config.OUT / nombre
    if not ruta.exists():
        raise SimuladorNoListo(f"Falta out/{nombre}. Corre primero:  python run_all.py")
    return ruta, (recursos()["firma"], str(ruta), ruta.stat().st_mtime_ns), (
        f"out/{nombre} es de una version anterior del codigo (faltan columnas de {pide}). "
        "Corre:  python run_all.py --desde score")


def densidad(altitud_m=None, temp_c=None, hr_pct=None, rho=None):
    """Densidad relativa del aire de un lugar (1 = 1.225 kg/m3): la que se da en
    `rho`, o la de esa altitud con su temperatura y humedad. Sin temperatura,
    la atmosfera estandar corregida por config.CLIMA_FACTOR (features.densidad_aire)."""
    if rho is None:
        if altitud_m is None:
            raise ValueError("Para la densidad del aire hace falta la altitud (metros) o la densidad misma")
        rho = features.densidad_aire(altitud_m, temp_c, hr_pct)
    rho = float(rho)
    if not 0.5 < rho < 1.3:
        raise ValueError(f"Densidad relativa del aire imposible: {rho:.3f}")
    return rho


def _beta(tipos, var):
    """Elasticidad de `var` para el tipo de cada renglon; sin medida, 1.0 (igual que features.transportar)."""
    tabla = {p: b[var] for p, b in recursos()["beta"].items() if var in b}
    return pd.to_numeric(tipos.map(tabla), errors="coerce").fillna(1.0).to_numpy(float)


def _un_valor_por_renglon(rho, n, que):
    rho = np.asarray(rho, dtype=float) * np.ones(n)
    if not ((rho > 0.5) & (rho < 1.3)).all():
        raise ValueError(f"Densidad relativa del aire {que} vacia o imposible")
    return rho


# ── de lo que captura el usuario a la fila del modelo ───────────────────────
def _captura(t):
    """La tabla que captura el usuario, revisada: numeros como numeros, columnas
    opcionales presentes (vacias si no se dieron) y la recta de referencia
    resuelta donde el pitcheo es su PROPIA recta primaria."""
    tipos = recursos()["defaults"]
    t = t.reset_index(drop=True)
    pt = t["pt"]
    if not pt.isin(list(tipos)).all():
        raise ValueError(f"Tipo de pitcheo desconocido o sin pitcheos en los datos: "
                         f"{sorted(set(pt[~pt.isin(list(tipos))].astype(str)))}. Validos: {list(tipos)}")
    num = lambda col: (pd.to_numeric(t[col], errors="coerce").astype(float) if col in t
                       else pd.Series(np.nan, index=t.index))
    c = pd.DataFrame({"pt": pt, **{dato: num(dato) for dato in FISICA.values()}})
    vacios = [dato for dato in FISICA.values() if not np.isfinite(c[dato]).all()]
    if vacios:
        raise ValueError(f"Datos del pitcheo vacios o que no son numeros: {vacios}")
    c["zurdo"] = num("zurdo").fillna(0) != 0
    # Una recta sin otra referencia es su propia recta primaria (diferenciales en cero) SOLO si ese tipo
    # suele ser la primaria de quien lo lanza (FF, SI): el cutter por lo general acompana a otra recta, y
    # tratarlo como primaria lo saca de lo que vio el modelo.
    fb_tipo = (t["fb_tipo"] if "fb_tipo" in t else pd.Series(None, index=t.index)).astype(object)
    comun = pt.map({p: d.get("fb_type") or "FF" for p, d in tipos.items()}).astype(object)
    dada = {k: num(k) for k in ("fb_velo", "fb_ivb", "fb_hb_brazo")}
    propia = ((fb_tipo == pt) | (fb_tipo.isna() & (comun == pt))) & pt.isin(FASTBALLS) \
        & dada["fb_velo"].isna() & dada["fb_ivb"].isna() & dada["fb_hb_brazo"].isna()
    c["fb_tipo"] = np.where(propia, pt.astype(object), np.where(fb_tipo.notna(), fb_tipo, comun))
    if not c["fb_tipo"].isin(FASTBALLS).all():
        raise ValueError(f"La recta de referencia debe ser una de {list(FASTBALLS)}")
    for k, propio in (("fb_velo", "velo"), ("fb_ivb", "ivb"), ("fb_hb_brazo", "hb_brazo")):
        c[k] = dada[k].mask(propia, c[propio])
    c["fb_rel_h"], c["fb_rel_s_brazo"] = num("fb_rel_h"), num("fb_rel_s_brazo")
    # punto de salida PROMEDIO del tipo (para el tunel); vacio = el renglon ya es el promedio de ese tipo
    c["rel_h_tipo"], c["rel_s_tipo"] = num("rel_h_tipo"), num("rel_s_tipo")
    return c


def tunel_tipico():
    """Tunel mediano (pulgadas) de cada tipo de pitcheo cuando NO es la recta
    primaria de quien lo lanza: lo que se supone cuando no se conoce el punto
    de salida de la recta. La mediana de sim_defaults.json no sirve para una
    recta: cuenta tambien a las primarias, que valen 0 (para la FF y el sinker
    son mas de la mitad). Sale de out/scores_pitch.parquet; {tipo: pulgadas}."""
    ruta, firma, viejo = _de_salida("scores_pitch.parquet", "tunel")
    if _CACHE.get("tunel", (None,))[0] != firma:
        try:
            t = pd.read_parquet(ruta, columns=["pt", "fb_type", "tunnel_dist"]).dropna()
        except (KeyError, ValueError) as e:
            raise SimuladorNoListo(viejo) from e
        t = t[t["pt"].astype(str) != t["fb_type"].astype(str)]
        _CACHE["tunel"] = (firma, t["tunnel_dist"].astype(float).groupby(t["pt"].astype(str)).median().to_dict())
    return _CACHE["tunel"][1]


def filas_pitcheo(t, rho_medicion):
    """fila_pitcheo para una tabla: un renglon de `t` por pitcheo, con las
    columnas pt, velo, ivb, hb_brazo, spin, rel_h, rel_s_brazo, ext y, si se
    conocen, zurdo, fb_tipo, fb_velo, fb_ivb, fb_hb_brazo, fb_rel_h,
    fb_rel_s_brazo, rel_h_tipo, rel_s_tipo (vacio = no se dio). rho_medicion:
    densidad relativa del aire donde se midio (un numero, o uno por renglon).
    Ver fila_pitcheo."""
    R, c = recursos(), _captura(t)
    rho = _un_valor_por_renglon(rho_medicion, len(c), "de la medicion")
    tip = lambda col: c["pt"].map({p: d.get(col) for p, d in R["defaults"].items()})   # mediana del tipo del renglon
    f = pd.DataFrame({"pt": c["pt"], **{col: c[dato] for col, dato in FISICA.items()}})
    f["spin_rate"] = f["spin_rate"].clip(lower=1.0)             # el movimiento por rpm divide entre el spin
    # la recta dada se guarda en la densidad de referencia con la elasticidad de SU tipo (como features.py);
    # la que no se dio es la recta tipica de quienes lanzan ese pitcheo (models/sim_defaults.json)
    a_ref = lambda var: (RHO_REF / rho) ** _beta(c["fb_tipo"], var)
    f["fb_type"] = c["fb_tipo"]
    f["fb_velo"] = c["fb_velo"].fillna(tip("fb_velo"))
    f["fb_ivb0"] = (c["fb_ivb"] * a_ref("ivb")).fillna(tip("fb_ivb0"))
    f["fb_hb0"] = (c["fb_hb_brazo"] * a_ref("hb_n")).fillna(tip("fb_hb0"))
    f["velo_diff_vs_fb"] = f["velocity"] - f["fb_velo"]
    # tunel: rasgo del REPERTORIO, como en features._arsenal: distancia entre el punto de salida PROMEDIO del
    # tipo y el de la recta primaria, y 0 cuando el pitcheo ES la recta primaria. Si el renglon es UN pitcheo,
    # el promedio de su tipo va en rel_h_tipo / rel_s_tipo (sin ellos, el renglon se toma como ese promedio).
    # Sin el punto de salida de la recta: el tunel mediano del tipo entre quienes no lo tienen de primaria.
    tunel = 12 * np.hypot(c["rel_h_tipo"].fillna(c["rel_h"]) - c["fb_rel_h"],
                          c["rel_s_tipo"].fillna(c["rel_s_brazo"]) - c["fb_rel_s_brazo"])
    primaria = c["fb_tipo"] == c["pt"]
    if (tunel.isna() & ~primaria).any():
        tunel = tunel.fillna(pd.to_numeric(c["pt"].map(tunel_tipico()), errors="coerce")).fillna(tip("tunnel_dist"))
    f["tunnel_dist"] = tunel.mask(primaria, 0.0)
    f["is_fastball"] = c["pt"].isin(FASTBALLS).astype(int)
    f["lefty"] = c["zurdo"].astype(int)
    f["air_density"] = rho
    # "transportar" a su propio aire no mueve nada y calcula, con el mismo codigo de las tablas, la recta
    # en ese aire, los diferenciales y el movimiento total y por rpm
    f = features.transportar(f, rho, R["beta"])
    for col in R["modelo"].feats_:                              # variables de un modelo anterior: mediana del tipo
        if col not in f and tip(col).notna().all():
            f[col] = tip(col)
    return f


def fila_pitcheo(pt, velo, ivb, hb_brazo, spin, rel_h, rel_s_brazo, ext, zurdo=False, fb_tipo=None,
                 fb_velo=None, fb_ivb=None, fb_hb_brazo=None, fb_rel_h=None, fb_rel_s_brazo=None,
                 altitud_medicion=None, rho_medicion=None, temp_medicion=None, hr_medicion=None,
                 rel_h_tipo=None, rel_s_tipo=None):
    """Una fila con las variables del modelo Stuff: UN pitcheo, tal como se midio.

    pt                      FF, SI, FC, SL, CU, CH o FS
    velo, spin              mph y rpm al soltar
    ivb, hb_brazo           quiebre vertical inducido y horizontal (pulgadas);
                            horizontal POSITIVO = hacia el lado del brazo
    rel_h, rel_s_brazo, ext altura, lado (positivo = lado del brazo) y extension
                            del punto de salida (pies)
    zurdo                   solo cambia lefty; los signos NO se voltean
    fb_tipo, fb_velo, fb_ivb, fb_hb_brazo   recta primaria del pitcher (FF, SI o
                            FC) en PROMEDIO, medida en el MISMO lugar. Sin ella,
                            una FF o SI es su propia referencia (tambien una FC
                            si se pasa fb_tipo="FC"); cualquier otro pitcheo usa
                            la recta tipica de quienes lo lanzan
    fb_rel_h, fb_rel_s_brazo  punto de salida PROMEDIO de esa recta (tunel); sin
                            ellos, el tunel mediano del tipo. La recta primaria
                            siempre lleva tunel 0
    altitud_medicion / rho_medicion   DONDE se midio (metros, o la densidad
                            relativa del aire). Obligatorio: no hay valor por omision
    temp_medicion, hr_medicion        clima de la medicion (C, %); opcional
    rel_h_tipo, rel_s_tipo  punto de salida PROMEDIO de este tipo de pitcheo del
                            pitcher (el tunel se mide entre promedios); sin
                            ellos se toma el de este pitcheo"""
    if altitud_medicion is None and rho_medicion is None:
        raise ValueError("Falta decir donde se midio el pitcheo (altitud_medicion en metros, o rho_medicion): "
                         "el mismo quiebre vale distinto segun el aire")
    rho = densidad(altitud_medicion, temp_medicion, hr_medicion, rho_medicion)
    return filas_pitcheo(pd.DataFrame([dict(
        pt=pt, velo=velo, ivb=ivb, hb_brazo=hb_brazo, spin=spin, rel_h=rel_h, rel_s_brazo=rel_s_brazo, ext=ext,
        zurdo=zurdo, fb_tipo=fb_tipo, fb_velo=fb_velo, fb_ivb=fb_ivb, fb_hb_brazo=fb_hb_brazo, fb_rel_h=fb_rel_h,
        fb_rel_s_brazo=fb_rel_s_brazo, rel_h_tipo=rel_h_tipo, rel_s_tipo=rel_s_tipo)]), rho)


# ── calificacion de pitcheos concretos ──────────────────────────────────────
def calificar(filas, rho_destino):
    """Stuff+ de una o varias filas de fila_pitcheo llevadas a la densidad
    destino (un numero, o uno por fila). Una fila de salida por pitcheo:
      stuff_plus   100 = pitcheo promedio de la liga en ESE aire; 10 puntos = una
                   desviacion estandar entre pitcheos EN LAS TABLAS (scale.json
                   "stuff", de los modelos de la validacion cruzada; la de este
                   modelo sobre los mismos pitcheos esta en scale.json "simulador")
      score        -xRV (carreras por pitcheo; mas alto = mejor para el pitcher)
      percentil    lugar de ese Stuff+ entre los PITCHEOS de la liga (0 a 100)
      densidad, ivb, hb       el aire destino y el quiebre que tendria ahi
      p_<clase>    probabilidad de cada uno de los 10 resultados
      p_swing, whiff_por_swing, rodado_por_batazo, duro_por_batazo
                   (duro = linea o elevado de 95+ mph, por batazo en juego)
    El modelo Stuff no ve ubicacion ni cuenta: son promedios sobre ambas."""
    R = recursos()
    m, rej = R["modelo"], R["rejilla"]
    rho = np.asarray(rho_destino, dtype=float) * np.ones(len(filas))
    if not (np.isfinite(rho).all() and rho.min() > rej[0, 0] - MARGEN_REJILLA
            and rho.max() < rej[-1, 0] + MARGEN_REJILLA):
        raise ValueError(f"Densidad destino fuera de lo que cubre la escala del simulador "
                         f"({rej[0, 0]:.2f} a {rej[-1, 0]:.2f}): {np.unique(rho.round(3)).tolist()}")
    d = features.transportar(filas, rho, R["beta"])
    faltan = [f for f in m.feats_ if f not in d or d[f].isna().any()]
    if faltan:
        raise ValueError(f"A la fila le faltan variables del modelo Stuff: {faltan}. Armala con fila_pitcheo()")
    P = proba(m, d).astype(float)
    score = -xrv(P, R["valores"], rho=rho)
    plus = 100.0 + PLUS_SCALE * (score - np.interp(rho, rej[:, 0], rej[:, 1])) / R["escala"]["stuff"]["sd"]
    pct = R["escala"].get("simulador", {}).get("percentiles_plus")
    out = pd.DataFrame({"stuff_plus": plus, "score": score,
                        "percentil": np.interp(plus, pct, np.arange(len(pct))) if pct else np.nan,
                        "densidad": rho, "ivb": d["ivb"].to_numpy(float), "hb": d["hb_n"].to_numpy(float)},
                       index=filas.index)
    for c, i in I.items():
        out[f"p_{c}"] = P[:, i]
    swing, batazo = P[:, SWING].sum(1), P[:, IP].sum(1)
    out["p_swing"] = swing
    out["whiff_por_swing"] = P[:, I["whiff"]] / swing
    out["rodado_por_batazo"] = P[:, I["ip_gb"]] / batazo
    out["duro_por_batazo"] = (P[:, I["ip_ld"]] + P[:, I["ip_fb_hard"]]) / batazo
    return out


def evaluar(fila, rho_destino):
    """Un pitcheo en la densidad destino: dict con las columnas de calificar().
    Para pasar de altitud y clima a densidad: densidad(altitud, temp, humedad)."""
    return {k: float(v) for k, v in calificar(fila.iloc[:1], rho_destino).iloc[0].items()}


def comparar(fila, rho_cdmx=RHO_CDMX, rho_mar=RHO_MAR):
    """El mismo pitcheo en el Harp Helu y al nivel del mar (por omision, las
    densidades de referencia de config.py, RHO_CDMX y RHO_MAR: las de las tablas).
      delta_stuff_plus   RELATIVO: cuanto sube o baja frente a los demas
                         pitcheos al pasar del mar al Harp Helu (puntos por pitcheo)
      costo_cdmx_rv100   ABSOLUTO: carreras por 100 pitcheos que cuesta lanzarlo
                         en el Harp Helu; incluye el efecto de parque, comun a todos"""
    cdmx, mar = evaluar(fila, rho_cdmx), evaluar(fila, rho_mar)
    return {"harp_helu": cdmx, "nivel_del_mar": mar,
            "delta_stuff_plus": cdmx["stuff_plus"] - mar["stuff_plus"],
            "costo_cdmx_rv100": 100.0 * (mar["score"] - cdmx["score"])}


# ── calificacion de PROMEDIOS (lo que se teclea en el dashboard) ────────────
def dispersion(n=N_NUBE):
    """Como repite un pitcher real un mismo pitcheo: por tipo, N_NUBE
    desviaciones reales (pitcheo menos el promedio de su arsenal, en arsenales
    de MIN_PITCHES_ARSENAL pitcheos o mas) de velocidad, quiebre, spin y punto
    de salida, centradas en cero. El quiebre va en la densidad media de la liga.
    Sale de out/scores_pitch.parquet; {tipo: matriz (n, 7)} en el orden de
    FISICA. Con n < N_NUBE, las primeras n de esas mismas, vueltas a centrar."""
    if not 1 <= n <= N_NUBE:
        raise ValueError(f"Las repeticiones de un promedio van de 1 a {N_NUBE} (N_NUBE): {n}")
    cols = [{"ivb": "ivb_neutral", "hb_n": "hb_neutral"}.get(col, col) for col in FISICA]
    ruta, firma, viejo = _de_salida("scores_pitch.parquet", "quiebre transportado")
    if _CACHE.get("dispersion", (None,))[0] != firma:
        try:
            t = pd.read_parquet(ruta, columns=["pitcher_id", "season", "pt", *cols]).dropna()
        except (KeyError, ValueError) as e:
            raise SimuladorNoListo(viejo) from e
        g = t.groupby(["pitcher_id", "season", "pt"])
        desvio = (t[cols].astype(float) - g[cols].transform("mean").astype(float))[
            g[cols[0]].transform("size") >= MIN_PITCHES_ARSENAL]
        nube = {}
        for pt, d in desvio.groupby(t["pt"]):
            d = d.sample(N_NUBE, replace=len(d) < N_NUBE, random_state=RANDOM_STATE).to_numpy()
            nube[pt] = d - d.mean(0)
        _CACHE["dispersion"] = (firma, nube)
    nube = _CACHE["dispersion"][1]
    return nube if n == N_NUBE else {pt: d[:n] - d[:n].mean(0) for pt, d in nube.items()}


def repeticiones(t, rho_medicion, n=N_NUBE):
    """Cada renglon de `t` es el PROMEDIO de un pitcheo (columnas de
    filas_pitcheo); se convierte en `n` pitcheos: ese promedio mas la
    dispersion real de su tipo (el quiebre se dispersa en proporcion al aire,
    como el quiebre mismo). La recta de referencia NO se dispersa: es un
    promedio. El tunel tampoco: es un rasgo del repertorio, asi que las `n`
    repeticiones llevan el mismo, el del punto de salida PROMEDIO tecleado. Un
    tipo sin arsenales grandes no tiene dispersion medida: sus repeticiones son
    iguales. Devuelve (pitcheos, renglon de origen, densidad)."""
    c = _captura(t)
    if not len(c):
        raise ValueError("No hay ningun pitcheo que calificar")
    rho = _un_valor_por_renglon(rho_medicion, len(c), "de la medicion")
    nube, quieto = dispersion(n), np.zeros((n, len(FISICA)))
    # lo tecleado ES el promedio del tipo: se guarda antes de dispersar, y de ahi sale el tunel de todas
    c["rel_h_tipo"], c["rel_s_tipo"] = c["rel_h_tipo"].fillna(c["rel_h"]), c["rel_s_tipo"].fillna(c["rel_s_brazo"])
    origen = np.repeat(np.arange(len(c)), n)
    x = c.iloc[origen].reset_index(drop=True)
    desvio = np.vstack([nube.get(p, quieto) for p in c["pt"]])
    a_su_aire = rho[origen] / recursos()["valores"]["densidad"]["rho_liga"]
    for j, (col, dato) in enumerate(FISICA.items()):
        x[dato] = x[dato] + desvio[:, j] * (a_su_aire ** _beta(x["pt"], col) if col in ("ivb", "hb_n") else 1.0)
    return x, origen, rho[origen]


def calificar_promedio(t, rho_medicion, *destinos, n=N_NUBE):
    """Stuff+ del PROMEDIO de un pitcheo en cada densidad destino: una tabla por
    destino (cada uno, un numero) con las columnas de calificar() menos
    percentil, promediadas sobre las `n` repeticiones de cada renglon de `t`.
    stuff_plus sigue en puntos por pitcheo. Cada tabla trae ademas el error de
    simulacion (error estandar del promedio de las repeticiones: cuanto cambiaria
    el numero con otras `n` desviaciones; NO es el error del modelo):
      stuff_plus_ee   del stuff_plus de ese destino
      dif_plus_ee     de la diferencia de stuff_plus entre el PRIMER destino y este
      dif_score_ee    de la diferencia de score entre el PRIMER destino y este"""
    x, origen, rho = repeticiones(t, rho_medicion, n)
    filas = filas_pitcheo(x, rho)
    reps = [calificar(filas, d).drop(columns="percentil") for d in destinos]
    ee = lambda v: v.groupby(origen).std() / np.sqrt(n)
    return [r.groupby(origen).mean().assign(stuff_plus_ee=ee(r["stuff_plus"]),
                                           dif_plus_ee=ee(reps[0]["stuff_plus"] - r["stuff_plus"]),
                                           dif_score_ee=ee(reps[0]["score"] - r["score"])) for r in reps]


def tipico(pt, rho):
    """Pitcheo tipico de la liga de ese tipo (medianas de sim_defaults.json) con
    el quiebre que tendria en la densidad `rho`, y la recta primaria tipica de
    quienes lo lanzan: el punto de partida de los controles del dashboard. Las
    medianas se toman como medidas en la densidad media de la liga."""
    R = recursos()
    t, beta = R["defaults"][pt], R["beta"]
    fb = t.get("fb_type") or "FF"
    r = rho / R["valores"]["densidad"]["rho_liga"]
    return {"velo": t["velocity"], "ivb": t["ivb"] * r ** beta.get(pt, {}).get("ivb", 1.0),
            "hb": t["hb_n"] * r ** beta.get(pt, {}).get("hb_n", 1.0), "spin": t["spin_rate"],
            "rel_h": t["release_height"], "rel_s": t["rel_side_n"], "ext": t["extension"],
            "fb_tipo": fb, "fb_velo": t["fb_velo"],
            "fb_ivb": t["fb_ivb0"] * (rho / RHO_REF) ** beta.get(fb, {}).get("ivb", 1.0),
            "fb_hb": t["fb_hb0"] * (rho / RHO_REF) ** beta.get(fb, {}).get("hb_n", 1.0)}


# ── arsenal completo (prospecto) y su lugar en la liga ──────────────────────
def _con_recta_del_arsenal(t, g, con_salida=False):
    """Pone como recta de referencia de cada pitcheo la recta (FF, SI o FC) de
    mas uso de su arsenal `g`, como en features.py. Un arsenal sin recta queda
    vacio: filas_pitcheo usa entonces la recta tipica de quienes lanzan cada
    pitcheo. con_salida: el punto de salida de cada renglon es el de ESE tipo;
    se pasa tambien el de la recta y el tunel sale de los dos. Si no (un solo
    punto de salida para todo el arsenal), queda el tunel mediano del tipo."""
    rectas = (t.assign(_g=g)[t["pt"].isin(FASTBALLS)].sort_values("uso", ascending=False, kind="stable")
              .drop_duplicates("_g").set_index("_g"))
    pares = [("fb_tipo", "pt"), ("fb_velo", "velo"), ("fb_ivb", "ivb"), ("fb_hb_brazo", "hb_brazo")]
    if con_salida:
        pares += [("fb_rel_h", "rel_h"), ("fb_rel_s_brazo", "rel_s_brazo")]
    return t.assign(**{k: g.map(rectas[col]) for k, col in pares})


def arsenales(t, rho_medicion, grupo=(), rho_cdmx=RHO_CDMX, rho_mar=RHO_MAR, salida_por_tipo=False, n=N_NUBE):
    """Arsenales completos tecleados: `t` trae un renglon por tipo de pitcheo
    con el PROMEDIO de ese pitcheo (columnas de filas_pitcheo) mas `uso` (en
    cualquier escala: se normaliza dentro del arsenal). `grupo` = columnas que
    separan un arsenal de otro (vacio: toda la tabla es un solo arsenal).
    salida_por_tipo: True si rel_h / rel_s_brazo de cada renglon son los de ESE
    tipo (el tunel sale de ellos y de los de la recta del arsenal); False si
    se capturo un solo punto de salida para todo el arsenal (tunel mediano de
    cada tipo: con un solo punto valdria 0 para todos, y ningun pitcher lanza
    asi). `n` = repeticiones por pitcheo.
    Devuelve (por_pitcheo, por_arsenal). Por arsenal, pesando por uso:
      score_cdmx / score_mar   -xRV medio (carreras por pitcheo)
      delta                    cambio relativo, en puntos por pitcheo
      costo_cdmx_rv100         costo absoluto de lanzar en el Harp Helu
      delta_ee, costo_ee       error de simulacion de esos dos (ver calificar_promedio)"""
    t = t.reset_index(drop=True)
    g = t.groupby(list(grupo), sort=False).ngroup() if len(grupo) else pd.Series(0, index=t.index)
    t = t.assign(uso=pd.to_numeric(t["uso"], errors="coerce"))
    if not (t["uso"] > 0).all():
        raise ValueError("El uso de cada pitcheo debe ser un numero mayor que cero")
    con = _con_recta_del_arsenal(t, g, salida_por_tipo)
    c, m = calificar_promedio(con, rho_medicion, rho_cdmx, rho_mar, n=n)
    pp = t[[*grupo, "pt"]].assign(
        peso=t["uso"] / t["uso"].groupby(g).transform("sum"),
        recta_ref=np.where(con["fb_tipo"].notna(), con["fb_tipo"], "tipica"),
        stuff_cdmx=c["stuff_plus"], stuff_mar=m["stuff_plus"], delta=c["stuff_plus"] - m["stuff_plus"],
        costo_cdmx_rv100=100.0 * (m["score"] - c["score"]), ivb_cdmx=c["ivb"], hb_cdmx=c["hb"],
        ivb_mar=m["ivb"], hb_mar=m["hb"], score_cdmx=c["score"], score_mar=m["score"],
        whiff_por_swing=c["whiff_por_swing"], rodado_por_batazo=c["rodado_por_batazo"],
        delta_ee=m["dif_plus_ee"], costo_ee=100.0 * m["dif_score_ee"])
    pa = pp[["score_cdmx", "score_mar", "delta", "costo_cdmx_rv100"]].mul(pp["peso"], axis=0).groupby(g).sum()
    for col in ("delta_ee", "costo_ee"):    # las repeticiones de un tipo no dependen de las de otro: en cuadratura
        pa[col] = np.sqrt(((pp[col] * pp["peso"]) ** 2).groupby(g).sum())
    if len(grupo):
        pa = pd.concat([t.groupby(g)[list(grupo)].first(), pa], axis=1)
    return pp, pa.reset_index(drop=True)


def _arsenales_de_la_liga():
    """out/scores_arsenal.csv como lo teclearia un usuario: un renglon por pitcher-temporada-tipo con las
    columnas de filas_pitcheo (el quiebre, en el aire del Harp Helu) mas uso y total_temporada."""
    pide = ["pitcher_id", "season", "pt", "uso", "total_temporada", "lefty", "velo", "ivb_cdmx", "hb_cdmx",
            "spin", "rel_h", "rel_s", "ext"]
    ruta, _, viejo = _de_salida("scores_arsenal.csv", "arsenales")
    a = pd.read_csv(ruta, dtype={"pitcher_id": str})
    if [c for c in pide if c not in a]:
        raise SimuladorNoListo(viejo)
    return (a[pide].dropna().rename(columns={"ivb_cdmx": "ivb", "hb_cdmx": "hb_brazo", "rel_s": "rel_s_brazo",
                                             "lefty": "zurdo"}).reset_index(drop=True))


def liga_tecleada():
    """Los arsenales REALES de la liga calificados por el mismo camino que un
    prospecto: de out/scores_arsenal.csv se toma el promedio de cada tipo de
    cada pitcher-temporada (con el quiebre en el aire del Harp Helu) y se
    califica en el Harp Helu como si se hubiera tecleado. Es la referencia para
    ubicar un pitcheo o un arsenal tecleado: mismo modelo y mismo camino en los
    dos lados (con N_NUBE_LIGA repeticiones por arsenal y con el tunel real de
    cada uno: la tabla trae el punto de salida de cada tipo). Devuelve
    (por_pitcheo, por_pitcher):
      por_pitcheo   todos los arsenales de la tabla: pitcher_id, season, pt, score_cdmx
      por_pitcher   los pitcher-temporada de MIN_PITCHES_PITCHER pitcheos o mas
                    (los del Ranking), cada uno representado por sus arsenales
                    listados: pitcher_id, season, score_cdmx pesado por uso,
                    pitcheos y cobertura (parte de sus pitcheos que esta en
                    esos arsenales)"""
    firma = _de_salida("scores_arsenal.csv", "arsenales")[1]
    if _CACHE.get("liga", (None,))[0] != firma:
        t = _arsenales_de_la_liga()
        pp = t[["pitcher_id", "season", "pt"]].assign(score_cdmx=np.nan)
        pit = pd.DataFrame(columns=["pitcher_id", "season", "score_cdmx", "pitcheos", "cobertura"])
        if len(t):
            g = t.groupby([t["pitcher_id"], t["season"]], sort=False).ngroup()
            pp["score_cdmx"] = calificar_promedio(_con_recta_del_arsenal(t, g, con_salida=True), RHO_CDMX, RHO_CDMX,
                                                  n=N_NUBE_LIGA)[0]["score"]
            cubre = t["uso"].groupby(g).transform("sum")
            entra = t["total_temporada"] >= MIN_PITCHES_PITCHER
            pit = pd.DataFrame({"pitcher_id": t["pitcher_id"][entra].groupby(g).first(),
                                "season": t["season"][entra].groupby(g).first(),
                                "score_cdmx": (pp["score_cdmx"] * t["uso"] / cubre)[entra].groupby(g).sum(),
                                "pitcheos": t["total_temporada"][entra].groupby(g).first(),
                                "cobertura": cubre[entra].groupby(g).first()}).reset_index(drop=True)
        _CACHE["liga"] = (firma, (pp, pit))
    return _CACHE["liga"][1]


def _plus_entre_pitchers(score_cdmx, liga):
    """score en el Harp Helu -> puntos entre pitchers: media y desviacion de la liga tecleada, pesadas por pitcheos."""
    x, w = liga["score_cdmx"].to_numpy(float), liga["pitcheos"].to_numpy(float)
    mu = np.average(x, weights=w)
    return 100.0 + PLUS_SCALE * (score_cdmx - mu) / np.sqrt(np.average((x - mu) ** 2, weights=w))


def entre_pitchers(score_cdmx):
    """Un arsenal tecleado en la escala ENTRE PITCHERS: (plus, percentil, n) de
    su score en el Harp Helu contra los n pitcher-temporada de la liga
    calificados igual (media y desviacion pesadas por pitcheos, como en el
    Ranking). NaN con menos de 5."""
    liga = liga_tecleada()[1]
    if len(liga) < 5:
        return np.nan, np.nan, int(len(liga))
    x = liga["score_cdmx"].to_numpy(float)
    return float(_plus_entre_pitchers(score_cdmx, liga)), float(100.0 * np.mean(x <= score_cdmx)), int(len(x))


def contra_el_ranking():
    """Que tanto coincide el numero "entre pitchers" de un arsenal tecleado con
    el del Ranking, medido en esta corrida: los pitcher-temporada de la liga
    pasados por el camino del Prospecto (liga_tecleada) contra su Stuff+ en el
    Harp Helu de out/scores_pitcher.csv (stuff_cdmx_plus_enc, el que ordena la
    pestana Ranking; sale de los modelos de la validacion cruzada).
    {"n", "r", "dif_abs"}: pitcher-temporadas, correlacion y diferencia absoluta
    media en puntos entre pitchers; NaN con menos de 5. Con este numero decide
    el dashboard si puede llamar comparables a los dos."""
    liga = liga_tecleada()[1]
    ruta, firma, viejo = _de_salida("scores_pitcher.csv", "pitchers")
    if _CACHE.get("ranking", (None,))[0] != firma:
        j = pd.read_csv(ruta, dtype={"pitcher_id": str})
        if [c for c in ("pitcher_id", "season", "stuff_cdmx_plus_enc") if c not in j]:
            raise SimuladorNoListo(viejo)
        r = {"n": 0, "r": np.nan, "dif_abs": np.nan}
        if len(liga) >= 5:
            m = liga.assign(sim=_plus_entre_pitchers(liga["score_cdmx"].astype(float), liga)).merge(
                j[["pitcher_id", "season", "stuff_cdmx_plus_enc"]].dropna(), on=["pitcher_id", "season"])
            if len(m) >= 5:
                r = {"n": int(len(m)), "r": float(m["sim"].corr(m["stuff_cdmx_plus_enc"])),
                     "dif_abs": float((m["sim"] - m["stuff_cdmx_plus_enc"]).abs().mean())}
        _CACHE["ranking"] = (firma, r)
    return _CACHE["ranking"][1]


def entre_los_de_su_tipo(pt, score_cdmx):
    """(percentil, n) del promedio de un pitcheo en el Harp Helu entre los n
    arsenales de la liga de ese tipo, calificados igual. NaN con menos de 5."""
    liga = liga_tecleada()[0]
    x = liga.loc[liga["pt"] == pt, "score_cdmx"].to_numpy(float)
    return (float(100.0 * np.mean(x <= score_cdmx)) if len(x) >= 5 else np.nan), int(len(x))


# ── prueba de regresion (python simulator.py) ───────────────────────────────
N_POR_TIPO = 150           # 7 tipos: ~1,000 pitcheos reales
N_ARSENALES = 300          # arsenales reales con que se prueba calificar_promedio
R_MINIMA = 0.95            # correlacion y correlacion de rangos que debe dar la reconstruccion
DIF_MAXIMA = 1.0           # y diferencia absoluta media maxima, en puntos por pitcheo
MAX_CAMBIO_MANO = 3.0      # puntos que puede mover, en promedio, cambiar solo la mano
R_PROMEDIOS, SESGO_PROMEDIOS = 0.90, 1.0     # debajo de esto calificar_promedio amerita un aviso
EXACTO = 1e-9              # dos caminos que deben dar el mismo numero (cableado)
N_LIGA_PRUEBA = 40         # pitcher-temporadas de la liga pasados por arsenales()


def _reales():
    """Pitcheos reales de out/scores_pitch.parquet que tienen recta de referencia y todos los datos, con el
    punto de salida PROMEDIO de su tipo (pitcher-temporada-tipo): con el se arma el tunel, como en features.py."""
    ruta = _de_salida("scores_pitch.parquet", "scores")[0]
    cols = dict.fromkeys(recursos()["modelo"].feats_ + [
        "pitcher_id", "season", "pt", "lefty", "air_density", "fb_type", "fb_velo", "fb_ivb0", "fb_hb0", "fb_ivb",
        "fb_hb", "fb_rel_h", "fb_rel_s", "ivb_cdmx", "hb_cdmx", "stuff_cdmx_plus"])
    t = pd.read_parquet(ruta, columns=list(cols))
    medio = t.groupby(["pitcher_id", "season", "pt"])[["release_height", "rel_side_n"]].transform("mean")
    t = t.assign(rel_h_tipo=medio["release_height"].astype(float), rel_s_tipo=medio["rel_side_n"].astype(float))
    return t[t.notna().all(axis=1)].reset_index(drop=True)


def _reconstruir(t, recta=True, tunel=True, velo=0.0):
    """Los pitcheos reales pasados por filas_pitcheo. recta / tunel = False no
    pasan la recta del pitcher o los puntos de salida promedio (el del tipo y
    el de la recta), como cuando el usuario no los captura; `velo` suma mph a
    la velocidad (barrido)."""
    datos = pd.DataFrame({dato: t[col] for col, dato in FISICA.items()}).assign(pt=t["pt"], zurdo=t["lefty"])
    datos["velo"] = datos["velo"] + velo
    if recta:
        datos = datos.assign(fb_tipo=t["fb_type"], fb_velo=t["fb_velo"], fb_ivb=t["fb_ivb"], fb_hb_brazo=t["fb_hb"])
    if recta and tunel:
        datos = datos.assign(fb_rel_h=t["fb_rel_h"], fb_rel_s_brazo=t["fb_rel_s"],
                             rel_h_tipo=t["rel_h_tipo"], rel_s_tipo=t["rel_s_tipo"])
    return filas_pitcheo(datos, t["air_density"].to_numpy(float))


def _tablas(t, rho):
    """El camino de las tablas: el mismo modelo sobre las filas REALES llevadas a rho."""
    R = recursos()
    score = -xrv(proba(R["modelo"], features.transportar(t, rho, R["beta"])), R["valores"], rho=rho)
    return 100.0 + PLUS_SCALE * (score - np.interp(rho, *R["rejilla"].T)) / R["escala"]["stuff"]["sd"]


def _parecido(a, b):
    """(correlacion, correlacion de rangos, diferencia media, diferencia absoluta media, maxima)."""
    a, b = pd.Series(np.asarray(a, dtype=float)), pd.Series(np.asarray(b, dtype=float))
    return a.corr(b), a.rank().corr(b.rank()), (a - b).mean(), (a - b).abs().mean(), (a - b).abs().max()


def _prueba_promedios(todos):
    """calificar_promedio contra la realidad: el promedio de cada arsenal real,
    tecleado (con su recta y los dos puntos de salida promedio), contra el
    Stuff+ medio de sus pitcheos reales (mismo modelo). Devuelve
    (comprobaciones, aviso): la lista de (nombre, paso) y el texto del aviso
    cuando calificar_promedio queda debajo de lo que se espera de el."""
    llave = [todos["pitcher_id"], todos["season"], todos["pt"]]
    todos = todos.assign(arsenal=todos.groupby(llave).ngroup(), n=todos.groupby(llave)["pt"].transform("size"))
    grandes = pd.Series(todos.loc[todos["n"] >= MIN_PITCHES_ARSENAL, "arsenal"].unique())
    elegidos = grandes.sample(min(N_ARSENALES, len(grandes)), random_state=RANDOM_STATE)
    t = todos[todos["arsenal"].isin(elegidos)].reset_index(drop=True)
    if t["arsenal"].nunique() < 5:
        print(f"   sin datos: menos de 5 arsenales con {MIN_PITCHES_ARSENAL} pitcheos")
        return [], None
    t["real"] = _tablas(t, RHO_CDMX)
    al_harp = lambda col, var: t[col] * (RHO_CDMX / RHO_REF) ** _beta(t["fb_type"], var)
    a = t.assign(fb_ivb=al_harp("fb_ivb0", "ivb"), fb_hb_brazo=al_harp("fb_hb0", "hb_n")).groupby("arsenal").agg(
        pt=("pt", "first"), zurdo=("lefty", "first"), velo=("velocity", "mean"), ivb=("ivb_cdmx", "mean"),
        hb_brazo=("hb_cdmx", "mean"), spin=("spin_rate", "mean"), rel_h=("release_height", "mean"),
        rel_s_brazo=("rel_side_n", "mean"), ext=("extension", "mean"), fb_tipo=("fb_type", "first"),
        fb_velo=("fb_velo", "first"), fb_ivb=("fb_ivb", "first"), fb_hb_brazo=("fb_hb_brazo", "first"),
        fb_rel_h=("fb_rel_h", "first"), fb_rel_s_brazo=("fb_rel_s", "first"),
        real=("real", "mean")).reset_index(drop=True)
    nube = calificar_promedio(a, RHO_CDMX, RHO_CDMX)[0]
    a["nube"] = nube["stuff_plus"]
    a["un_pitcheo"] = calificar(filas_pitcheo(a, RHO_CDMX), RHO_CDMX)["stuff_plus"]
    print(f"   {len(a)} arsenales reales con {MIN_PITCHES_ARSENAL}+ pitcheos, Harp Helu, puntos por pitcheo; lo real = "
          "Stuff+ medio de sus pitcheos con el mismo modelo")
    print(f"   {'camino':45s} {'r':>7s} {'r rangos':>9s} {'dif media':>10s} {'|dif| media':>12s}")
    medida = {}
    for nombre, col in ((f"calificar_promedio ({N_NUBE} repeticiones)", "nube"),
                        ("el promedio como si fuera UN pitcheo", "un_pitcheo")):
        r, rs, media, absoluta, _ = medida[col] = _parecido(a[col], a["real"])
        print(f"   {nombre:45s} {r:7.3f} {rs:9.3f} {media:+10.2f} {absoluta:12.2f}")
    g = (a["nube"] - a["real"]).groupby(a["pt"])
    print("   calificar_promedio menos lo real, por tipo: "
          + ", ".join(f"{pt} {v:+.1f} ({int(n)})" for pt, v, n in zip(g.mean().index, g.mean(), g.size())))
    print(f"   error de simulacion de calificar_promedio (no es el error del modelo): mediana "
          f"{nube['stuff_plus_ee'].median():.2f} puntos, maximo {nube['stuff_plus_ee'].max():.2f}")
    # con UNA repeticion la dispersion vale cero: el camino del dashboard debe dar entonces lo mismo que la
    # fila unica, que es la que la seccion 1 compara con las filas reales
    una = calificar_promedio(a, RHO_CDMX, RHO_CDMX, n=1)[0]["stuff_plus"]
    cableado = float((una - a["un_pitcheo"]).abs().max())
    print(f"   a) cableado: con una sola repeticion (dispersion cero) calificar_promedio difiere de la fila unica a "
          f"lo mas {cableado:.1e} puntos")
    (r, _, media, absoluta, _), (r1, _, _, absoluta1, _) = medida["nube"], medida["un_pitcheo"]
    mejor = bool(r > r1 and absoluta < absoluta1)
    print(f"   b) con la dispersion se parece {'MAS' if mejor else 'MENOS'} a lo real que sin ella (r {r:.3f} contra "
          f"{r1:.3f}; |dif| media {absoluta:.2f} contra {absoluta1:.2f})")
    aviso = None
    if not (r >= R_PROMEDIOS and abs(media) <= SESGO_PROMEDIOS):
        aviso = (f"calificar_promedio reproduce los arsenales reales con r = {r:.3f} y sesgo {media:+.2f} (se espera "
                 f"r >= {R_PROMEDIOS} y sesgo de a lo mas {SESGO_PROMEDIOS:.0f} punto): lee con cautela el Simulador "
                 "y el Prospecto del dashboard")
        print("   AVISO: " + aviso)
    return [("promedio tecleado sin dispersion = fila unica (cableado)", cableado < EXACTO),
            ("promedio tecleado: la dispersion acerca a lo real", mejor)], aviso


def _prueba_liga():
    """El Prospecto contra su referencia: pitcher-temporadas de la liga pasados
    por arsenales() deben recibir el score que les da liga_tecleada (mismo
    camino en los dos lados) y entre_pitchers debe ubicarlos. Devuelve la lista
    de (nombre, paso)."""
    liga = liga_tecleada()[1]
    if len(liga) < 5:
        print(f"   sin datos: menos de 5 pitcher-temporadas con {MIN_PITCHES_PITCHER} pitcheos en out/scores_arsenal.csv")
        return []
    t = _arsenales_de_la_liga().merge(liga[["pitcher_id", "season"]].head(N_LIGA_PRUEBA), on=["pitcher_id", "season"])
    pa = arsenales(t, RHO_CDMX, grupo=("pitcher_id", "season"), salida_por_tipo=True, n=N_NUBE_LIGA)[1]
    m = pa.merge(liga, on=["pitcher_id", "season"], suffixes=("", "_liga"))
    dif = float((m["score_cdmx"] - m["score_cdmx_liga"]).abs().max()) if len(m) else np.nan
    lugar = np.array([entre_pitchers(s)[:2] for s in m["score_cdmx"]], dtype=float).reshape(-1, 2)
    ubica = bool(len(m) and np.isfinite(lugar).all() and (lugar[:, 1] >= 0).all() and (lugar[:, 1] <= 100).all())
    print(f"   a) cableado: {len(m)} pitcher-temporadas de la liga pasados por arsenales() difieren de liga_tecleada a lo "
          f"mas {dif:.1e} carreras por pitcheo" + (f"; en puntos entre pitchers van de {lugar[:, 0].min():.0f} a "
                                                  f"{lugar[:, 0].max():.0f}" if ubica else ""))
    c = contra_el_ranking()
    print(f"   Informativo: los {c['n']} pitcher-temporadas de la liga calificados por este camino contra su Stuff+ "
          f"Harp Helu del Ranking\n   (modelos de fold): r = {c['r']:.3f}, |dif| media = {c['dif_abs']:.1f} puntos "
          "entre pitchers. El dashboard lo dice con este numero.")
    return [("arsenal de la liga por arsenales() = liga_tecleada (cableado)", bool(len(m) and dif < EXACTO)),
            ("entre_pitchers ubica esos arsenales", ubica)]


def prueba():
    """Prueba de regresion del simulador y barridos de cordura. Devuelve True si pasa.
    Que prueba: que el simulador arma, para un pitcheo real, la misma fila que
    las tablas (seccion 1), que su interfaz y el camino del dashboard estan bien
    cableados (2e, 3a, 4a) y que la dispersion hace lo que promete (3b). NO
    prueba que el modelo sea bueno (eso es model.py validar) ni que la fisica sea
    cierta (score.py, val_contrafactual.csv): 2b y 2d solo comprueban que aqui se
    aplica la misma formula que en features.py y physics.py."""
    pd.set_option("display.width", 200)
    R, todos = recursos(), _reales()
    orden = todos.sample(frac=1.0, random_state=RANDOM_STATE)
    t = orden[orden.groupby("pt").cumcount() < N_POR_TIPO].reset_index(drop=True)
    m, rho_liga = R["modelo"], R["valores"]["densidad"]["rho_liga"]
    destinos = (("Harp Helu", RHO_CDMX), ("nivel del mar", RHO_MAR), ("densidad de liga", rho_liga))
    print(f"PRUEBA DEL SIMULADOR: {len(t):,} pitcheos reales (hasta {N_POR_TIPO} por tipo; {int(t['lefty'].sum()):,} "
          f"de zurdo), modelo stuff.joblib ({m.iters_} arboles, {m.backend_})")
    ok = []

    print("\n1. UN pitcheo: fila_pitcheo + evaluar contra el mismo modelo sobre las filas reales (puntos por pitcheo)")
    print(f"   {'que se captura':43s} {'destino':17s} {'r':>7s} {'r rangos':>9s} {'dif media':>10s} "
          f"{'|dif| media':>12s} {'|dif| max':>10s}")
    caminos = (("todo: pitcheo, recta y salidas promedio", {}),
               ("pitcheo y recta (tunel mediano del tipo)", {"tunel": False}),
               ("solo el pitcheo (recta tipica del tipo)", {"recta": False}))
    difs = {}
    for k, (nombre, opciones) in enumerate(caminos):
        filas = _reconstruir(t, **opciones)
        if k == 0:
            dif_tunel = float(np.abs(filas["tunnel_dist"].to_numpy(float) - t["tunnel_dist"].to_numpy(float)).max())
        for destino, rho in destinos:
            sim, ref = calificar(filas, rho)["stuff_plus"].to_numpy(), _tablas(t, rho)
            r, rs, media, absoluta, maxima = _parecido(sim, ref)
            print(f"   {nombre:43s} {destino:17s} {r:7.4f} {rs:9.4f} {media:+10.3f} {absoluta:12.3f} {maxima:10.2f}")
            if k == 0:
                ok.append((f"pitcheos reales reconstruidos, {destino}",
                           bool(r > R_MINIMA and rs > R_MINIMA and absoluta < DIF_MAXIMA)))
            if rho == RHO_CDMX:
                difs[k] = sim - ref
    print(f"   Se exige el primer renglon (r y r de rangos > {R_MINIMA}, |dif| media < {DIF_MAXIMA:.0f} punto): con la "
          "recta del pitcher y los dos puntos\n   de salida PROMEDIO (el del tipo y el de la recta), el simulador arma "
          "la misma fila que las tablas, tunel\n   incluido. Los otros dos son informativos: cuanto se mueve el Stuff+ "
          "de UN pitcheo cuando no se conoce el\n   tunel de su repertorio o la recta misma.")
    distintos = int((np.abs(difs[0]) > 1e-6).sum())
    print(f"   El tunel que arma el simulador difiere del de features.py a lo mas {dif_tunel:.1e} pulgadas; el Stuff+ "
          f"sale distinto en {distintos} de\n   {len(t):,} pitcheos."
          + (" Causa probable: scores_pitch.parquet guarda los decimales en 32 bits,\n   el "
             "tunel y los diferenciales recalculados cambian en la sexta cifra, y cuando un corte de arbol cae justo "
             "en el\n   tunel de un repertorio sus pitcheos pasan a la hoja de al lado." if distintos else ""))
    dif = pd.DataFrame({"pt": t["pt"], "mano": np.where(t["lefty"] == 1, "zurdo", "derecho"),
                        "completo": difs[0], "sin_tunel": difs[1], "sin_recta": difs[2]})
    dif = dif.join(dif[["completo", "sin_tunel", "sin_recta"]].abs().add_suffix("_abs"))
    for llave in ("pt", "mano"):
        g = dif.groupby(llave)
        print(f"\n   Harp Helu, simulador menos tablas por {llave}: diferencia media y absoluta media (_abs)")
        print(g[sorted(c for c in dif if c not in ("pt", "mano"))].mean().assign(n=g.size()).round(2).to_string())
    peores = dif["sin_tunel"].abs().nlargest(5).index
    print("\n   Peores casos sin los puntos de salida (el tunel de ese repertorio se aparta del mediano de su tipo):")
    print(t.loc[peores, ["pt", "lefty", "velocity", "ivb", "hb_n", "tunnel_dist"]].assign(
        dif=dif.loc[peores, "sin_tunel"]).round(2).to_string(index=False))
    r_fold = _parecido(_tablas(t, RHO_CDMX), t["stuff_cdmx_plus"])
    print(f"\n   Informativo: el modelo del simulador (todos los datos) contra los modelos de fold de las tablas, "
          f"mismas filas:\n   r = {r_fold[0]:.3f}, |dif| media = {r_fold[3]:.1f} puntos por pitcheo")

    print("\n2. Barridos de cordura")
    ff = t[t["pt"] == "FF"]
    if len(ff):
        s = {dv: calificar(_reconstruir(ff, recta=False, velo=dv), rho_liga)["stuff_plus"].to_numpy()
             for dv in (-2.0, -1.0, 0.0, 1.0, 2.0)}
        sube = float((s[1.0] - s[0.0]).mean())
        print(f"   a) recta de 4 costuras como recta primaria ({len(ff)} pitcheos), Stuff+ medio segun el cambio de "
              f"velocidad:\n      " + ", ".join(f"{dv:+.0f} mph {v.mean():.1f}" for dv, v in s.items()))
        print(f"      +1 mph mueve el Stuff+ {sube:+.2f} en promedio; lo baja mas de medio punto en "
              f"{100 * np.mean(s[1.0] - s[0.0] < -0.5):.0f}% de los pitcheos")
        ok.append(("+1 mph no baja el Stuff+ medio de la recta de 4 costuras", sube >= 0))
    filas = _reconstruir(t)
    mismo = calificar(filas, t["air_density"].to_numpy(float))
    quieto = float(max((mismo["ivb"] - filas["ivb"]).abs().max(), (mismo["hb"] - filas["hb_n"]).abs().max()))
    print(f"   b) cableado: evaluado en la densidad donde se midio, el quiebre cambia a lo mas {quieto:.1e} pulgadas")
    ok.append(("misma densidad, mismo quiebre (cableado)", quieto < EXACTO))
    mano = calificar(filas.assign(lefty=1 - filas["lefty"]), RHO_CDMX)["stuff_plus"] \
        - calificar(filas, RHO_CDMX)["stuff_plus"]
    print(f"   c) cambiar solo la mano (datos en lado del brazo): {mano.abs().mean():.2f} puntos en promedio, "
          f"p95 {mano.abs().quantile(.95):.1f}, maximo {mano.abs().max():.1f}")
    ok.append((f"cambiar solo la mano mueve menos de {MAX_CAMBIO_MANO:.0f} puntos en promedio",
               bool(mano.abs().mean() < MAX_CAMBIO_MANO)))
    e = pd.read_csv(config.OUT / "altitude_elasticities.csv")
    e = e[e["var"].isin(["ivb", "hb_n"]) & e["pt"].isin(list(R["defaults"]))].dropna(subset=["media_mar", "media_cdmx"])
    if len(e):
        nivel = lambda v: np.where(e["var"] == v, e["media_mar"], 5.0)
        en_cdmx = calificar(filas_pitcheo(pd.DataFrame({
            "pt": e["pt"].to_numpy(), "velo": 90.0, "ivb": nivel("ivb"), "hb_brazo": nivel("hb_n"), "spin": 2200.0,
            "rel_h": 6.0, "rel_s_brazo": 1.5, "ext": 6.0}), RHO_MAR), RHO_CDMX)
        sim = np.where(e["var"] == "ivb", en_cdmx["ivb"], en_cdmx["hb"])
        peor = float(np.abs(sim - e["media_cdmx"]).max())
        for (pt, var), nombre in ((("FF", "ivb"), "recta de 4 costuras, IVB"), (("SL", "hb_n"), "slider, HB")):
            f = ((e["pt"] == pt) & (e["var"] == var)).to_numpy()
            if f.any():
                print(f"   d) {nombre}: {e['media_mar'][f].iloc[0]:+.2f} pulg al nivel del mar -> el simulador da "
                      f"{sim[f][0]:+.2f} en el Harp Helu; la tabla de elasticidades dice {e['media_cdmx'][f].iloc[0]:+.2f}")
        print(f"      cableado: todos los tipos, IVB y HB, diferencia maxima contra altitude_elasticities.csv "
              f"{peor:.1e} pulgadas")
        ok.append(("el transporte del simulador = el de altitude_elasticities.csv (cableado)", peor < 1e-6))
    # la interfaz de UN pitcheo (fila_pitcheo, evaluar, comparar) contra el camino por tablas de la seccion 1
    peor = 0.0
    for i in range(min(5, len(t))):
        p = t.iloc[i]
        uno = comparar(fila_pitcheo(
            p["pt"], p["velocity"], p["ivb"], p["hb_n"], p["spin_rate"], p["release_height"], p["rel_side_n"],
            p["extension"], zurdo=bool(p["lefty"]), fb_tipo=p["fb_type"], fb_velo=p["fb_velo"], fb_ivb=p["fb_ivb"],
            fb_hb_brazo=p["fb_hb"], fb_rel_h=p["fb_rel_h"], fb_rel_s_brazo=p["fb_rel_s"],
            rho_medicion=float(p["air_density"]), rel_h_tipo=p["rel_h_tipo"], rel_s_tipo=p["rel_s_tipo"]))
        for lugar, rho in (("harp_helu", RHO_CDMX), ("nivel_del_mar", RHO_MAR)):
            peor = max(peor, abs(uno[lugar]["stuff_plus"] - calificar(filas.iloc[[i]], rho)["stuff_plus"].iloc[0]))
    print(f"   e) cableado: fila_pitcheo + comparar, pitcheo por pitcheo, difieren del camino por tablas a lo mas "
          f"{peor:.1e} puntos")
    ok.append(("fila_pitcheo + comparar = el camino por tablas (cableado)", peor < EXACTO))

    print("\n3. El PROMEDIO de un pitcheo: lo que se teclea en el dashboard (Simulador)")
    comprobaciones, aviso = _prueba_promedios(todos)
    ok += comprobaciones

    print("\n4. El arsenal de un prospecto contra la liga de referencia (Prospecto)")
    ok += _prueba_liga()

    pasa = all(v for _, v in ok)
    print("\nCOMPROBACIONES")
    for nombre, v in ok:
        print(f"   {'pasa ' if v else 'FALLA'}  {nombre}")
    print("   'cableado' = dos caminos del codigo que deben dar el mismo numero; no dice nada del modelo ni de la fisica")
    print("\n" + ("PASA" if pasa else "FALLA") + f": {sum(v for _, v in ok)} de {len(ok)} comprobaciones")
    if aviso:
        print("AVISO: " + aviso)
    return pasa


if __name__ == "__main__":
    try:
        sys.exit(0 if prueba() else 1)
    except SimuladorNoListo as e:
        sys.exit(str(e))
