"""
SCORES — Stuff+, Location+ y Pitching+ por pitcheo, pitcher y arsenal, y el
contrafactual de altitud: el mismo pitcheo en el Harp Helu y al nivel del mar.

FUERA DE MUESTRA: a cada pitcher lo califica el modelo del fold que NUNCA lo vio
(models/*_fold{k}.joblib y out/folds.csv, de `model.py validar`). Ningun score
entregado sale de un modelo entrenado con pitcheos de ese mismo pitcher.

ESCALA (definicion del reto):  score = -xRV  (menos carreras esperadas = mejor)
y  Plus = 100 + 10 * z(score). Hay DOS escalas y sus puntos no se comparan:
  - por PITCHEO: z por temporada sobre todos los pitcheos de la liga. Un punto
    es 1/10 de la desviacion ENTRE PITCHEOS.
  - por PITCHER y por ARSENAL: se promedia el score y se estandariza entre los
    pitchers (o arsenales) calificados, pesando por pitcheos. Un punto es 1/10
    de la desviacion ENTRE PITCHERS, que es varias veces mas chica.

CONTRAFACTUAL DE ALTITUD:
  1. features.transportar lleva el movimiento del pitcheo, y el de la recta de
     referencia de su pitcher, a la densidad destino con la elasticidad medida
     en la LMB (physics.py, columna beta_usada).
  2. el modelo Stuff se evalua con esas variables y los batazos se valuan en la
     densidad destino (model.xrv).
  Tres versiones de cada pitcheo: neutral (densidad media de la liga: es el
  Stuff+ que se reporta), Harp Helu (RHO_CDMX) y nivel del mar (RHO_MAR).
  CADA version se centra en SU PROPIA media (100 = el pitcheo promedio lanzado
  en ESE aire) y todas usan la desviacion de la neutral. Por eso hay dos numeros:
    delta_altitud     = stuff_cdmx_plus - stuff_mar_plus. Es RELATIVO: cuanto
                        sube o baja el pitcheo frente a los demas al pasar del
                        nivel del mar al Harp Helu. En la liga promedia 0.
    costo_cdmx_rv100  = carreras por 100 pitcheos que cuesta lanzar ESE pitcheo
                        en el Harp Helu en vez de al nivel del mar. Es ABSOLUTO
                        e incluye el efecto de parque, comun a todos.

INTERVALOS: remuestrean PITCHERS completos sobre sumas por pitcher. Miden el
ruido de muestreo; NO incluyen la incertidumbre de las elasticidades ni la del
valor de los batazos segun la densidad (cada una esta en su archivo). Donde
importa se agrega cuanto cambia el resultado de un modelo de fold a otro.
Los intervalos por pitcher y por arsenal (agregar) toman el JUEGO como unidad
de muestreo. Ninguno incluye el error del modelo: cuanto cambia un pitcher de
un modelo de fold a otro se mide aparte (acuerdo_modelos, val_stability.json).

Escribe en out/: scores_pitch.parquet (interno), scores_pitcher.csv,
scores_arsenal.csv, predicciones_por_pitcheo.csv.gz (el entregable por
pitcheo), altitude_by_pitch_type.csv, altitude_park_effect.json,
park_scores.csv, val_contrafactual.csv y val_stability.json; y en models/:
scale.json y sim_defaults.json (para el simulador).
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy import stats

# Las rutas (OUT, MODELS) se leen como config.X al usarse, para que una prueba
# pueda cambiarlas antes de llamar a main().
import config
import features
from config import (RAPIDO, RANDOM_STATE, PLUS_SCALE, RHO_CDMX, RHO_MAR, MIN_PITCHES_PITCHER,
                    MIN_PITCHES_ARSENAL, MIN_PITCHERS_BETA)
from features import OUTCOMES, TIPOS, PITCHING_FEATS, CATEGORICAL
from model import proba, xrv, outcome_values, I, IP, IN_PLAY, SWING, SUBMODELOS
from physics import FAMILIAS

a_densidad = features.transportar      # nombre anterior del contrafactual (compatibilidad)

N_BOOT = 100 if RAPIDO else 300        # remuestreos de pitchers de cada intervalo
MIN_PITCHERS_IC = 5                    # con menos pitchers un remuestreo no dice nada: sin intervalo
MIN_PARES = 5                          # filas minimas para calcular una correlacion
MIN_CELDA = 15                         # pitcheos en CADA banda para que una celda entre a las pruebas del contrafactual
MIN_CELDAS = 20                        # celdas minimas para reportar una prueba
N_REJILLA = 30_000 if RAPIDO else 100_000    # pitcheos con los que se calcula la media por densidad del simulador
N_ACUERDO = 50_000 if RAPIDO else 100_000    # pitcheos que califican TODOS los modelos de fold (acuerdo_modelos)
MAX_SIN_JUEGO = 0.02                   # fraccion de pitcheos sin game_id que se tolera (cada uno = un juego propio)

PUNTAJES = ("stuff", "stuff_cdmx", "stuff_mar", "location", "pitching", "pitching_neutral")
REFERENCIA = {"stuff_cdmx": "stuff", "stuff_mar": "stuff"}     # de que version toma la desviacion cada una
# pitching_neutral es el Pitching+ comparable ENTRE PARQUES (pitching valua los batazos en el parque del pitcher)
CON_INTERVALO = ("stuff", "stuff_cdmx", "location", "pitching", "pitching_neutral")
AIRES = ("neutral", "cdmx", "mar")
TRANSPORTE = ["pt", "fb_type", "ivb", "hb_n", "spin_rate", "air_density", "fb_ivb0", "fb_hb0"]
CONTEOS = ["y_whiff", "y_swing", "y_chase", "y_ooz", "y_called_strike", "y_gb", "y_bip", "y_hard", "y_ev"]

# Lo que se conserva de la tabla de modelado (y termina en scores_pitch.parquet)
IDS = ["pitch_id", "game_id", "pitcher_id", "pitcher_name", "season", "pitch_type"]
AMBIENTE = ["stadium", "altitude_category", "altitude_m", "air_density", "banda", "es_casa"]
RECTA = ["fb_type", "fb_velo", "fb_rel_h", "fb_rel_s", "fb_ivb0", "fb_hb0", "fb_ivb", "fb_hb"]
RESULTADO = ["count", "call", "outcome", "bb_type", "exit_velo", "launch_angle", "run_value"] + CONTEOS + ["y_weak"]

GRUPOS_PARQUE = ["casa", "alta_sin_casa", "media", "mar"]
COLS_TIPO = ["pt", "pitcheos", "pitchers", "stuff_mar", "stuff_cdmx", "delta", "delta_lo", "delta_hi",
             "delta_lo_folds", "delta_hi_folds", "delta_mov", "delta_mov_lo", "delta_mov_hi", "delta_valor",
             "costo_cdmx_rv100", "costo_lo", "costo_hi", "significativo"]
COLS_CF = ["bloque", "modelo", "pt", "variable", "unidad", "celdas", "pitchers", "nivel_mar", "nivel_alta",
           "predicho", "observado", "valor", "ic_lo", "ic_hi", "error_abs", "obs_lo", "obs_hi", "pasa",
           "ic_dentro_de_tolerancia", "con_potencia", "factor_escala_pitcher", "nota"]
# Pruebas del contrafactual, (bloque, modelo, variable, unidad). Cuando pasa cada una: veredicto_fila() y contrafactual().
_TASAS = (("swstr_pct", "puntos porcentuales"), ("gb_bip_pct", "puntos porcentuales"),
          ("rv100", "carreras por 100 pitcheos"))
_EN_SU_DIRECCION = "pulgadas en la direccion del quiebre"
PRUEBAS = ([("neutralidad", "stuff", "stuff_plus", "puntos de Stuff+ por pitcheo"),
            ("transporte", "fisica", "ivb", _EN_SU_DIRECCION), ("transporte", "fisica", "hb_n", _EN_SU_DIRECCION)]
           + [("resultados", modelo, var, unidad) for modelo in ("pitching", "stuff_transportado")
              for var, unidad in _TASAS])
PRUEBA_PENDIENTE = ("neutralidad", "stuff", "stuff_plus_pendiente", "puntos de Stuff+ por pitcheo")
BLOQUES = ("neutralidad", "transporte", "transporte_beta_1", "resultados", "entre_pitchers")    # orden del archivo
TOLERANCIA = {"neutralidad": 1.0, "transporte": 0.5}      # puntos de Stuff+ por pitcheo y pulgadas
NOTA_FAMILIA = ("celdas pitcher-temporada con las etiquetas de la familia juntas; el movimiento medio tambien "
                "cambia si el pitcher cambia de mezcla dentro de la familia")
NOTA_REPERTORIO = "celdas pitcher-temporada con todo el repertorio: incluye el cambio de mezcla entre parques"
NOTA_PENDIENTE = ("pendiente dentro de pitcher-temporada-tipo del Stuff+ neutral sobre la densidad media del parque, "
                  "con todas las bandas, por (RHO_CDMX - RHO_MAR)")
NOTA_BETA_1 = "referencia: el mismo transporte con beta = 1 (quiebre proporcional a la densidad); informativa"
NOTA_SIN_POTENCIA = ("sin potencia: la prediccion de referencia (el aire no mueve el pitcheo) tampoco se habria "
                     "rechazado, asi que la prueba no podia fallar")
NOTA_RV100 = ("el valor de los batazos segun la densidad se ajusto con estos mismos datos: comprobacion dentro de "
              "muestra")
NOTA_ENTRE = f"pitcher-temporadas con {MIN_PITCHES_PITCHER}+ pitcheos; informativa"


def plus(score, mu, sd):
    return 100.0 + PLUS_SCALE * (score - mu) / sd


def _aviso_grave(texto, detener=True):
    """Aviso que pone en duda los scores; con ESTRICTO=1 (corrida de entrega) detiene la corrida.
    detener=False solo avisa: quien llama decide cuando parar (main lo hace al final, con todo escrito)."""
    print("\n" + "!" * 78 + f"\nAVISO: {texto}\n" + "!" * 78 + "\n")
    if detener and config.ESTRICTO:
        raise SystemExit("ESTRICTO=1: corrida detenida por el aviso anterior.")


def razon(num, den):
    """num / den sin avisos; NaN donde el denominador no es positivo."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ── remuestreo de pitchers ──────────────────────────────────────────────────
def remuestreo(pitcher_id):
    """Para los intervalos se remuestrean PITCHERS completos: los pitcheos de un
    mismo pitcher se parecen, y remuestrear pitcheos da intervalos demasiado
    angostos. Devuelve (codigo de pitcher de cada fila, pitchers, W). Cada fila
    de W dice cuantas veces entra cada pitcher en una replica; la fila 0 es la
    muestra real. Con W cualquier media remuestreada es una multiplicacion de
    matrices sobre SUMAS POR PITCHER: nunca se remuestrea pitcheo por pitcheo."""
    cod, pitchers = pd.factorize(pitcher_id)
    n = len(pitchers)
    rng = np.random.default_rng(RANDOM_STATE)
    W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT)])
    return cod, pd.Index(pitchers), W


def sumas(cod, W, v):
    """Suma de v en la muestra real ([0]) y en cada replica ([1:])."""
    return W @ np.bincount(cod, weights=v, minlength=W.shape[1])


def ic(v, pitchers=MIN_PITCHERS_IC):
    """(valor real, limite inferior, limite superior) del IC 95% de un arreglo
    [real, replicas...]. Sin limites si salio de menos de MIN_PITCHERS_IC pitchers."""
    replicas = v[1:][np.isfinite(v[1:])]
    if pitchers < MIN_PITCHERS_IC or not len(replicas):
        return float(v[0]), np.nan, np.nan
    lo, hi = np.percentile(replicas, [2.5, 97.5])
    return float(v[0]), float(lo), float(hi)


def corr_boot(x, y, cod, W):
    """Correlacion de x con y: [muestra real, replicas...]. En cada replica una
    fila pesa las veces que salio su pitcher."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < MIN_PARES:
        return np.full(len(W), np.nan)
    w = W[:, cod]
    n = w.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = w @ x / n, w @ y / n
        sxy = w @ (x * y) / n - mx * my
        sxx, syy = w @ (x * x) / n - mx ** 2, w @ (y * y) / n - my ** 2
        return np.where((sxx > 1e-12) & (syy > 1e-12), sxy / np.sqrt(sxx * syy), np.nan)


def pendiente_boot(x, y, cod, W):
    """Pendiente de y sobre x (minimos cuadrados): [muestra real, replicas...],
    con los mismos pesos por pitcher que corr_boot."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < MIN_PARES:
        return np.full(len(W), np.nan)
    w = W[:, cod]
    n = w.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = w @ x / n, w @ y / n
        sxy, sxx = w @ (x * y) / n - mx * my, w @ (x * x) / n - mx ** 2
        return np.where(sxx > 1e-12, sxy / sxx, np.nan)


# ── carga y calificacion fuera de muestra ───────────────────────────────────
def _jonron(df):
    """1.0 si el batazo fue jonron (features.es_jonron: la misma regla que el
    resto del pipeline); NaN si el dataset no trae ni home_run ni play_result."""
    if not {"home_run", "play_result"} & set(df.columns):
        return np.nan
    return (features.es_jonron(df) & (df["y_bip"] == 1)).astype(float)


def cargar():
    """Tabla de modelado (solo las columnas que se usan), elasticidades, valores
    de cada resultado, modelos de cada fold y el fold de cada pitcheo: el del
    modelo que NO vio a su pitcher."""
    faltan = [p.name for p in (config.MODELS / "outcome_values.json", config.OUT / "folds.csv") if not p.exists()]
    if faltan:
        raise SystemExit(f"Faltan {faltan}: corre primero  {config.comando('valores')}")
    valores = json.loads((config.MODELS / "outcome_values.json").read_text())
    df = features.cargar_features()
    beta = features.cargar_elasticidades()
    falta = [c for c in TRANSPORTE + CONTEOS + ["pitch_id", "pitcher_id", "pitcher_name", "season", "stadium",
                                               "banda", "es_casa", "count", "outcome", "run_value", "in_zone"]
             if c not in df]
    if falta:
        raise SystemExit(f"A out/features.parquet le faltan columnas {falta}: vuelve a correr  "
                         + config.comando("features"))
    hr = _jonron(df)
    cols = dict.fromkeys(IDS + AMBIENTE + PITCHING_FEATS + CATEGORICAL + RECTA + RESULTADO)
    df = df[[c for c in cols if c in df]].copy()
    df["y_hr"], df["y_fb"] = hr, df["outcome"].isin(["ip_fb_soft", "ip_fb_hard"]).astype(int)
    df["total_temporada"] = df.groupby(["pitcher_id", "season"])["pitch_id"].transform("size")

    folds = pd.read_csv(config.OUT / "folds.csv", dtype={"pitcher_id": str}).drop_duplicates("pitcher_id")
    fold = df["pitcher_id"].astype(str).map(folds.set_index("pitcher_id")["fold"])
    nuevos = df.loc[fold.isna(), "pitcher_id"].nunique()
    if nuevos:
        _aviso_grave(f"{nuevos} pitchers ({int(fold.isna().sum()):,} pitcheos) no estan en out/folds.csv: los datos "
                     "cambiaron despues de la validacion. Se califican con el modelo del fold 0; lo correcto es "
                     f"volver a correr  {config.comando('validar')}")
    fold = fold.fillna(0).astype(int).to_numpy()
    modelos = {}
    for nm in SUBMODELOS:
        for k in sorted(set(fold.tolist())):
            ruta = config.MODELS / f"{nm}_fold{k}.joblib"
            if not ruta.exists():
                raise SystemExit(f"Falta {ruta}: corre primero  {config.comando('validar')}")
            modelos.setdefault(nm, {})[k] = joblib.load(ruta)
    return df, beta, valores, modelos, fold


def predecir(por_fold, X, fold):
    """P(resultado) de cada pitcheo, (n, 10), con el modelo del fold de su
    pitcher: el unico de los N que se entreno SIN ese pitcher."""
    P = np.zeros((len(X), len(OUTCOMES)), dtype=np.float32)
    for k, m in por_fold.items():
        filas = np.flatnonzero(fold == k)
        if len(filas):
            P[filas] = proba(m, X.iloc[filas])
    return P


def _cols(df, modelos, nombre):
    """Columnas de df que usa un sub-modelo; para Stuff, ademas las del transporte."""
    feats = next(iter(modelos[nombre].values())).feats_
    return [c for c in dict.fromkeys(feats + (TRANSPORTE if nombre == "stuff" else [])) if c in df]


def _chequeo_fuera_de_muestra(df, P):
    """El log-loss de Pitching+ aqui tiene que ser el fuera de muestra de la
    validacion. Si no coincide, los modelos de models/ no corresponden a estos
    datos (por ejemplo se volvio a correr features.py o physics.py sin validar)."""
    y = pd.Index(OUTCOMES).get_indexer(df["outcome"])
    aqui = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1.0)).mean())
    ruta = config.OUT / "val_resumen.csv"
    val = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=["modelo", "logloss"])
    val = val.loc[val["modelo"] == "pitching", "logloss"]
    if not len(val):
        print(f"  log-loss de Pitching+ fuera de muestra: {aqui:.4f} (no hay val_resumen.csv para comparar)")
    elif abs(aqui - float(val.iloc[0])) > 2e-3:
        _aviso_grave(f"el log-loss de Pitching+ aqui ({aqui:.4f}) no es el de la validacion "
                     f"({float(val.iloc[0]):.4f}): los modelos de models/ no corresponden a estos datos. "
                     f"Vuelve a correr  {config.comando('validar')}  antes de entregar.")
    else:
        print(f"  log-loss de Pitching+ fuera de muestra: {aqui:.4f} (validacion: {float(val.iloc[0]):.4f}) -> coincide")


def calificar(df, beta, valores, modelos, fold):
    """Agrega a df los scores por pitcheo (score = -xRV), todos fuera de muestra:
      stuff_score        pitcheo llevado a la densidad media de la liga, batazos a valor de liga
      stuff_cdmx_score   llevado al aire del Harp Helu y valuado ahi; stuff_mar_score igual al nivel del mar
      location_score     ubicacion y cuenta, batazos a valor de liga
      pitching_score     todo junto, batazos valuados en el parque donde se lanzo
      pitching_neutral_score   igual pero a valor de liga: comparable entre parques
    mas las coordenadas transportadas (ivb_/hb_ neutral, cdmx, mar), las diez
    probabilidades de Pitching+ y el costo de lanzar en CDMX por pitcheo.
    Devuelve la probabilidad media de cada clase de batazo en CDMX y en el mar
    (para separar el efecto de parque)."""
    den = valores["densidad"]
    X = df[_cols(df, modelos, "stuff")]
    xr, batazos = {}, {}
    for aire, rho in zip(AIRES, (den["rho_liga"], RHO_CDMX, RHO_MAR)):
        d = features.transportar(X, rho, beta)
        P = predecir(modelos["stuff"], d, fold)
        df[f"ivb_{aire}"], df[f"hb_{aire}"] = d["ivb"].to_numpy(), d["hb_n"].to_numpy()
        # (batazos a valor de liga, batazos en ese aire); en "neutral" son lo mismo
        xr[aire], batazos[aire] = (xrv(P, valores), xrv(P, valores, rho=rho)), P[:, IP].mean(0).astype(float)
    df["stuff_score"] = -xr["neutral"][0]
    df["stuff_cdmx_score"], df["stuff_mar_score"] = -xr["cdmx"][1], -xr["mar"][1]
    df["costo_cdmx_rv100"] = 100 * (xr["cdmx"][1] - xr["mar"][1])
    df["costo_mov_rv100"] = 100 * (xr["cdmx"][0] - xr["mar"][0])      # solo por el cambio de movimiento
    cuenta = df["count"]
    P = predecir(modelos["location"], df[_cols(df, modelos, "location")], fold)
    df["location_score"] = -xrv(P, valores, counts=cuenta)
    P = predecir(modelos["pitching"], df[_cols(df, modelos, "pitching")], fold)
    _chequeo_fuera_de_muestra(df, P)
    df["pitching_score"] = -xrv(P, valores, rho=df["air_density"].to_numpy(float), counts=cuenta)
    df["pitching_neutral_score"] = -xrv(P, valores, counts=cuenta)
    df["xrv"] = -df["pitching_score"]
    for c, i in I.items():
        df[f"p_{c}"] = P[:, i]
    df["p_swing"] = P[:, SWING].sum(1)
    df["p_chase"] = df["p_swing"].where(df["in_zone"] == 0)           # P(swing) de un pitcheo fuera de zona
    df["p_linea_o_elevado_duro"] = df["p_ip_ld"] + df["p_ip_fb_hard"]
    df["fold"] = fold
    return batazos


def a_escala(df):
    """Plus por pitcheo: z por temporada sobre todos los pitcheos. Las tres
    versiones de Stuff se centran cada una en su media y comparten la
    desviacion de la neutral (ver el encabezado del modulo).
    delta_altitud = stuff_cdmx_plus - stuff_mar_plus equivale a
    -(costo - costo medio de la liga) en puntos, asi que se separa igual que el
    costo: delta_mov es la parte que viene del cambio de movimiento y el resto
    (delta_altitud - delta_mov) viene de cuanto mas valen sus batazos."""
    temporada, sd = df["season"], {}
    for s in PUNTAJES:
        ref = df[f"{REFERENCIA.get(s, s)}_score"]
        mu = df[f"{s}_score"].groupby(temporada).transform("mean")
        sd[s] = ref.groupby(temporada).transform("std")
        sd[s] = sd[s].where(sd[s] > 0, ref.std())
        df[f"{s}_plus"] = plus(df[f"{s}_score"], mu, sd[s])
    df["delta_altitud"] = df["stuff_cdmx_plus"] - df["stuff_mar_plus"]
    mov = df["costo_mov_rv100"]
    df["delta_mov"] = -PLUS_SCALE * (mov - mov.groupby(temporada).transform("mean")) / 100.0 / sd["stuff"]


# ── agregados por pitcher y por arsenal ─────────────────────────────────────
def tasas(g):
    """Tasas observadas de un groupby, cada una con SU denominador (fracciones
    de 0 a 1): swing fallado por pitcheo y por swing, chase por pitcheo fuera
    de zona, strike cantado por pitcheo, rodados por batazo en juego y contacto
    duro (95+ mph) por batazo con velocidad de salida medida."""
    s, n = g[CONTEOS].sum(), g.size()
    entre = lambda num, den: s[num] / s[den].where(s[den] > 0)
    return pd.DataFrame({"swstr_pct": s["y_whiff"] / n, "whiff_swing_pct": entre("y_whiff", "y_swing"),
                         "chase_pct": entre("y_chase", "y_ooz"), "cs_pct": s["y_called_strike"] / n,
                         "gb_pct": entre("y_gb", "y_bip"), "hard_pct": entre("y_hard", "y_ev")})


def juegos(df):
    """game_id de cada pitcheo, o None si el dataset no lo trae o le falta a mas de MAX_SIN_JUEGO de los
    pitcheos. A un pitcheo sin juego se le da uno propio (cuenta como pitcheo independiente): sin esto UN solo
    hueco devolvia a TODOS los pitchers al ruido de pitcheos independientes."""
    if "game_id" not in df or df["game_id"].isna().mean() > MAX_SIN_JUEGO:
        return None
    falta = df["game_id"].isna().to_numpy()
    if not falta.any():
        return df["game_id"]
    g = df["game_id"].astype(str).to_numpy(dtype=object)
    g[falta] = [f"sin_juego_{i}" for i in np.flatnonzero(falta)]
    return pd.Series(g, index=df.index, name="game_id")


def ruido_del_promedio(df, llaves, cols, ok):
    """Varianza del promedio de `cols` en los grupos de `llaves` que marca `ok`,
    con el JUEGO como unidad de muestreo. Devuelve (ruido, icc).

    Los pitcheos de un mismo juego se parecen (la mezcla de ese dia, como venia
    el brazo, el parque). Suponerlos independientes (s2 / n) daba una
    confiabilidad mayor que la que mide este mismo modulo partiendo por juegos
    (estabilidad). Por eso

        ruido = s2 / n * (1 + icc * (m - 1)),    m = suma(n_juego^2) / n

    m son los pitcheos efectivos por juego del grupo e icc la correlacion
    intraclase, UNA por score y comun a todos los grupos: la que iguala,
    sumando los grupos con 2 o mas juegos, la varianza del promedio con los
    juegos como conglomerados, G / (G - 1) * suma_juego(residuo_juego^2) / n^2.
    Grupo por grupo esa varianza es inestable (cinco juegos son cinco datos);
    con el icc comun es estable y cada grupo conserva SU m: un abridor (m ~ 80)
    se infla mas que un relevista (m ~ 20). Sin game_id, o sin grupos con dos
    juegos, icc = 0 y el ruido es el de pitcheos independientes."""
    g = df.groupby(llaves)
    n = g.size()[ok].to_numpy(float)
    ruido = g[cols].var()[ok].div(n, axis=0)
    icc = dict.fromkeys(cols, 0.0)
    gid = juegos(df)
    if gid is None:
        return ruido, icc
    juego = (df[cols] - g[cols].transform("mean")).groupby([df[k] for k in llaves] + [gid])
    n_juego, nivel = juego.size(), list(range(len(llaves)))
    G = n_juego.groupby(level=nivel).size()[ok].to_numpy(float)
    m = (n_juego ** 2).groupby(level=nivel).sum()[ok].to_numpy(float) / n
    residuo2 = (juego.sum() ** 2).groupby(level=nivel).sum()[ok]
    varios = G >= 2
    for c in cols:
        iid = ruido[c].to_numpy(float)
        conglomerado = residuo2[c].to_numpy(float) * G / np.maximum(G - 1, 1) / n ** 2
        den = float(np.sum((n * iid * (m - 1))[varios]))
        if den > 0:
            icc[c] = float(np.clip(np.sum((n * (conglomerado - iid))[varios]) / den, 0.0, 1.0))
        ruido[c] = iid * (1 + icc[c] * (m - 1))
    return ruido, icc


def agregar(df, llaves, minimo):
    """Una fila por grupo (pitcher-temporada, o pitcher-temporada-tipo) con al
    menos `minimo` pitcheos. Devuelve (tabla, escala) con escala[s] = media y
    desviacion entre grupos con las que se calculo <s>_plus, y icc_juego, la
    correlacion intraclase por juego con la que se calculo su ruido.

    Para cada score s:  <s>_raw (promedio) y <s>_plus (escala ENTRE GRUPOS:
    cada version centrada en su media pesada por pitcheos; desviacion de su
    version de referencia). Para los de CON_INTERVALO ademas:
      <s>_fiab      confiabilidad = tau2 / (tau2 + ruido): que parte de lo que
                    se ve en ese grupo es talento y no ruido de muestreo, DADO
                    EL MODELO (tau2 = varianza entre grupos ya sin ruido; ruido
                    = varianza del promedio del grupo, con el juego como unidad:
                    ruido_del_promedio)
      <s>_plus_enc  el plus encogido hacia 100 por esa confiabilidad. Es el que
                    se usa para ordenar: con 150 pitcheos Location+ y Pitching+
                    son en buena parte ruido; Stuff+ se mueve poco.
      <s>_lo/_hi    intervalo posterior del 95%, simetrico alrededor del encogido.
                    Solo ruido de muestreo (juegos y pitcheos). NO incluye el
                    error del modelo, que se mide aparte (acuerdo_modelos).
    pitching valua los batazos en el parque donde lanzo el pitcher y por eso
    sigue al parque; para comparar pitchers de parques distintos esta
    pitching_neutral, con las mismas columnas.
    delta_altitud = stuff_cdmx_plus - stuff_mar_plus (relativo, escala entre
    grupos) con delta_lo / delta_hi = +/- 1.96 errores estandar del promedio
    (mismo ruido por juego)."""
    g = df.groupby(llaves)
    n = g.size()
    cuerpo = g.agg(pitcher_name=("pitcher_name", "first"), pitcheos=("pitch_id", "size"),
                   rho=("air_density", "mean"), lefty=("lefty", "first"),
                   total_temporada=("total_temporada", "first"), velo=("velocity", "mean"),
                   velo_recta=("fb_velo", "first"), ivb=("ivb", "mean"), hb=("hb_n", "mean"),
                   ivb_cdmx=("ivb_cdmx", "mean"), hb_cdmx=("hb_cdmx", "mean"), spin=("spin_rate", "mean"),
                   ext=("extension", "mean"), rel_h=("release_height", "mean"), rel_s=("rel_side_n", "mean"),
                   rv100=("run_value", "mean"))
    cuerpo["rv100"] *= 100
    cuerpo.insert(3, "pct_alta", (df["banda"] == "alta").groupby([df[k] for k in llaves]).mean())
    cuerpo.insert(6, "uso", cuerpo["pitcheos"] / cuerpo["total_temporada"])
    cols = [f"{s}_score" for s in PUNTAJES] + ["costo_cdmx_rv100"]
    ok = n >= minimo
    if not ok.any():
        raise SystemExit(f"Ningun grupo de {llaves} llega a {minimo} pitcheos: no hay nada que agregar")
    a, n = pd.concat([cuerpo, tasas(g)], axis=1)[ok], n[ok].to_numpy(float)
    media = g[cols].mean()[ok]
    ruido, icc = ruido_del_promedio(df, llaves, cols, ok)                    # ruido = varianza del promedio
    media_var = lambda x: (np.average(x, weights=n), np.average((x - np.average(x, weights=n)) ** 2, weights=n))
    escala, nuevas = {"delta_altitud": {"icc_juego": icc["costo_cdmx_rv100"]}}, {}
    for s in PUNTAJES:
        m, v = media[f"{s}_score"].to_numpy(float), ruido[f"{s}_score"].to_numpy(float)
        mu, var = media_var(m)
        sd = np.sqrt(media_var(media[f"{REFERENCIA.get(s, s)}_score"].to_numpy(float))[1])
        escala[s] = {"mu": float(mu), "sd": float(sd), "icc_juego": icc[f"{s}_score"]}
        nuevas[f"{s}_raw"], nuevas[f"{s}_plus"] = m, plus(m, mu, sd)
        if s in CON_INTERVALO:
            tau2 = max(1e-12, var - np.average(v, weights=n))
            fiab = tau2 / (tau2 + v)
            margen = 1.96 * PLUS_SCALE * np.sqrt(fiab * v) / sd        # varianza posterior = fiab * ruido
            enc = 100.0 + fiab * (nuevas[f"{s}_plus"] - 100.0)
            nuevas.update({f"{s}_fiab": fiab, f"{s}_plus_enc": enc, f"{s}_lo": enc - margen, f"{s}_hi": enc + margen})
    # el delta es una funcion lineal del costo por pitcheo: su error estandar sale de ahi
    delta = nuevas["stuff_cdmx_plus"] - nuevas["stuff_mar_plus"]
    margen = 1.96 * PLUS_SCALE * np.sqrt(ruido["costo_cdmx_rv100"].to_numpy(float)) / 100.0 / escala["stuff"]["sd"]
    nuevas.update({"delta_altitud": delta, "delta_lo": delta - margen, "delta_hi": delta + margen,
                   "costo_cdmx_rv100": media["costo_cdmx_rv100"].to_numpy(float)})
    return pd.concat([a, pd.DataFrame(nuevas, index=a.index)], axis=1).reset_index(), escala


# ── efecto de la altitud ────────────────────────────────────────────────────
def _fuera_de_cero(lo, hi):
    return bool(lo > 0 or hi < 0)


def por_tipo(df, cod, W):
    """Efecto de la altitud por tipo de pitcheo, en la escala POR PITCHEO.
      stuff_mar / stuff_cdmx   nivel del tipo en cada aire (100 = pitcheo promedio de ese aire)
      delta                    cambio RELATIVO (CDMX - mar) frente a la liga
      delta_lo / delta_hi      IC 95% remuestreando pitchers; en cada replica el tipo se
                               vuelve a comparar contra la liga de esa replica
      delta_lo_folds / _hi_folds   IC 95% ENTRE MODELOS: el delta calculado por separado con
                               cada modelo de fold (cada uno califica a sus propios pitchers)
                               y una t de Student sobre esos valores. Remuestrear pitchers no
                               ve que otro modelo, entrenado con otros pitchers, da otro delta
      delta_mov + delta_valor = delta: lo que viene del cambio de movimiento y lo que viene
                               de cuanto mas valen en CDMX los batazos que ese tipo permite
      costo_cdmx_rv100         costo ABSOLUTO en carreras por 100 pitcheos, con su IC
      significativo            los DOS intervalos del delta excluyen 0 y hay al menos
                               MIN_PITCHERS_BETA pitchers"""
    delta, mov, costo = (df[c].to_numpy(float) for c in ("delta_altitud", "delta_mov", "costo_cdmx_rv100"))
    uno = np.ones(len(df))
    media = lambda v, m: razon(sumas(cod[m], W, v[m]), sumas(cod[m], W, uno[m]))
    todos = np.ones(len(df), dtype=bool)
    liga, liga_mov = media(delta, todos), media(mov, todos)
    d = pd.DataFrame({"pt": df["pt"].to_numpy(), "fold": df["fold"].to_numpy(), "delta": delta})
    por_fold = d.groupby(["pt", "fold"])["delta"].mean().unstack("fold") - d.groupby("fold")["delta"].mean()
    filas = []
    for pt in TIPOS:
        m = (df["pt"] == pt).to_numpy()
        if not m.any():
            continue
        pitchers = int(len(np.unique(cod[m])))
        v, lo, hi = ic(media(delta, m) - liga, pitchers)
        k = por_fold.loc[pt].dropna()
        margen = stats.t.ppf(0.975, len(k) - 1) * k.std(ddof=1) / np.sqrt(len(k)) if len(k) > 1 else np.nan
        v_mov, mov_lo, mov_hi = ic(media(mov, m) - liga_mov, pitchers)
        c, c_lo, c_hi = ic(media(costo, m), pitchers)
        filas.append({"pt": pt, "pitcheos": int(m.sum()), "pitchers": pitchers,
                      "stuff_mar": float(df.loc[m, "stuff_mar_plus"].mean()),
                      "stuff_cdmx": float(df.loc[m, "stuff_cdmx_plus"].mean()),
                      "delta": v, "delta_lo": lo, "delta_hi": hi, "delta_lo_folds": v - margen,
                      "delta_hi_folds": v + margen, "delta_mov": v_mov, "delta_mov_lo": mov_lo,
                      "delta_mov_hi": mov_hi, "delta_valor": v - v_mov, "costo_cdmx_rv100": c, "costo_lo": c_lo,
                      "costo_hi": c_hi,
                      "significativo": bool(pitchers >= MIN_PITCHERS_BETA and _fuera_de_cero(lo, hi)
                                            and _fuera_de_cero(v - margen, v + margen))})
    return pd.DataFrame(filas, columns=COLS_TIPO).sort_values("delta", ascending=False)


def efecto_parque(df, pit, cod, W, valores, batazos):
    """Costo ABSOLUTO de lanzar en el Harp Helu en vez de al nivel del mar
    (carreras por 100 pitcheos, promedio de la liga) y sus dos partes:
      por_movimiento     el pitcheo quiebra menos (batazos a valor de liga)
      por_valor_batazos  el mismo batazo vale mas en aire delgado (el resto)
    Tres medidas de incertidumbre, porque son tres fuentes distintas:
      *_ic95       remuestreo de pitchers: solo ruido de muestreo
      *_por_fold   el mismo numero con los pitchers de cada modelo de fold
      *_ic95_recta la parte de los batazos depende de la recta de valor contra
                   densidad (outcome_values.json): se vuelve a ajustar quitando
                   un estadio (o categoria de altitud) a la vez, como en
                   physics.py. Con menos de 6 puntos es solo orientativo."""
    uno = np.ones(len(df))
    n, den = sumas(cod, W, uno), valores["densidad"]
    total = razon(sumas(cod, W, df["costo_cdmx_rv100"].to_numpy(float)), n)
    mov = razon(sumas(cod, W, df["costo_mov_rv100"].to_numpy(float)), n)
    rv = 100 * df["run_value"].to_numpy(float)
    casa, mar = (df["es_casa"] == 1).to_numpy(), (df["banda"] == "mar").to_numpy()
    media = lambda m: razon(sumas(cod[m], W, rv[m]), sumas(cod[m], W, uno[m]))
    out = {}
    for llave, llave_ic, v in (("costo_cdmx_vs_mar_rv100", "ic95", total),
                               ("por_movimiento_rv100", "por_movimiento_ic95", mov),
                               ("por_valor_batazos_rv100", "por_valor_batazos_ic95", total - mov),
                               ("observado_casa_menos_mar_rv100", "observado_ic95", media(casa) - media(mar))):
        out[llave], *out[llave_ic] = ic(v)
    # la parte de los batazos es lineal en las pendientes: k = carreras por unidad de pendiente de cada clase
    k = 100 * (batazos["cdmx"] * (RHO_CDMX - den["rho_liga"]) - batazos["mar"] * (RHO_MAR - den["rho_liga"]))
    d = df[["outcome", "count", "run_value", "stadium", "air_density"]]
    estadios = d["stadium"].unique()
    K, margen = len(estadios), np.nan
    if K >= 3:
        pendientes = [outcome_values(d[d["stadium"] != e])["densidad"]["pendiente"] for e in estadios]
        sin_uno = np.array([k @ [p[c] for c in IN_PLAY] for p in pendientes])
        margen = stats.t.ppf(0.975, K - 1) * np.sqrt((K - 1) / K * np.sum((sin_uno - sin_uno.mean()) ** 2))
    por_fold = df.groupby("fold")[["costo_cdmx_rv100", "costo_mov_rv100"]].mean()
    return {**out, "ic95_recta": [out["costo_cdmx_vs_mar_rv100"] - margen, out["costo_cdmx_vs_mar_rv100"] + margen],
            "por_valor_batazos_ic95_recta": [out["por_valor_batazos_rv100"] - margen,
                                             out["por_valor_batazos_rv100"] + margen],
            "recta_ic_fiable": bool(K >= 6), "recta_n_puntos": int(K),
            "costo_por_fold": por_fold["costo_cdmx_rv100"].tolist(),
            "por_movimiento_por_fold": por_fold["costo_mov_rv100"].tolist(),
            "rango_pitchers_p10_p90": [float(pit["costo_cdmx_rv100"].quantile(q)) for q in (0.1, 0.9)],
            "pitchers": int(W.shape[1]), "pitcher_temporadas": int(len(pit)),
            "rho_cdmx": RHO_CDMX, "rho_mar": RHO_MAR, "rho_liga": den["rho_liga"]}


def _muestra(df):
    """Posiciones de hasta N_REJILLA pitcheos al azar (siempre los mismos)."""
    if len(df) <= N_REJILLA:
        return np.arange(len(df))
    return np.sort(np.random.default_rng(RANDOM_STATE).choice(len(df), N_REJILLA, replace=False))


def costo_entre_bandas(df, beta, valores, modelos, fold, costo):
    """El costo modelado compara RHO_CDMX con RHO_MAR; el observado (casa menos
    banda mar) solo cubre las densidades que de verdad se midieron, un rango mas
    corto, asi que lado a lado no son lo mismo. Devuelve el costo modelado
    entre ESAS dos densidades (rho_casa_obs, rho_mar_obs), que es el que se
    compara con observado_casa_menos_mar_rv100. Se calcula en la muestra de
    _muestra y se lleva al total con la razon contra el costo de referencia de
    esa misma muestra. NaN si falta alguna de las dos bandas."""
    rho = {"rho_casa_obs": float(df.loc[df["es_casa"] == 1, "air_density"].mean()),
           "rho_mar_obs": float(df.loc[df["banda"] == "mar", "air_density"].mean())}
    idx = _muestra(df)
    ref = float(df["costo_cdmx_rv100"].to_numpy(float)[idx].mean())
    if not (np.isfinite(list(rho.values())).all() and np.isfinite(costo) and abs(ref) > 1e-9):
        return {"costo_entre_bandas_rv100": np.nan, **rho}
    X, f = df[_cols(df, modelos, "stuff")].iloc[idx], fold[idx]
    en = lambda r: xrv(predecir(modelos["stuff"], features.transportar(X, r, beta), f), valores, rho=r)
    bandas = 100 * float((en(rho["rho_casa_obs"]) - en(rho["rho_mar_obs"])).mean())
    return {"costo_entre_bandas_rv100": float(costo * bandas / ref), **rho}


def tabla_parques(df):
    """Scores y resultados observados por grupo de parques: casa, resto de la
    banda alta, media y mar. Los plus son promedios en la escala por pitcheo.
    Son promedios crudos: incluyen al staff local, no solo al parque.
    hr_fb = TODOS los jonrones entre los elevados, la convencion usual y la
    misma de physics.park_table (hr_por_fb). OJO: no es una proporcion. El
    numerador cuenta todo jonron, tambien los que hit_type etiqueta como linea
    u otro tipo, y esos no estan en el denominador."""
    grupo = np.where(df["es_casa"] == 1, "casa", np.where(df["banda"] == "alta", "alta_sin_casa", df["banda"]))
    g = df.groupby(grupo)
    hr = g[["y_hr", "y_fb"]].sum(min_count=1)
    t = pd.concat([g.size().rename("n"), g["pitcher_id"].nunique().rename("pitchers"),
                   g[["stuff_plus", "location_plus", "pitching_plus", "pitching_neutral_plus"]].mean(),
                   (100 * g[["xrv", "run_value"]].mean()).set_axis(["xrv_x100", "rv_x100"], axis=1), tasas(g),
                   (hr["y_hr"] / hr["y_fb"].where(hr["y_fb"] > 0)).rename("hr_fb")], axis=1)
    t = t.reindex(GRUPOS_PARQUE).rename_axis("grupo").reset_index()
    t[["n", "pitchers"]] = t[["n", "pitchers"]].fillna(0).astype(int)
    return t


# ── pruebas de aceptacion del contrafactual ─────────────────────────────────
def celdas_alta_mar(df, llaves, filas=None):
    """Pitcheos de las celdas (grupos de `llaves`, p. ej. pitcher-temporada-tipo)
    con al menos MIN_CELDA pitcheos en la banda alta y MIN_CELDA en la banda
    mar: lo mismo del mismo pitcher, lanzado arriba y abajo. `filas` limita los
    pitcheos que entran. Columnas: fila (posicion en df), celda, alta (True /
    False) y u, el peso del pitcheo. Cada celda pesa n_alta * n_mar /
    (n_alta + n_mar) y ese peso se reparte entre sus pitcheos de CADA banda:
    asi las dos bandas se comparan con la misma mezcla de pitchers y pitcheos."""
    en = df["banda"].isin(["alta", "mar"]).to_numpy()
    filas = np.flatnonzero(en if filas is None else en & np.asarray(filas))
    d = df.iloc[filas]
    celda = d.groupby(list(llaves), sort=False).ngroup().to_numpy()
    alta = (d["banda"] == "alta").to_numpy()
    K = int(celda.max()) + 1 if len(celda) else 0
    n_a, n_m = np.bincount(celda[alta], minlength=K), np.bincount(celda[~alta], minlength=K)
    ok = ((n_a >= MIN_CELDA) & (n_m >= MIN_CELDA))[celda]
    u = (n_a * n_m / np.maximum(n_a + n_m, 1))[celda] / np.maximum(np.where(alta, n_a[celda], n_m[celda]), 1)
    return pd.DataFrame({"fila": filas[ok], "celda": celda[ok], "alta": alta[ok], "u": u[ok]})


def por_banda(e, cod, W, x, den=1.0):
    """(alta, mar): media de x, o razon x / den, en cada banda de las celdas de
    `e`; cada una como [muestra real, replicas...]."""
    alta, u, cp = e["alta"].to_numpy(), e["u"].to_numpy(), cod[e["fila"].to_numpy()]
    x, den = u * np.asarray(x, dtype=float), u * np.asarray(den, dtype=float) * np.ones(len(e))
    return [razon(sumas(cp[b], W, x[b]), sumas(cp[b], W, den[b])) for b in (alta, ~alta)]


def veredicto_fila(bloque, v, lo, hi, con_potencia=None):
    """(pasa, ic_dentro_de_tolerancia) de una fila; None = sin veredicto.
      neutralidad, transporte   pasa si |valor| < tolerancia. NO se pide que el IC
            contenga 0: con muchos datos una diferencia de decimas con un IC
            angosto sigue siendo despreciable. ic_dentro_de_tolerancia dice si
            TODO el intervalo cabe en +/- la tolerancia ("equivalente") o si solo
            pasa por la estimacion puntual.
      resultados   NO pasa si el IC de (predicho - observado) excluye 0. Si lo
            contiene, solo pasa cuando la prueba podia fallar (con_potencia: la
            prediccion de referencia "el aire no mueve el pitcheo" SI habria
            fallado; ver contrafactual). Sin potencia no hay veredicto.
      los demas bloques son informativos: sin veredicto."""
    con_ic, tol = bool(np.isfinite(lo) and np.isfinite(hi)), TOLERANCIA.get(bloque)
    if not np.isfinite(v):
        return None, None
    if tol is not None:
        return bool(abs(v) < tol), (bool(-tol < lo and hi < tol) if con_ic else None)
    if bloque != "resultados" or not con_ic:
        return None, None
    if _fuera_de_cero(lo, hi):
        return False, None
    return (True if con_potencia else None), None


def _fila(prueba, pt, e, cod, dif=None, nota="", **mas):
    """Una fila de val_contrafactual.csv. Sin `dif` la fila dice "sin datos"."""
    bloque, modelo, var, unidad = prueba
    pitchers = int(len(np.unique(cod[e["fila"].to_numpy()])))
    fila = {"bloque": bloque, "modelo": modelo, "pt": pt, "variable": var, "unidad": unidad,
            "celdas": int(e["celda"].nunique()), "pitchers": pitchers, "nota": nota}
    if dif is None:
        return {**fila, "nota": "sin datos"}
    v, lo, hi = ic(dif, pitchers)
    pasa, dentro = veredicto_fila(bloque, v, lo, hi, mas.get("con_potencia"))
    if bloque == "resultados" and pasa is None and mas.get("con_potencia") is False:
        fila["nota"] = "; ".join(filter(None, [nota, NOTA_SIN_POTENCIA]))
    return {**fila, "valor": v, "ic_lo": lo, "ic_hi": hi, "pasa": pasa, "ic_dentro_de_tolerancia": dentro, **mas}


def _pruebas(df, e, pt, nota, beta, valores, modelos, fold, cod, W, signo):
    """Las tres pruebas sobre un conjunto de celdas `e` (ver contrafactual).
    signo[var]: signo del quiebre medio de la liga del tipo de cada pitcheo."""
    if e["celda"].nunique() < MIN_CELDAS:
        return [_fila(prueba, pt, e, cod) for prueba in PRUEBAS]
    fila_e, alta = e["fila"].to_numpy(), e["alta"].to_numpy()
    celda, mar, d = pd.factorize(e["celda"])[0], ~alta, df.iloc[fila_e]
    K, pitchers = int(celda.max()) + 1, int(len(np.unique(cod[fila_e])))
    n_a, n_m = np.bincount(celda[alta], minlength=K), np.bincount(celda[mar], minlength=K)
    bandas = lambda x, den=1.0: por_banda(e, cod, W, x, den)

    def en_mar(v, resto=0.0):
        """Arreglo de todas las filas con v en las de la banda mar."""
        x = np.full(len(e), resto, dtype=float) if np.isscalar(resto) else np.asarray(resto, dtype=float).copy()
        x[mar] = v
        return x

    def transporte(T, var):
        """(dif, columnas) del error al transportar `var`, medido en la direccion del quiebre."""
        sg, medido = signo[var][fila_e], d[var].to_numpy(float)
        if pt in FAMILIAS:
            # en una familia el signo no puede salir de la etiqueta (slider y curva tienen IVB de signo contrario y
            # la etiqueta migra con la altitud): se usa el del quiebre medio de la propia celda, que no la usa
            sg = np.sign(np.bincount(celda, weights=medido, minlength=K) / (n_a + n_m))[celda]
        x = sg * en_mar(T[var].to_numpy(float), medido)               # arriba lo medido, abajo lo transportado
        obs, pred = bandas(x)
        por_celda = (np.bincount(celda[mar], weights=x[mar], minlength=K) / n_m
                     - np.bincount(celda[alta], weights=x[alta], minlength=K) / n_a)
        return pred - obs, {"nivel_mar": bandas(sg * medido)[1][0], "nivel_alta": obs[0], "predicho": pred[0],
                            "observado": obs[0],
                            "error_abs": float(np.average(np.abs(por_celda), weights=n_a * n_m / (n_a + n_m)))}

    def de_modelo(P, xr):
        """variable -> (numerador, denominador) por pitcheo, a partir de probabilidades y xRV."""
        return {"swstr_pct": (100 * P[:, I["whiff"]], np.ones(len(P))),
                "gb_bip_pct": (100 * P[:, I["ip_gb"]], P[:, IP].sum(1)), "rv100": (100 * xr, np.ones(len(P)))}

    # la mitad del nivel del mar: tal cual, y llevada a la densidad de la mitad alta de su celda
    rho_alta = (np.bincount(celda[alta], weights=d["air_density"].to_numpy(float)[alta], minlength=K) / n_a)[celda[mar]]
    X = d.loc[mar, _cols(df, modelos, "stuff")]
    T = features.transportar(X, rho_alta, beta)
    # referencia para la lectura de las elasticidades: el mismo transporte con beta = 1 ({} = sin elasticidad medida)
    T_beta_1 = features.transportar(X, rho_alta, {}) if pt == "TODOS" else None
    P0, P1 = (predecir(modelos["stuff"], x, fold[fila_e][mar]) for x in (X, T))
    origen = de_modelo(P0, xrv(P0, valores, rho=X["air_density"].to_numpy(float)))
    destino = de_modelo(P1, xrv(P1, valores, rho=rho_alta))
    sin_mover = de_modelo(P0, xrv(P0, valores, rho=rho_alta))        # sin transportar: solo cambia el valor del batazo
    medido = de_modelo(d[[f"p_{c}" for c in OUTCOMES]].to_numpy(float), d["xrv"].to_numpy(float))
    observado = {"swstr_pct": (100 * d["y_whiff"], 1.0), "gb_bip_pct": (100 * d["y_gb"], d["y_bip"]),
                 "rv100": (100 * d["run_value"], 1.0)}
    filas = []
    for prueba in PRUEBAS:
        bloque, modelo, var, unidad = prueba
        nota_fila = nota
        if bloque == "neutralidad":
            a, b = bandas(d["stuff_plus"])
            dif, mas = a - b, {"nivel_mar": b[0], "nivel_alta": a[0]}
        elif bloque == "transporte":
            dif, mas = transporte(T, var)
        else:
            o_a, o_m = bandas(*observado[var])
            # `nulo` = lo que predice la referencia "el aire no mueve el pitcheo": si la prueba no la rechaza, no
            # podia fallar. Pitching+: ningun cambio. Stuff: el mismo modelo sin transportar.
            if modelo == "pitching":
                p_a, p_m = bandas(*medido[var])
                pred, nulo = p_a - p_m, 0.0
            else:
                base = bandas(*map(en_mar, origen[var]))[1]
                pred = bandas(*map(en_mar, destino[var]))[1] - base
                nulo = bandas(*map(en_mar, sin_mover[var]))[1] - base
            obs = o_a - o_m
            _, obs_lo, obs_hi = ic(obs, pitchers)
            _, nulo_lo, nulo_hi = ic(nulo - obs, pitchers)
            con_ic = np.isfinite(nulo_lo) and np.isfinite(nulo_hi)
            dif, mas = pred - obs, {"nivel_mar": o_m[0], "nivel_alta": o_a[0], "predicho": pred[0], "observado": obs[0],
                                    "obs_lo": obs_lo, "obs_hi": obs_hi,
                                    "con_potencia": _fuera_de_cero(nulo_lo, nulo_hi) if con_ic else None}
            if var == "rv100":
                nota_fila = "; ".join(filter(None, [nota, NOTA_RV100]))
        filas.append(_fila(prueba, pt, e, cod, dif, nota_fila, **mas))
        if bloque == "transporte" and pt == "TODOS":
            dif, mas = transporte(T_beta_1, var)
            filas.append(_fila(("transporte_beta_1", modelo, var, unidad), pt, e, cod, dif, NOTA_BETA_1, **mas))
    return filas


def pendiente_neutral(df, cod, W):
    """Neutralidad con TODAS las bandas: (celdas, dif). La comparacion alta
    contra mar solo usa las celdas con pitcheos en las dos puntas y deja fuera
    las categorias intermedias. Aqui entra toda celda pitcher-temporada-tipo con
    2 * MIN_CELDA pitcheos lanzados en parques de mas de una densidad: se
    ajusta UNA pendiente del Stuff+ neutral sobre la densidad media del parque
    (dentro de celda) y dif = pendiente * (RHO_CDMX - RHO_MAR), lo que cambiaria
    el Stuff+ neutral del mismo pitcheo del nivel del mar al Harp Helu si la
    relacion fuera lineal. Se usa la densidad media de cada parque (con el
    dataset oficial, la de su categoria de altitud)."""
    c = df.groupby(["pitcher_id", "season", "pt"], sort=False).ngroup().to_numpy()
    n = np.bincount(c)
    x = df.groupby("stadium")["air_density"].transform("mean").to_numpy(float)
    y = df["stuff_plus"].to_numpy(float)
    dx, dy = x - (np.bincount(c, weights=x) / n)[c], y - (np.bincount(c, weights=y) / n)[c]
    ok = ((n >= 2 * MIN_CELDA) & (np.bincount(c, weights=dx ** 2) > 1e-9))[c]
    e = pd.DataFrame({"fila": np.flatnonzero(ok), "celda": c[ok]})
    pend = razon(sumas(cod[ok], W, (dx * dy)[ok]), sumas(cod[ok], W, (dx ** 2)[ok]))
    return e, (RHO_CDMX - RHO_MAR) * pend


def filas_entre_pitchers(pit, pitchers, W):
    """Dos filas informativas (sin `pasa`) de lo que las pruebas DENTRO del
    mismo pitcher-temporada no ven, entre pitcher-temporadas calificadas:
      corr_stuff_plus_densidad   correlacion del Stuff+ neutral con la densidad
            media del aire donde lanzo. Si no es 0 puede ser composicion (quien
            lanza en altitud) o un transporte incompleto: aqui no se separan.
      cambio_stuff_plus_por_densidad   mismo pitcher, temporadas consecutivas:
            pendiente del cambio de Stuff+ sobre el cambio de su densidad media,
            en puntos (escala entre pitchers) por +0.1 de densidad. Se mide en
            cada corrida. `celdas` = pares."""
    cp, n = pitchers.get_indexer(pit["pitcher_id"]), int(pit["pitcher_id"].nunique())
    p = pit[["pitcher_id", "season", "stuff_plus", "rho"]]
    yy = p.merge(p.assign(season=p["season"] - 1), on=["pitcher_id", "season"], suffixes=("", "_sig"))
    pend = 0.1 * pendiente_boot(yy["rho_sig"] - yy["rho"], yy["stuff_plus_sig"] - yy["stuff_plus"],
                                pitchers.get_indexer(yy["pitcher_id"]), W)
    filas = []
    for var, unidad, celdas, n_pit, v in (
            ("corr_stuff_plus_densidad", "correlacion", len(pit), n, corr_boot(pit["stuff_plus"], pit["rho"], cp, W)),
            ("cambio_stuff_plus_por_densidad", "puntos de Stuff+ entre pitchers por +0.1 de densidad", len(yy),
             int(yy["pitcher_id"].nunique()), pend)):
        v, lo, hi = ic(v, n_pit)
        filas.append({"bloque": "entre_pitchers", "modelo": "stuff", "pt": "TODOS", "variable": var, "unidad": unidad,
                      "celdas": int(celdas), "pitchers": n_pit, "valor": v, "ic_lo": lo, "ic_hi": hi,
                      "nota": NOTA_ENTRE if np.isfinite(v) else "sin datos"})
    return filas


def contrafactual(df, beta, valores, modelos, fold, cod, W, pit=None, pitchers=None):
    """Pruebas de que el contrafactual de altitud es valido. Se hacen sobre
    CELDAS: el mismo pitcheo del mismo pitcher-temporada lanzado en la banda
    alta y en la banda mar (celdas_alta_mar). IC remuestreando pitchers. Cuando
    pasa cada una lo decide veredicto_fila().

      neutralidad  el Stuff+ neutral no debe depender de DONDE se lanzo el
                   pitcheo: valor = Stuff+ medio de la mitad alta menos el de la
                   mitad mar. pasa: |valor| < 1 punto por pitcheo.
                   factor_escala_pitcher = por cuanto multiplicar valor, IC y
                   tolerancia para leerlos en puntos de la escala ENTRE
                   PITCHERS, la de los rankings (un punto por pitcheo son
                   varios de esos).
      transporte   la mitad del nivel del mar, llevada a la densidad de la mitad
                   alta de su celda, debe reproducir el quiebre medido arriba.
                   Se mide EN LA DIRECCION DEL QUIEBRE: cada pitcheo se
                   multiplica por el signo del quiebre medio de su tipo en la
                   liga, y valor = predicho - observado en pulgadas. Negativo =
                   el transporte encoge de mas. Sin el signo, el error de un
                   quiebre hacia el brazo (recta, sinker, cambio) y el de uno
                   hacia el guante (slider, curva) se cancelaban y una
                   elasticidad mal medida pasaba la prueba. error_abs = error
                   absoluto medio entre celdas. pasa: |valor| < 0.5.
      transporte_beta_1   informativa, solo TODOS: la misma medida con beta = 1
                   en vez de la elasticidad medida (que tan lejos queda el
                   quiebre "proporcional a la densidad").
      resultados   cambio (alta - mar) de swing fallado por pitcheo, rodados por
                   batazo y run value por 100: valor = predicho - observado.
                   NO pasa si el IC excluye 0; si lo contiene solo pasa cuando
                   la prueba podia fallar (con_potencia). Sin potencia `pasa`
                   queda vacio: con pocas celdas pasaria cualquier prediccion.
                   obs_lo / obs_hi = IC del cambio observado. En rv100 el valor
                   de los batazos se ajusto con estos mismos datos (ver nota).
                   modelo "pitching": probabilidades de Pitching+ fuera de
                   muestra en los pitcheos TAL COMO SE MIDIERON. con_potencia:
                   el cambio observado se distingue de 0 (predecir "la altitud
                   no cambia nada" habria fallado).
                   modelo "stuff_transportado": lo que predice el contrafactual,
                   es decir el modelo Stuff en la mitad mar transportada menos
                   el mismo modelo en esa mitad sin transportar. con_potencia:
                   habria fallado ese mismo modelo SIN transportar el pitcheo
                   (en rv100 eso deja solo el valor de los batazos en el aire
                   de arriba; en las tasas, ningun cambio).

    Alcances (columna pt), cada uno si tiene al menos MIN_CELDAS celdas:
      TODOS        todas las celdas pitcher-temporada-tipo (siempre se escribe;
                   con pocas celdas dice "sin datos"). En neutralidad hay una
                   segunda fila, variable stuff_plus_pendiente, que usa todas
                   las bandas y no solo alta y mar (pendiente_neutral)
      FF, SI, ...  las celdas de un tipo
      FAM_*        celdas pitcher-temporada con los tipos de la familia JUNTOS
                   (physics.FAMILIAS: rectas FF+SI, quiebre SL+CU, cambio CH+FS;
                   el cutter no esta en ninguna familia y solo tiene su fila
                   por tipo). La etiqueta AutoPitchType migra con la altitud (la
                   misma recta sale mas veces como FF arriba), y eso descuadra
                   los tipos por separado sin que falle el contrafactual
      REPERTORIO   solo neutralidad: celdas pitcher-temporada con todos sus
                   pitcheos. Mide cuanto cambia el Stuff+ de un pitcher por lo
                   que lanza en cada parque (mezcla y etiquetas), no por el aire.

    Con `pit` y `pitchers` (scores por pitcher y el indice de remuestreo) se
    agregan las dos filas del bloque entre_pitchers (ver filas_entre_pitchers)."""
    tipo, temporada, pt = ["pitcher_id", "season", "pt"], ["pitcher_id", "season"], df["pt"].to_numpy()
    signo = {var: df["pt"].map(np.sign(df.groupby("pt")[var].mean())).to_numpy(float) for var in ("ivb", "hb_n")}
    e = celdas_alta_mar(df, tipo)
    alcances = ([("TODOS", e, "")] + [(t, e[pt[e["fila"].to_numpy()] == t], "") for t in TIPOS]
                + [(fam, celdas_alta_mar(df, temporada, np.isin(pt, tipos)), NOTA_FAMILIA)
                   for fam, tipos in FAMILIAS.items()])
    filas = []
    for nombre, celdas, nota in alcances:
        if nombre == "TODOS" or celdas["celda"].nunique() >= MIN_CELDAS:
            filas += _pruebas(df, celdas, nombre, nota, beta, valores, modelos, fold, cod, W, signo)
    rep = celdas_alta_mar(df, temporada)
    if rep["celda"].nunique() >= MIN_CELDAS:
        a, b = por_banda(rep, cod, W, df["stuff_plus"].to_numpy(float)[rep["fila"].to_numpy()])
        filas.append(_fila(PRUEBAS[0], "REPERTORIO", rep, cod, a - b, NOTA_REPERTORIO, nivel_mar=b[0], nivel_alta=a[0]))
    celdas, dif = pendiente_neutral(df, cod, W)
    filas.append(_fila(PRUEBA_PENDIENTE, "TODOS", celdas, cod, dif, NOTA_PENDIENTE)
                 if celdas["celda"].nunique() >= MIN_CELDAS else _fila(PRUEBA_PENDIENTE, "TODOS", celdas, cod))
    if pit is not None and pitchers is not None:
        filas += filas_entre_pitchers(pit, pitchers, W)
    cf = pd.DataFrame(filas, columns=COLS_CF)
    if pit is not None and len(pit):
        # desviacion por pitcheo entre desviacion entre pitcher-temporadas (la de stuff_plus en scores_pitcher.csv)
        m, n = pit["stuff_raw"].to_numpy(float), pit["pitcheos"].to_numpy(float)
        sd = float(np.sqrt(np.average((m - np.average(m, weights=n)) ** 2, weights=n)))
        if sd > 0:
            cf.loc[cf["bloque"] == "neutralidad", "factor_escala_pitcher"] = float(df["stuff_score"].std()) / sd
    orden = {b: i for i, b in enumerate(BLOQUES)}
    return cf.sort_values("bloque", key=lambda s: s.map(orden), kind="stable")


# ── estabilidad y poder predictivo ──────────────────────────────────────────
def estabilidad(df, pit, pitchers, W):
    """Lo que le importa a un director deportivo, con IC remuestreando pitchers:
      split_half_*   consistencia interna: mitades de JUEGOS (los de rango par
                     contra los de rango impar de cada pitcher-temporada) con la
                     correccion de Spearman-Brown. Partir por pitcheos alternos
                     daba casi 1 por construccion: las dos mitades compartian
                     juegos y parques. Sin game_id se parte por pitcheos.
      estabilidad_*  la misma metrica en la temporada t y en la t+1
      predice_rv_*   que predice mejor las carreras de la t+1: el Stuff+ de la t,
                     el Pitching+ de la t o las propias carreras de la t (signo:
                     positivo = predice). dif_*_vs_rv = diferencia contra las
                     carreras; solo es "mejor" si su IC excluye 0."""
    out = {}
    llave = [df["pitcher_id"], df["season"]]
    gid = juegos(df)
    por_juego = gid is not None
    if por_juego:
        juego = pd.Series(pd.factorize(gid, sort=True)[0], index=df.index)
        mitad = (juego.groupby(llave).rank(method="dense") % 2).astype(int)
    else:
        mitad = df.groupby(["pitcher_id", "season"]).cumcount() % 2
    cols = {"stuff": "stuff_score", "location": "location_score", "pitching": "pitching_score",
            "run_value_observado": "run_value"}
    g = df[list(cols.values())].groupby(llave + [mitad.rename("mitad")])
    n = g.size().unstack("mitad").reindex(columns=[0, 1]).fillna(0)
    m = g.mean().unstack("mitad").reindex(columns=pd.MultiIndex.from_product([list(cols.values()), [0, 1]]))
    m = m[(n.min(axis=1) >= MIN_PITCHES_PITCHER // 2).to_numpy()]
    cod = pitchers.get_indexer(m.index.get_level_values("pitcher_id"))
    for nombre, col in cols.items():
        r = corr_boot(m[(col, 0)], m[(col, 1)], cod, W) if len(m) else np.full(len(W), np.nan)
        v, lo, hi = ic(np.clip(razon(2 * r, 1 + r), -1.0, 1.0))      # Spearman-Brown
        out[f"split_half_{nombre}"], out[f"split_half_{nombre}_ci"] = round(v, 3), [round(lo, 3), round(hi, 3)]
    out["split_half_por"], out["split_half_pitcher_temporadas"] = ("juego" if por_juego else "pitcheo"), int(len(m))

    p = pit[["pitcher_id", "season", "stuff_plus", "pitching_plus", "rv100"]]
    yy = p.merge(p.assign(season=p["season"] - 1), on=["pitcher_id", "season"], suffixes=("", "_sig"))
    cod = pitchers.get_indexer(yy["pitcher_id"])
    r = lambda a, b, signo=1.0: signo * corr_boot(yy[a], yy[b], cod, W)
    medidas = {"estabilidad_stuff_plus_t_t1": r("stuff_plus", "stuff_plus_sig"),
               "estabilidad_pitching_plus_t_t1": r("pitching_plus", "pitching_plus_sig"),
               "estabilidad_rv_observado_t_t1": r("rv100", "rv100_sig"),
               "predice_rv_t1__con_stuff_plus_t": r("stuff_plus", "rv100_sig", -1.0),
               "predice_rv_t1__con_pitching_plus_t": r("pitching_plus", "rv100_sig", -1.0),
               "predice_rv_t1__con_rv_observado_t": r("rv100", "rv100_sig")}
    medidas["dif_stuff_vs_rv"] = medidas["predice_rv_t1__con_stuff_plus_t"] - medidas["predice_rv_t1__con_rv_observado_t"]
    medidas["dif_pitching_vs_rv"] = (medidas["predice_rv_t1__con_pitching_plus_t"]
                                     - medidas["predice_rv_t1__con_rv_observado_t"])
    for nombre, v in medidas.items():
        v, lo, hi = ic(v, yy["pitcher_id"].nunique())
        out[nombre], out[f"{nombre}_ci"] = round(v, 3), [round(lo, 3), round(hi, 3)]
    out["pares"], out["pitchers"] = int(len(yy)), int(yy["pitcher_id"].nunique())
    return out


def acuerdo_modelos(df, beta, valores, modelos, sd):
    """El error DEL MODELO, que ningun intervalo de scores_pitcher.csv incluye.
    Todos los modelos de fold califican los MISMOS pitcheos (pitcher-temporadas
    calificadas completas, al azar hasta sumar N_ACUERDO pitcheos) y se
    comparan sus promedios por pitcher-temporada, en puntos de la escala entre
    pitchers (`sd` = desviacion entre pitchers del score neutral). Para el
    Stuff+ neutral (stuff) y para delta_altitud (delta):
      acuerdo_modelos_*    correlacion media entre pares de modelos
      error_modelo_*_pts   desviacion de la diferencia entre dos modelos entre
                           raiz de 2: el error tipico de UN modelo. Es UNA
                           desviacion: se compara con el error estandar del
                           intervalo (semi-ancho de stuff_lo/_hi y de
                           delta_lo/_hi entre 1.96)
      desfase_modelos_*_pts   nivel medio de cada modelo en esos pitcheos menos
                           el promedio de los modelos (uno por fold). A cada
                           pitcher lo califica el modelo de su fold: una
                           diferencia de ese tamano entre pitchers de folds
                           distintos es del modelo, no del pitcher
    Es ORIENTATIVO y no sustituye a un ajuste repetido: los modelos comparten
    buena parte de sus filas de entrenamiento (eso los acerca) y cada pitcher
    estuvo en el entrenamiento de todos menos uno (eso los separa). No ajusta
    ningun modelo; cuesta tres predicciones por modelo sobre la muestra."""
    out = {"acuerdo_modelos_k": int(len(modelos["stuff"])), "acuerdo_modelos_pitcher_temporadas": 0,
           "acuerdo_modelos_pitcheos": 0}
    for nombre in ("stuff", "delta"):
        out.update({f"acuerdo_modelos_{nombre}": np.nan, f"error_modelo_{nombre}_pts": np.nan,
                    f"desfase_modelos_{nombre}_pts": []})
    g = df.groupby(["pitcher_id", "season"], sort=False).ngroup().to_numpy()
    n = np.bincount(g)
    orden = np.random.default_rng(RANDOM_STATE).permutation(np.flatnonzero(n >= MIN_PITCHES_PITCHER))
    elegidos = orden[np.cumsum(n[orden]) - n[orden] < N_ACUERDO]
    if len(elegidos) < MIN_PARES or len(modelos["stuff"]) < 2 or not sd > 0:
        return out
    filas = np.flatnonzero(np.isin(g, elegidos))
    grupo = pd.factorize(g[filas])[0]
    n_g = np.bincount(grupo).astype(float)
    X, den = df[_cols(df, modelos, "stuff")].iloc[filas], valores["densidad"]
    aire = {a: features.transportar(X, rho, beta) for a, rho in zip(AIRES, (den["rho_liga"], RHO_CDMX, RHO_MAR))}
    nivel, delta = [], []
    for k in sorted(modelos["stuff"]):
        P = {a: proba(modelos["stuff"][k], d) for a, d in aire.items()}
        costo = xrv(P["cdmx"], valores, rho=RHO_CDMX) - xrv(P["mar"], valores, rho=RHO_MAR)
        nivel.append(np.bincount(grupo, weights=-xrv(P["neutral"], valores)) / n_g)
        delta.append(np.bincount(grupo, weights=-costo) / n_g)
    pares = [(a, b) for a in range(len(nivel)) for b in range(a + 1, len(nivel))]
    for nombre, v in (("stuff", nivel), ("delta", delta)):
        v = PLUS_SCALE * np.array(v) / sd                 # (modelos, pitcher-temporadas) en puntos entre pitchers
        medias = v @ n_g / n_g.sum()
        out[f"acuerdo_modelos_{nombre}"] = round(float(np.mean([np.corrcoef(v[a], v[b])[0, 1] for a, b in pares])), 3)
        out[f"error_modelo_{nombre}_pts"] = round(float(np.sqrt(np.mean([np.var(v[a] - v[b]) for a, b in pares]) / 2)), 2)
        out[f"desfase_modelos_{nombre}_pts"] = (medias - medias.mean()).round(2).tolist()
    out["acuerdo_modelos_pitcher_temporadas"], out["acuerdo_modelos_pitcheos"] = int(len(elegidos)), int(len(filas))
    return out


# ── lo que necesita el simulador ────────────────────────────────────────────
def escalas(df, beta, valores, modelos, fold, esc_pit, esc_ars, pit):
    """models/scale.json: media y desviacion de cada score por pitcheo, entre
    pitchers y entre arsenales, y la REJILLA del simulador: el score medio de
    la liga con todos los pitcheos llevados a cada densidad (100 = el pitcheo
    promedio lanzado en ese aire). La rejilla se calcula con el modelo que usa
    el simulador (models/stuff.joblib, ajustado con todos los datos). Ademas
    de 15 densidades parejas lleva las de referencia y los dos limites donde
    el valor de los batazos deja de cambiar (model.xrv): ahi la curva tiene un
    quiebre y, sin esos puntos, la interpolacion lineal lo recorta y se
    equivoca cerca de el (por ejemplo en la densidad del Harp Helu)."""
    e = {s: {"mu": float(df[f"{s}_score"].mean()), "sd": float(df[f"{REFERENCIA.get(s, s)}_score"].std())}
         for s in PUNTAJES}
    idx = _muestra(df)
    X, f = df[_cols(df, modelos, "stuff")].iloc[idx], fold[idx]
    ruta = config.MODELS / "stuff.joblib"
    final = joblib.load(ruta) if ruta.exists() else None
    if final is None:
        print(f"AVISO: no existe models/stuff.joblib (corre  {config.comando('final')}): la rejilla del simulador "
              "se calcula con los modelos de fold")

    def en_aire(rho):
        """Score de los pitcheos de la muestra llevados a la densidad rho y valuados ahi."""
        d = features.transportar(X, rho, beta)
        return -xrv(proba(final, d) if final is not None else predecir(modelos["stuff"], d, f), valores, rho=rho)

    den = valores["densidad"]
    nodos = sorted({*np.linspace(0.72, 1.00, 15).round(4).tolist(), RHO_CDMX, RHO_MAR,
                    *(float(den[k]) for k in ("rho_liga", "rho_min", "rho_max"))})
    e["rejilla"] = [{"rho": r, "mu": float(en_aire(r).mean())} for r in nodos]
    neutral = en_aire(den["rho_liga"])
    e["simulador"] = {"modelo": "stuff.joblib (todos los datos)" if final is not None else "modelos de fold",
                      "pitcheos": int(len(idx)), "mu": float(neutral.mean()), "sd": float(neutral.std()),
                      # percentiles del Stuff+ del simulador a densidad de liga, con la desviacion de "stuff"
                      "percentiles_plus": np.percentile(plus(neutral, neutral.mean(), e["stuff"]["sd"]),
                                                        np.arange(101)).round(3).tolist()}
    e["pitcher"], e["arsenal"] = esc_pit, esc_ars
    e["percentiles_pitcher_stuff"] = np.percentile(pit["stuff_raw"], np.arange(101)).tolist()
    return e


def defaults_simulador(df, modelos, rho_liga):
    """models/sim_defaults.json: por tipo de pitcheo, la mediana de cada variable
    del modelo Stuff y de la recta primaria de quienes lanzan ese tipo (fb_velo,
    y fb_ivb0 / fb_hb0 a la densidad de referencia RHO_REF), mas el tipo de
    recta mas comun. Es lo que el simulador pone cuando el usuario no da un dato."""
    feats = next(iter(modelos["stuff"].values())).feats_
    num = [c for c in dict.fromkeys(feats + ["release_height", "rel_side_n", "extension", "spin_rate", "tunnel_dist",
                                             "velo_diff_vs_fb", "fb_velo", "fb_ivb0", "fb_hb0"])
           if c != "pt" and c in df]
    g = df.groupby("pt")
    med, n = g[num].median(), g.size()
    moda = g["fb_type"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
    out = {pt: {**{c: float(v) for c, v in med.loc[pt].items()}, "fb_type": moda[pt], "pitcheos": int(n[pt])}
           for pt in med.index}
    out["liga"] = {"rho_liga": rho_liga}
    return out


# ── corrida ─────────────────────────────────────────────────────────────────
def guardar_predicciones(df):
    """El entregable por pitcheo, con los nombres OFICIALES del diccionario para
    poder unirlo de vuelta por PitchUID. Sin estadio, sin juego y sin nombres.
    Solo lleva los pitcheos calificados: los que features.py descarta en su
    limpieza no tienen prediccion, y se dice en la consola."""
    oficial = {"pitch_id": "PitchUID", "pitcher_id": "pitcher_anon_id", "season": "year",
               "pitch_type": "AutoPitchType", "stuff_plus": "stuff_plus", "stuff_cdmx_plus": "stuff_plus_cdmx",
               "stuff_mar_plus": "stuff_plus_mar", "delta_altitud": "delta_altitud",
               "location_plus": "location_plus", "pitching_plus": "pitching_plus", "xrv": "xRV",
               **{f"p_{c}": f"p_{c}" for c in OUTCOMES},
               # objetivos del reto que son suma de clases (p_chase solo existe fuera de la zona)
               "p_swing": "p_swing", "p_chase": "p_chase", "p_linea_o_elevado_duro": "p_linea_o_elevado_duro"}
    decimales = {c: 5 if c == "xrv" or c.startswith("p_") else 2
                 for c in oficial if c in df and df[c].dtype.kind == "f"}
    ruta = config.OUT / "predicciones_por_pitcheo.csv.gz"
    (df[[c for c in oficial if c in df]].round(decimales).rename(columns=oficial)
     .to_csv(ruta, index=False, compression={"method": "gzip", "compresslevel": 6, "mtime": 0}))
    print(f"\n{ruta.name}: {len(df):,} pitcheos con prediccion, {ruta.stat().st_size / 1e6:.1f} MB. Los pitcheos que "
          "features.py descarta en su limpieza (su consola dice cuantos y por que) no estan en el archivo: al unirlo "
          "por PitchUID quedan sin prediccion.")


def main():
    t0 = time.time()
    pd.set_option("display.width", 250)
    df, beta, valores, modelos, fold = cargar()
    print(f"{len(df):,} pitcheos | {df['pitcher_id'].nunique():,} pitchers | modo categoria | "
          f"{len(modelos['stuff'])} folds" + (" | MODO RAPIDO" if RAPIDO else ""))
    print("elasticidades usadas (ivb / hb):", {pt: tuple(round(b.get(v, 1.0), 2) for v in ("ivb", "hb_n"))
                                               for pt, b in beta.items()})
    batazos = calificar(df, beta, valores, modelos, fold)
    a_escala(df)
    print(f"scores por pitcheo fuera de muestra ({time.time() - t0:.0f}s)", flush=True)

    cod, pitchers, W = remuestreo(df["pitcher_id"])
    pit, esc_pit = agregar(df, ["pitcher_id", "season"], MIN_PITCHES_PITCHER)
    ars, esc_ars = agregar(df, ["pitcher_id", "season", "pt"], MIN_PITCHES_ARSENAL)
    pit.to_csv(config.OUT / "scores_pitcher.csv", index=False)
    ars.to_csv(config.OUT / "scores_arsenal.csv", index=False)
    print(f"{len(pit):,} pitcher-temporadas con {MIN_PITCHES_PITCHER}+ pitcheos y {len(ars):,} arsenales con "
          f"{MIN_PITCHES_ARSENAL}+ | confiabilidad mediana: "
          + ", ".join(f"{s} {pit[f'{s}_fiab'].median():.2f}" for s in CON_INTERVALO))
    if juegos(df) is not None:
        print("  ruido de cada promedio con el juego como unidad; correlacion intraclase por juego (pitcher): "
              + ", ".join(f"{s} {esc_pit[s]['icc_juego']:.3f}" for s in CON_INTERVALO + ("delta_altitud",)))
        if df["game_id"].isna().any():
            print(f"AVISO: pitcheos sin game_id: {int(df['game_id'].isna().sum()):,} "
                  "(cada uno cuenta como un juego propio)")
    else:
        print("AVISO: falta game_id (no viene, o le falta a mas de "
              f"{100 * MAX_SIN_JUEGO:g}% de los pitcheos): la confiabilidad y los intervalos por pitcher suponen "
              "pitcheos independientes y salen optimistas")

    tipo = por_tipo(df, cod, W)
    tipo.to_csv(config.OUT / "altitude_by_pitch_type.csv", index=False)
    print("\nEfecto de la altitud por tipo (delta = puntos por pitcheo, relativo; costo = carreras por 100, absoluto):")
    print(tipo.round(2).to_string(index=False))
    parque = efecto_parque(df, pit, cod, W, valores, batazos)
    parque.update(costo_entre_bandas(df, beta, valores, modelos, fold, parque["costo_cdmx_vs_mar_rv100"]))
    (config.OUT / "altitude_park_effect.json").write_text(json.dumps(parque, indent=1))
    print(f"\nCosto de lanzar en CDMX vs nivel del mar: {parque['costo_cdmx_vs_mar_rv100']:+.2f} carreras por 100 "
          f"pitcheos [{parque['ic95'][0]:+.2f}, {parque['ic95'][1]:+.2f}] = movimiento "
          f"{parque['por_movimiento_rv100']:+.2f} + valor de los batazos {parque['por_valor_batazos_rv100']:+.2f} "
          f"| entre las densidades observadas en casa ({parque['rho_casa_obs']:.3f}) y en banda mar "
          f"({parque['rho_mar_obs']:.3f}): modelado {parque['costo_entre_bandas_rv100']:+.2f}, observado "
          f"{parque['observado_casa_menos_mar_rv100']:+.2f}")
    parques = tabla_parques(df)
    parques.to_csv(config.OUT / "park_scores.csv", index=False)
    print("\nScores por grupo de parques (plus en escala por pitcheo):")
    print(parques.round(3).to_string(index=False))

    cf = contrafactual(df, beta, valores, modelos, fold, cod, W, pit, pitchers)
    cf.to_csv(config.OUT / "val_contrafactual.csv", index=False)
    print("\nPruebas del contrafactual (lo mismo del mismo pitcher en banda alta y en banda mar; transporte en la "
          "direccion del quiebre):")
    print(cf.drop(columns=["unidad", "nota"]).round(3).to_string(index=False))
    todas = cf[(cf["pt"] == "TODOS") & cf["pasa"].notna()]
    sin_potencia = int(((cf["pt"] == "TODOS") & (cf["bloque"] == "resultados") & cf["pasa"].isna()
                        & cf["valor"].notna()).sum())
    # la corrida de entrega se detiene con la MISMA regla que `pasa` (veredicto_fila), y solo por neutralidad o transporte
    no_pasan = todas[todas["bloque"].isin(list(TOLERANCIA)) & ~todas["pasa"].astype(bool)]
    if todas.empty:
        print(f"Aceptacion: sin datos (menos de {MIN_CELDAS} celdas con {MIN_CELDA}+ pitcheos arriba y abajo)")
    else:
        print("Aceptacion con todas las celdas: " + " | ".join(
            [f"{b}: pasan {int(g['pasa'].astype(bool).sum())} de {len(g)}" for b, g in todas.groupby("bloque", sort=False)]
            + ([f"resultados sin potencia para fallar (sin veredicto): {sin_potencia}"] if sin_potencia else [])))
    if len(no_pasan):
        # aqui solo se avisa: el alto de ESTRICTO va al final de main, con todos los archivos ya escritos
        _aviso_grave("el contrafactual de altitud NO pasa la prueba de neutralidad o la de transporte ("
                     + "; ".join(f"{r.bloque} {r.variable} = {r.valor:+.2f}, tolerancia {TOLERANCIA[r.bloque]:g}"
                                 for r in no_pasan.itertuples())
                     + "): stuff_cdmx, stuff_mar y delta_altitud no son confiables hasta revisarlo "
                     "(out/val_contrafactual.csv)", detener=False)
    val = estabilidad(df, pit, pitchers, W)
    val.update(acuerdo_modelos(df, beta, valores, modelos, esc_pit["stuff"]["sd"]))
    (config.OUT / "val_stability.json").write_text(json.dumps(val, indent=1))
    print(f"\nConsistencia entre mitades de {val['split_half_por']}s ({val['split_half_pitcher_temporadas']} "
          "pitcher-temporadas): " + ", ".join(f"{k} {val['split_half_' + k]:.2f}"
                                              for k in ("stuff", "location", "pitching", "run_value_observado")))
    print(f"De una temporada a la siguiente ({val['pares']} pares, {val['pitchers']} pitchers): "
          + " | ".join(f"{k} {val[k]:+.2f} [{val[k + '_ci'][0]:+.2f}, {val[k + '_ci'][1]:+.2f}]"
                       for k in val if k.startswith(("estabilidad", "predice", "dif")) and not k.endswith("_ci")))
    semi = {k: (pit[f"{k}_hi"] - pit[f"{k}_lo"]).mean() / 2 for k in ("stuff", "delta")}
    print(f"Los {val['acuerdo_modelos_k']} modelos de fold sobre los mismos pitcheos "
          f"({val['acuerdo_modelos_pitcher_temporadas']} pitcher-temporadas; puntos entre pitchers): "
          + " | ".join(f"{nombre}: r {val['acuerdo_modelos_' + k]:.2f}, error de modelo "
                       f"{val['error_modelo_' + k + '_pts']:.1f} (error estandar medio del intervalo publicado "
                       f"{semi[k] / 1.96:.1f}), desfase de nivel por fold {val['desfase_modelos_' + k + '_pts']}"
                       for nombre, k in (("Stuff+", "stuff"), ("delta_altitud", "delta"))))

    escala = escalas(df, beta, valores, modelos, fold, esc_pit, esc_ars, pit)
    (config.MODELS / "scale.json").write_text(json.dumps(escala, indent=1))
    (config.MODELS / "sim_defaults.json").write_text(
        json.dumps(defaults_simulador(df, modelos, valores["densidad"]["rho_liga"]), indent=1))
    guardar_predicciones(df)
    del df["stadium"]            # solo repite altitude_category
    # archivo interno para insights, dashboard y simulador (decimales en 32 bits: pesa la mitad)
    df.astype({c: "float32" for c in df.columns if df[c].dtype == "float64"}).to_parquet(
        config.OUT / "scores_pitch.parquet", index=False)
    print(f"\nscore.py listo en {time.time() - t0:.0f}s: scores_pitch.parquet ({df.shape[1]} columnas), "
          "scores_pitcher.csv, scores_arsenal.csv, predicciones_por_pitcheo.csv.gz y validaciones")
    if len(no_pasan) and config.ESTRICTO:
        raise SystemExit(
            "ESTRICTO=1: el contrafactual de altitud no paso su prueba de aceptacion (ver el AVISO de arriba).\n"
            "  1. Revisa que las categorias de altitud de config.ALTITUDE_M correspondan a tus datos (features.py\n"
            "     imprime las que encontro). Si corriges algo:  python run_all.py --estricto\n"
            "  2. Si estan bien no hay nada que repetir: TODOS los archivos de score.py quedaron escritos y el\n"
            "     documento reporta la prueba como no superada. Termina la entrega con:\n"
            "     python run_all.py --estricto --desde insights")


if __name__ == "__main__":
    main()
