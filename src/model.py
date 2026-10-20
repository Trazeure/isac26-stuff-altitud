"""
MODELOS — Stuff+, Location+ y Pitching+ como clasificadores de 10 resultados.

IDEA CENTRAL: en vez de predecir el run value observado (muy ruidoso), cada
modelo predice la PROBABILIDAD de los 10 resultados posibles de un pitcheo
(bola, strike cantado, swing fallado, foul, golpe, rodado, linea, elevado
suave, elevado duro, pop). El valor esperado en carreras es

      xRV = suma_c  P(c) * valor_c

y valor_c sale de la propia liga (outcome_values):
  - Stuff+ no ve la cuenta: cada resultado vale lo que vale en la MEZCLA de
    cuentas de la liga ("neutral").
  - Location+ y Pitching+ si la ven: el valor del resultado EN esa cuenta.
  - Los batazos valen distinto segun el aire: una recta en la densidad del
    aire por cada clase de batazo (un elevado duro cuesta mas en altitud).

Etapas (python model.py <etapa>):
  valores  models/outcome_values.json (no ajusta ningun modelo)
  validar  elige el numero de arboles, hace el cross-fit por pitcher y escribe
           las validaciones (out/val_*.csv). Los modelos de cada fold son los
           que califican en score.py: a cada pitcher lo califica un modelo que
           nunca lo vio.
  final    models/stuff.joblib con todos los datos: SOLO para el simulador
  todo     las tres

Algoritmo: gradient boosting por histogramas de scikit-learn. LightGBM es
opcional (config.BACKEND = "lightgbm").
"""
import json
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# Las rutas (OUT, MODELS) se leen como config.X al usarse, para que una prueba
# pueda cambiarlas antes de llamar a una etapa.
import config
import features
from config import (BACKEND, ITERS_MAX, N_FOLDS, MAX_FILAS_FOLD, RANDOM_STATE, N_THREADS, RAPIDO,
                    RHO_CDMX, RHO_MAR)
from features import OUTCOMES, STUFF_FEATS, LOCATION_FEATS, PITCHING_FEATS, CATEGORICAL, TIPOS

# categorical_features="from_dtype" (asi entra `pt` a los modelos) existe desde
# scikit-learn 1.4. Con una version anterior el primer ajuste truena, ya
# corridos features y physics, con un mensaje en ingles que no dice que hacer.
if tuple(int(x) for x in re.findall(r"\d+", sklearn.__version__)[:2]) < (1, 4):
    raise SystemExit(f"scikit-learn {sklearn.__version__} es muy viejo para este proyecto (hace falta 1.4 o mas "
                     "nuevo). Actualizalo con:  python -m pip install -r requirements.txt")

if BACKEND == "lightgbm":
    # CAMINO OPCIONAL Y SIN PROBAR: todo lo entregado se corrio con "sklearn".
    try:
        import lightgbm as lgb
    except Exception as e:                      # noqa: BLE001
        raise SystemExit(f"config.BACKEND = 'lightgbm' pero no se pudo importar lightgbm ({e}). "
                         "Instalalo (requirements-opcional.txt) o deja BACKEND = 'sklearn'.")
elif BACKEND != "sklearn":
    raise SystemExit(f"config.BACKEND = '{BACKEND}' no existe: usa 'sklearn' o 'lightgbm'")

SUBMODELOS = {
    "stuff":    STUFF_FEATS + CATEGORICAL,
    "location": LOCATION_FEATS + CATEGORICAL,
    "pitching": PITCHING_FEATS + CATEGORICAL,
}
IN_PLAY = [c for c in OUTCOMES if c.startswith("ip_")]
I = {c: i for i, c in enumerate(OUTCOMES)}                 # posicion de cada clase en las matrices de probabilidad
IP = [I[c] for c in IN_PLAY]
SWING = [I["whiff"], I["foul"]] + IP                       # el bateador hizo swing
TOMA = [I["ball"], I["called_strike"], I["hbp"]]           # no hizo swing
N_BOOT = 170 if RAPIDO else 500                            # remuestreos de los intervalos
# En el parque no visto tambien se remuestrean los juegos de ENTRENAMIENTO, y
# cada remuestreo recalcula los valores de los resultados (outcome_values sobre
# todas las filas de entrenamiento). De ahi sale una desviacion estandar, no un
# percentil: con 100 basta.
N_BOOT_VALORES = 100
# Minimo de arboles de un modelo. En la prueba rapida se achica igual que el
# tope (ITERS_MAX): con 30% de los pitchers el minimo de Stuff puede caer
# debajo de 30 arboles, y forzar mas arboles de los que pide la validacion lo
# sobreajusta.
PISO_ITERS = 10 if RAPIDO else 30
# Cortes 85/15 de pitchers con los que se elige el numero de arboles (se
# promedian sus curvas; los que no estan aqui usan uno). Stuff lleva 3: su
# minimo es temprano y angosto, y con un solo corte el entero dependia de que
# pitchers caian en la validacion, y con el el log-loss y la dispersion del
# xRV. En Location y Pitching la curva es plana cerca del minimo: un corte
# basta, y cada corte de mas es otro ajuste con ITERS_MAX arboles.
CORTES_ITERS = {"stuff": 3}
TROZO = 200_000                                            # filas por llamada al predecir (memoria)
MIN_CORTE = 1000                                           # filas minimas de cada lado de un corte de validacion

# Lo que este modulo usa de la tabla de modelado. Con los mas de 600 mil pitcheos
# del archivo oficial la tabla completa (~130 columnas) no debe copiarse en cada
# corte: se carga una vez y se dejan solo estas columnas.
COLS_VALOR = ["outcome", "count", "run_value", "stadium", "air_density"]
ETIQUETAS = COLS_VALOR + ["pitcher_id", "season", "banda", "es_casa", "in_zone", "y_chase"]

# Los seis modelos que se comparan en cada validacion y como se valua su xRV:
#   "neutral": valores a cuenta promedio y densidad de liga, contra el run value
#              llevado a cuenta promedio (rv_cuenta_media). Es la escala de Stuff+.
#   "cuenta":  valores de la cuenta y de la densidad del pitcheo, contra el run
#              value real. Es la escala de Location+ / Pitching+ y del nulo del reto.
VALUACION = {"nulo_global": "neutral", "nulo_cuenta": "cuenta", "stuff": "neutral",
             "stuff_cuenta": "cuenta", "location": "cuenta", "pitching": "cuenta"}
CONDICIONALES = ["auc_whiff_swing", "logloss_whiff_swing", "auc_cs_take", "auc_gb_bip"]
COLS_HOLDOUT = (["modelo", "fold", "valuacion", "n", "logloss", "brier_whiff", "rmse_xrv", "corr_xrv"]
                + CONDICIONALES)


def _cols_comparacion(medida, nulo):
    """Columnas de una comparacion contra un nulo: valor y limites del IC 95%."""
    base = f"{medida}_vs_{nulo}"
    return [base + ("_pct" if medida == "mejora" else ""), base + "_ci_lo", base + "_ci_hi"]


COLS_RESUMEN = (["modelo", "valuacion", "n", "pitchers", "logloss", "brier_whiff", "rmse_xrv", "corr_xrv"]
                + CONDICIONALES
                + [c for nulo in ("global", "cuenta") for medida in ("mejora", "delta_brier", "delta_rmse")
                   for c in _cols_comparacion(medida, nulo)] + ["backend", "iters"])
COLS_PARQUE = ["escenario", "submodelo", "valuacion", "n", "n_train", "logloss", "xrv_medio_pred_x100",
               "rv_medio_real_x100", "sesgo_por_100", "sesgo_ci_lo", "sesgo_ci_hi", "sesgo_ee_valores", "n_juegos",
               "agrupado_por", "rho_test", "rho_train_min", "rho_train_max", "n_puntos"]
COLS_CAL = ["objetivo", "modelo", "grupo", "bin", "n", "pred_media", "obs_media"]
COLS_CAL_RES = ["objetivo", "modelo", "grupo", "n", "tasa_base", "ece", "ece_ruido", "pendiente", "pendiente_ci_lo",
                "pendiente_ci_hi", "brier"]
COLS_H4 = ["conjunto", "n_vars", "auc", "logloss", "auc_sd", "logloss_sd", "n", "referencia",
           "delta_auc", "delta_auc_lo", "delta_auc_hi", "delta_logloss", "delta_logloss_lo", "delta_logloss_hi",
           "delta_auc_vs_spin", "delta_auc_vs_spin_lo", "delta_auc_vs_spin_hi",
           "delta_logloss_vs_spin", "delta_logloss_vs_spin_lo", "delta_logloss_vs_spin_hi"]
COLS_IMP = ["variable", "delta_logloss", "delta_logloss_sd"]

# H4 ("el spin solo no basta"): conjuntos de variables cada vez mas ricos para
# predecir el swing fallado. Cada uno se compara con su `referencia` (el paso,
# que depende del orden en que entran las variables) y, los que van despues
# del spin, tambien con H4_BASE: ese es el contraste de la hipotesis.
H4_BASE = "+spin"
_BASE = ["pt", "spin_rate", "velocity"]
_MOV = ["ivb", "hb_n", "mov_total", "mov_per_krpm"]
_REL = ["release_height", "rel_side_n", "extension"]
H4 = {  # conjunto: (variables, referencia)
    "tipo":           (["pt"], None),
    "+spin":          (["pt", "spin_rate"], "tipo"),
    "+velocidad":     (_BASE, "+spin"),
    "+movimiento":    (_BASE + _MOV, "+velocidad"),
    "forma_sin_spin": (["pt", "velocity", "ivb", "hb_n", "mov_total"], "+movimiento"),
    "+release":       (_BASE + _MOV + _REL, "+movimiento"),
    "+diferenciales": (SUBMODELOS["stuff"], "+release"),
}


def _y(df):
    """Clase de resultado de cada fila como entero (posicion en OUTCOMES). Un
    resultado que no es ninguna de las clases detiene la corrida: con su -1 el
    modelo entrenaba una clase de mas, sin avisar, y esa probabilidad caia en
    la ultima columna (ip_pu)."""
    y = pd.Index(OUTCOMES).get_indexer(df["outcome"])
    if (y < 0).any():
        raras = df["outcome"][y < 0].value_counts(dropna=False).head(8).to_dict()
        raise SystemExit(f"{int((y < 0).sum()):,} pitcheos traen un resultado que no es ninguna de las clases del "
                         f"modelo: {features._ascii(raras)}. Pasa cuando PITCH_CALL_MAP o HIT_TYPE_MAP (config.py) "
                         "mandan una etiqueta a un valor nuevo: los unicos validos son ball, called_strike, whiff, "
                         "foul, in_play y hbp en el primero, y gb, ld, fb, pu y bunt en el segundo. Corrigelo y "
                         f"vuelve a correr  {config.comando('features')}")
    return y


def cargar():
    """La tabla de modelado, SOLO por features.cargar_features (asi el
    entrenamiento ve los mismos diferenciales que score.py y el simulador)."""
    df = features.cargar_features()
    faltan = [c for c in ETIQUETAS if c not in df]
    if faltan:
        raise SystemExit(f"A out/features.parquet le faltan columnas {faltan}: vuelve a correr  "
                         + config.comando("features"))
    cols = dict.fromkeys(PITCHING_FEATS + CATEGORICAL + ETIQUETAS + ["game_id"])
    return df[[c for c in cols if c in df]].copy()


# ── valor de cada resultado, estimado de la propia LMB ──────────────────────
def _recta(x, y, w):
    """Pendiente de minimos cuadrados ponderados; NaN sin dos puntos de x distinta."""
    x, y, w = (np.asarray(a, dtype=float) for a in (x, y, w))
    if len(x) < 2:
        return np.nan
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    sxx = np.sum(w * (x - xm) ** 2)
    return float(np.sum(w * (x - xm) * (y - ym)) / sxx) if sxx > 1e-12 else np.nan


def _batazos_por_densidad(d):
    """Valor de cada clase de batazo como RECTA en la densidad del aire: valor
    medio de la clase en cada estadio (o categoria de altitud) contra la
    densidad media de ese estadio, pesando por el numero de batazos.
    El IC de la pendiente quita un estadio a la vez (jackknife, igual que
    physics.py): con menos de 6 estadios es solo orientativo."""
    rho = d.groupby("stadium")["air_density"].mean()
    t = (d[d["outcome"].isin(IN_PLAY)].groupby(["stadium", "outcome"])["run_value"]
         .agg(["mean", "size"]).reset_index())
    t["rho"] = t["stadium"].map(rho)
    K = len(rho)
    pendiente, ic = {}, {}
    for c in IN_PLAY:
        g = t[t["outcome"] == c]
        b = _recta(g["rho"], g["mean"], g["size"])
        sin_uno = np.array([_recta(*(g.loc[g["stadium"] != s, col] for col in ("rho", "mean", "size")))
                            for s in rho.index])
        lo = hi = np.nan
        if K >= 3 and np.isfinite(b) and np.isfinite(sin_uno).all():
            ee = np.sqrt((K - 1) / K * np.sum((sin_uno - sin_uno.mean()) ** 2))
            lo, hi = b - stats.t.ppf(0.975, K - 1) * ee, b + stats.t.ppf(0.975, K - 1) * ee
        pendiente[c], ic[c] = (float(b) if np.isfinite(b) else 0.0), [float(lo), float(hi)]
    return {"rho_liga": float(d["air_density"].mean()),
            # limites para valuar: las densidades vistas, ampliadas a las dos de
            # referencia del proyecto (Harp Helu y nivel del mar de config.py)
            "rho_min": float(min(d["air_density"].min(), RHO_CDMX)),
            "rho_max": float(max(d["air_density"].max(), RHO_MAR)),
            # entre que densidades medias de estadio se ajusto la recta: valuar
            # fuera de este rango es extrapolar
            "rho_obs": [float(rho.min()), float(rho.max())],
            "pendiente": pendiente, "pendiente_ic": ic, "ic_fiable": bool(K >= 6), "n_puntos": int(K)}


def outcome_values(df):
    """Valor en carreras de cada resultado (positivo = bateador):
      media        promedio del run value de la clase
      cuenta       lo mismo dentro de cada cuenta (celda vacia = media)
      peso_cuenta  fraccion de los pitcheos de la liga en cada cuenta
      neutral      valor de la clase en la MEZCLA de cuentas de la liga: lo usa
                   Stuff+, que no sabe en que cuenta se lanzo el pitcheo
      densidad     recta del valor de cada batazo en la densidad del aire:
                   valor(rho) = neutral + pendiente * (rho - rho_liga)
    En las validaciones se llama con las filas de ENTRENAMIENTO de cada corte."""
    d = df[COLS_VALOR]
    d = d[np.isfinite(d["run_value"])]
    media = d.groupby("outcome")["run_value"].mean().reindex(OUTCOMES).fillna(0.0)
    cuenta = (d.groupby(["count", "outcome"])["run_value"].mean().unstack()
              .reindex(columns=OUTCOMES).fillna(media))
    peso = d["count"].value_counts(normalize=True).reindex(cuenta.index)
    neutral = cuenta.mul(peso, axis=0).sum()
    return {"media": {c: float(v) for c, v in media.items()},
            "cuenta": {k: {c: float(v) for c, v in fila.items()} for k, fila in cuenta.iterrows()},
            "peso_cuenta": {k: float(v) for k, v in peso.items()},
            "neutral": {c: float(v) for c, v in neutral.items()},
            "densidad": _batazos_por_densidad(d)}


def _tablas(valores):
    """(valor neutral por clase, tabla cuenta x clase, cuentas). La ULTIMA fila de
    la tabla es la neutral: una cuenta desconocida tiene codigo -1 y cae ahi."""
    neutral = np.array([valores["neutral"].get(c, 0.0) for c in OUTCOMES])
    cuentas = list(valores["cuenta"])
    tabla = np.array([[valores["cuenta"][k].get(c, neutral[i]) for i, c in enumerate(OUTCOMES)] for k in cuentas]
                     + [list(neutral)])
    return neutral, tabla, cuentas


def valores_en(valores, rho, clip=True):
    """Valor de cada resultado (a cuenta promedio) en un parque de densidad
    rho: los batazos se mueven sobre su recta; lo demas no depende del aire."""
    den = valores["densidad"]
    r = float(np.clip(rho, den["rho_min"], den["rho_max"])) if clip else float(rho)
    return {c: valores["neutral"].get(c, 0.0) + (den["pendiente"].get(c, 0.0) * (r - den["rho_liga"])
                                                  if c in IN_PLAY else 0.0) for c in OUTCOMES}


def xrv_from_proba(P, valores):
    """P: (n, 10) en el orden de OUTCOMES. valores: dict resultado -> carreras."""
    return P @ np.array([valores.get(c, 0.0) for c in OUTCOMES])


def xrv(P, valores, rho=None, counts=None, clip=True):
    """Carreras esperadas de cada pitcheo. P: (n, 10) en el orden de OUTCOMES.
      counts=None  valores a cuenta promedio (Stuff+)
      counts       valores de la cuenta de cada pitcheo ("b-s")
      rho=None     batazos valuados en la densidad media de la liga
      rho          numero o un valor por fila: batazos valuados en ese aire
      clip         no extrapolar la recta fuera de [rho_min, rho_max]"""
    neutral, tabla, cuentas = _tablas(valores)
    if counts is None:
        base = P @ neutral
    else:
        base = np.einsum("ij,ij->i", P, tabla[pd.Index(cuentas).get_indexer(counts)])
    if rho is None:
        return base
    den = valores["densidad"]
    r = np.asarray(rho, dtype=float)
    if clip:
        r = np.clip(r, den["rho_min"], den["rho_max"])
    b = np.array([den["pendiente"].get(c, 0.0) for c in IN_PLAY])
    return base + (P[:, IP] @ b) * (r - den["rho_liga"])


def rv_cuenta_media(df, valores):
    """Run value REALIZADO de cada pitcheo llevado a la mezcla de cuentas de la
    liga: se cambia el valor de su resultado en SU cuenta por el valor de ese
    resultado a cuenta promedio (el batazo conserva lo que se aparto de su clase).
    Es contra lo que se compara el xRV de Stuff+: un strike cantado vale mas con
    dos strikes, y los valores neutrales, por construccion, no promedian igual
    que el run value real."""
    neutral, tabla, cuentas = _tablas(valores)
    k, y = pd.Index(cuentas).get_indexer(df["count"]), _y(df)
    return df["run_value"].to_numpy(float) - tabla[k, y] + neutral[y]


# ── entrenamiento ───────────────────────────────────────────────────────────
def _prep(df, feats):
    """Matriz del modelo. Una variable que no venga queda vacia (proba avisa);
    pt lleva siempre las mismas categorias, en entrenamiento y al predecir."""
    X = df.reindex(columns=feats)
    if "pt" in feats:
        X["pt"] = pd.Categorical(X["pt"].where(X["pt"].isin(TIPOS)), categories=list(TIPOS))
    return X


def _modelo(iters):
    if BACKEND == "lightgbm":
        return lgb.LGBMClassifier(objective="multiclass", n_estimators=iters, learning_rate=0.06,
                                  num_leaves=48, min_child_samples=200, reg_lambda=1.0, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, n_jobs=N_THREADS,
                                  random_state=RANDOM_STATE, verbose=-1)
    return HistGradientBoostingClassifier(
        loss="log_loss", max_iter=iters, learning_rate=0.08, max_leaf_nodes=40, min_samples_leaf=200,
        l2_regularization=1.0, categorical_features="from_dtype", early_stopping=False,
        random_state=RANDOM_STATE)


def fit(df, feats, iters):
    """Ajusta un sub-modelo con un numero FIJO de arboles (el de elegir_iters).
    Solo entran las variables que existen y traen datos."""
    feats = [f for f in feats if f in df.columns and df[f].notna().any()]
    m = _modelo(iters).fit(_prep(df, feats), _y(df))
    m.feats_, m.iters_, m.backend_ = feats, int(iters), BACKEND
    return m


def _diez(m, P):
    """Probabilidades en las 10 columnas de OUTCOMES (una clase que no aparecio al entrenar queda en 0)."""
    full = np.zeros((len(P), len(OUTCOMES)), dtype=np.float32)
    full[:, m.classes_] = P
    return full


_AVISADAS = set()


def proba(m, df):
    """P(resultado) de cada pitcheo: (n, 10) float32 en el orden de OUTCOMES."""
    P = np.zeros((len(df), len(OUTCOMES)), dtype=np.float32)
    if not len(df):
        return P
    for f in m.feats_:
        if f not in _AVISADAS and (f not in df.columns or not df[f].notna().any()):
            _AVISADAS.add(f)
            print(f"AVISO: el modelo usa '{f}' y los datos a calificar no la traen (o viene vacia): "
                  "la prediccion se hace sin esa variable y no es comparable con las tablas")
    for i in range(0, len(df), TROZO):
        P[i:i + TROZO] = _diez(m, m.predict_proba(_prep(df.iloc[i:i + TROZO], m.feats_)))
    return P


def _normalizar(P):
    """Sin ceros (para el logaritmo) y sumando 1."""
    P = np.clip(P, 1e-6, 1.0)
    return P / P.sum(1, keepdims=True)


def elegir_iters(df, feats, n_cortes=1):
    """Numero de arboles de un sub-modelo: el que da el menor log-loss en
    pitchers que el modelo NO vio (85% / 15% de los PITCHERS; partir filas al
    azar deja a los mismos pitchers de los dos lados y siempre pide mas arboles).
    Con n_cortes > 1 el corte se repite con otros pitchers y se toma el minimo
    de la curva PROMEDIO (pesada por los pitcheos de validacion de cada corte):
    cuando la curva es plana, el minimo de un solo corte lo decide el azar.
    Devuelve (arboles, detalle); en el detalle, n_train es el promedio de los
    cortes, n_val la suma, pitchers_val los pitchers distintos y
    mejor_por_corte el minimo de cada curva (cuanto se mueve el entero)."""
    d = df.sample(MAX_FILAS_FOLD, random_state=RANDOM_STATE) if len(df) > MAX_FILAS_FOLD else df
    cortes = GroupShuffleSplit(n_splits=n_cortes, test_size=0.15, random_state=RANDOM_STATE)
    curvas, n_tr, n_va, pitchers = [], [], [], set()
    for tr, va in cortes.split(d, groups=d["pitcher_id"]):
        tr, va = d.iloc[tr], d.iloc[va]
        yva = _y(va)
        if BACKEND == "lightgbm":
            feats = [f for f in feats if f in tr.columns and tr[f].notna().any()]
            m = _modelo(ITERS_MAX).fit(_prep(tr, feats), _y(tr), eval_set=[(_prep(va, feats), yva)],
                                       eval_metric="multi_logloss")
            curvas.append(np.asarray(m.evals_result_["valid_0"]["multi_logloss"]))
        else:
            m = fit(tr, feats, ITERS_MAX)
            fila = np.arange(len(yva))
            curvas.append(np.array([-np.log(_normalizar(_diez(m, P))[fila, yva]).mean()
                                    for P in m.staged_predict_proba(_prep(va, m.feats_))]))
        n_tr.append(len(tr))
        n_va.append(len(va))
        pitchers.update(va["pitcher_id"].unique())
    curva = np.average(curvas, axis=0, weights=n_va)
    mejor = int(np.argmin(curva)) + 1
    iters = min(ITERS_MAX, max(PISO_ITERS, mejor))
    return iters, {"mejor": mejor, "logloss_val": float(curva[iters - 1]), "logloss_mejor": float(curva.min()),
                   "logloss_tope": float(curva[-1]), "n_train": int(np.mean(n_tr)), "n_val": int(np.sum(n_va)),
                   "pitchers_val": len(pitchers), "cortes": int(n_cortes),
                   "mejor_por_corte": [int(np.argmin(c)) + 1 for c in curvas]}


def iteraciones(df, nombres=tuple(SUBMODELOS), recalcular=False):
    """Arboles de cada sub-modelo, guardados en models/iters.json. Se eligen UNA
    vez (etapa validar) y con ese numero se ajustan los folds, los holdouts y
    el modelo final."""
    ruta = config.MODELS / "iters.json"
    base = {"backend": BACKEND, "iters_max": ITERS_MAX, "piso": PISO_ITERS, "filas": int(len(df))}
    info = json.loads(ruta.read_text()) if ruta.exists() and not recalcular else {}
    if any(info.get(k) != v for k, v in base.items()):         # otro backend, otro tope u otros datos
        info = dict(base)
    info.setdefault("detalle", {})
    for nm in nombres:
        if nm not in info:
            t = time.time()
            info[nm], det = elegir_iters(df, SUBMODELOS[nm], CORTES_ITERS.get(nm, 1))
            info["detalle"][nm] = det
            nota = (" | OJO: seguia bajando en el tope, se puede subir ITERS_MAX" if det["mejor"] == ITERS_MAX
                    else f" | OJO: se uso el piso de {PISO_ITERS}, el modelo queda algo sobreajustado"
                    if det["mejor"] < info[nm] else "")
            cortes = (f"; promedio de {det['cortes']} cortes con minimos en "
                      f"{', '.join(map(str, det['mejor_por_corte']))}" if det["cortes"] > 1 else "")
            print(f"  arboles de {nm:8s}: {info[nm]:>3d} (tope {ITERS_MAX}; el minimo estuvo en {det['mejor']}{cortes})"
                  f" | log-loss {det['logloss_val']:.4f} en {det['pitchers_val']} pitchers que no vio | "
                  f"{time.time() - t:.0f}s{nota}", flush=True)
    ruta.write_text(json.dumps(info, indent=1))
    return info


def _entrenamiento(df, mascara, semilla):
    """De las filas de entrenamiento de un corte: la muestra con la que se
    ajustan los modelos (a lo mas MAX_FILAS_FOLD), y los valores y frecuencias
    estimados con TODAS esas filas y solo con ellas (nada de las de prueba)."""
    idx = np.flatnonzero(mascara)
    base = df[COLS_VALOR].iloc[idx]
    valores, frec = outcome_values(base), frecuencias(base)
    if len(idx) > MAX_FILAS_FOLD:
        idx = np.sort(np.random.default_rng(RANDOM_STATE + semilla).choice(idx, MAX_FILAS_FOLD, replace=False))
    return df.iloc[idx], valores, frec


# ── modelos nulos y metricas ────────────────────────────────────────────────
def frecuencias(train, m=20.0):
    """(f(c) de la liga, f(c | cuenta)) en las filas de entrenamiento. Cada
    cuenta se suaviza con m pitcheos repartidos como la liga, para que ninguna
    celda quede en cero."""
    t = pd.crosstab(train["count"], train["outcome"]).reindex(columns=OUTCOMES, fill_value=0)
    f = t.sum() / t.to_numpy().sum()
    return f.to_numpy(float), (t + m * f).div(t.sum(axis=1) + m, axis=0)


def comparaciones(te, P, frec):
    """Los seis modelos de cada corte de validacion: nombre -> probabilidades.
      nulo_global   frecuencia de cada resultado en la liga
      nulo_cuenta   frecuencia por cuenta: el MODELO NULO que pide el reto
      stuff_cuenta  Stuff+ con la cuenta agregada por fuera, sin reentrenar:
                    P(c | x, cuenta) ~ P_stuff(c | x) * f(c | cuenta) / f(c).
                    Es la comparacion justa de un modelo que no ve la cuenta
                    contra el nulo por cuenta.
    P: {"stuff", "location", "pitching"} -> (n, 10) de las filas de te."""
    f, por_cuenta = frec
    pc = por_cuenta.reindex(te["count"]).to_numpy(float)
    pc = np.where(np.isnan(pc).any(1, keepdims=True), f, pc)        # cuenta que no estaba en el entrenamiento
    con_cuenta = P["stuff"] * pc / np.maximum(f, 1e-9)
    con_cuenta /= con_cuenta.sum(1, keepdims=True)
    return {"nulo_global": np.tile(f, (len(te), 1)), "nulo_cuenta": pc, "stuff": P["stuff"],
            "stuff_cuenta": con_cuenta, "location": P["location"], "pitching": P["pitching"]}


def _auc(y, p):
    """Area bajo la curva ROC; NaN si en y hay una sola clase."""
    y = np.asarray(y, dtype=int)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def _corr(a, b):
    """Correlacion; NaN si alguno de los dos es constante (no avisa ni truena)."""
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan


def _condicional(P, filas, clase, grupo):
    """P(clase | el resultado es alguno de `grupo`) en las filas pedidas."""
    return P[filas, clase] / P[filas][:, grupo].sum(1)


def _evaluar(test, P, valores, nombre, valuacion):
    """Metricas de un modelo y lo que hace falta por pitcheo para los
    intervalos: (fila, log-loss, Brier de whiff, xRV, objetivo del xRV)."""
    y = _y(test)
    P = _normalizar(P)
    ll = -np.log(P[np.arange(len(y)), y])
    if valuacion == "neutral":
        xr, obj = xrv(P, valores), rv_cuenta_media(test, valores)
    else:
        xr = xrv(P, valores, rho=test["air_density"].to_numpy(float), counts=test["count"])
        obj = test["run_value"].to_numpy(float)
    whiff = y == I["whiff"]
    brier = (P[:, I["whiff"]] - whiff) ** 2
    sw, toma, bip = np.isin(y, SWING), np.isin(y, TOMA), np.isin(y, IP)
    p_ws = np.clip(_condicional(P, sw, I["whiff"], SWING), 1e-6, 1 - 1e-6)
    ll_ws = float(-np.mean(np.where(whiff[sw], np.log(p_ws), np.log1p(-p_ws)))) if sw.any() else np.nan
    fila = {"modelo": nombre, "valuacion": valuacion, "n": int(len(y)),
            "logloss": float(ll.mean()), "brier_whiff": float(brier.mean()),
            "rmse_xrv": float(np.sqrt(np.mean((xr - obj) ** 2))), "corr_xrv": _corr(xr, obj),
            "auc_whiff_swing": _auc(whiff[sw], p_ws), "logloss_whiff_swing": ll_ws,
            "auc_cs_take": _auc(y[toma] == I["called_strike"], _condicional(P, toma, I["called_strike"], TOMA)),
            "auc_gb_bip": _auc(y[bip] == I["ip_gb"], _condicional(P, bip, I["ip_gb"], IP))}
    return fila, ll, brier, xr, obj


def metricas(test, P, valores, nombre, modo):
    """Metricas de un modelo en filas que no vio. modo "stuff": xRV con valores
    neutrales (cuenta promedio, densidad de liga) contra rv_cuenta_media;
    modo "cuenta": valores de la cuenta y la densidad de cada pitcheo contra el
    run value real. Las condicionales salen de las 10 clases:
    P(whiff | swing) = P_whiff / (P_whiff + P_foul + P_batazo), y igual para
    strike cantado sin swing y rodado entre los batazos."""
    return _evaluar(test, P, valores, nombre, "neutral" if modo == "stuff" else "cuenta")[0]


def _comparar(te, P, valores, frec):
    """Los seis modelos en un corte: (filas de metricas, modelo -> arreglo (4, n)
    con log-loss, Brier, xRV y objetivo de cada pitcheo)."""
    filas, por_fila = [], {}
    for nombre, Pm in comparaciones(te, P, frec).items():
        fila, *piezas = _evaluar(te, Pm, valores, nombre, VALUACION[nombre])
        filas.append(fila)
        por_fila[nombre] = np.array(piezas, dtype=np.float32)
    return filas, por_fila


def resumir(filas, por_fila, pitcher, iters):
    """Una fila por modelo con todos los pitcheos de prueba juntos y el IC 95%
    de cada comparacion contra los dos nulos. El IC remuestrea PITCHERS (no
    pitcheos: los de un mismo pitcher se parecen) y es pareado: cada replica
    usa los mismos pitchers para el modelo y para el nulo.
      mejora_vs_*_pct   100 * (1 - log-loss del modelo / log-loss del nulo)
      delta_brier_vs_*  Brier de whiff del modelo menos el del nulo (negativo = mejor)
      delta_rmse_vs_*   RMSE del xRV menos el del nulo; solo entre modelos con
                        la misma valuacion (si no, no son comparables: NaN)"""
    pid = pd.factorize(pitcher)[0]
    n_p = int(pid.max()) + 1
    rng = np.random.default_rng(RANDOM_STATE)
    # cada fila de W dice cuantas veces entra cada pitcher; la fila 0 es la muestra real
    W = np.vstack([np.ones(n_p), rng.multinomial(n_p, np.full(n_p, 1.0 / n_p), size=N_BOOT)])
    total = W @ np.bincount(pid, minlength=n_p)
    media = lambda v: (W @ np.bincount(pid, weights=v, minlength=n_p)) / total       # [0] real, [1:] replicas
    ll, br, mse = {}, {}, {}
    for nm, (l, b, x, o) in por_fila.items():
        ll[nm], br[nm], mse[nm] = media(l), media(b), media((x.astype(float) - o) ** 2)
    filas, out = pd.DataFrame(filas), []
    for nm in por_fila:
        f = filas[filas["modelo"] == nm]
        fila = {"modelo": nm, "valuacion": VALUACION[nm], "n": int(len(pid)), "pitchers": n_p,
                "logloss": ll[nm][0], "brier_whiff": br[nm][0], "rmse_xrv": np.sqrt(mse[nm][0]),
                # un modelo que predice una constante (nulo global) no tiene correlacion
                "corr_xrv": _corr(por_fila[nm][2], por_fila[nm][3]) if f["corr_xrv"].notna().any() else np.nan}
        for c in CONDICIONALES:                                   # promedio de los folds, pesado por sus filas
            ok = f[c].notna()
            fila[c] = float(np.average(f.loc[ok, c], weights=f.loc[ok, "n"])) if ok.any() else np.nan
        for ref in ("global", "cuenta"):
            nulo = "nulo_" + ref
            comparable = VALUACION[nm] == VALUACION[nulo]
            for medida, v in (("mejora", 100 * (1 - ll[nm] / ll[nulo])), ("delta_brier", br[nm] - br[nulo]),
                              ("delta_rmse", np.sqrt(mse[nm]) - np.sqrt(mse[nulo]) if comparable
                               else np.full(len(W), np.nan))):
                fila.update(zip(_cols_comparacion(medida, ref), (v[0], *np.percentile(v[1:], [2.5, 97.5]))))
        fila["backend"] = BACKEND
        fila["iters"] = iters.get(nm.split("_")[0], np.nan)      # stuff_cuenta usa el modelo stuff; los nulos, ninguno
        out.append(fila)
    return pd.DataFrame(out, columns=COLS_RESUMEN)


# ── validacion ──────────────────────────────────────────────────────────────
def crossfit(df, iters):
    """Cross-fit por pitcher: GroupKFold sobre TODOS los pitcheos. Cada fold
    entrena con los otros pitchers (a lo mas MAX_FILAS_FOLD filas) y predice
    todos los pitcheos de los suyos, que nunca vio. Guarda los modelos de cada
    fold (los usa score.py) y out/folds.csv (pitcher_id, fold).
    Devuelve (P, fold, cortes): P[modelo] = (n, 10) float32 fuera de muestra,
    alineada con df; cortes[k] = valores y frecuencias de las filas de
    entrenamiento del fold k, para valuar sin usar nada de los de prueba."""
    pitcher = df["pitcher_id"].to_numpy()
    fold = np.zeros(len(df), dtype=np.int8)
    for k, (_, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(pitcher, groups=pitcher)):
        fold[te] = k
    (pd.DataFrame({"pitcher_id": pitcher, "fold": fold}).drop_duplicates("pitcher_id")
     .sort_values(["fold", "pitcher_id"]).to_csv(config.OUT / "folds.csv", index=False))
    P = {nm: np.zeros((len(df), len(OUTCOMES)), dtype=np.float32) for nm in SUBMODELOS}
    cortes = []
    for k in range(N_FOLDS):
        t = time.time()
        te = np.flatnonzero(fold == k)
        tr, valores, frec = _entrenamiento(df, fold != k, k)
        prueba = df.iloc[te]
        for nm, feats in SUBMODELOS.items():
            m = fit(tr, feats, iters[nm])
            joblib.dump(m, config.MODELS / f"{nm}_fold{k}.joblib", compress=3)
            P[nm][te] = proba(m, prueba)
        cortes.append({"valores": valores, "frec": frec})
        print(f"  fold {k}: entrena con {len(tr):,} pitcheos, predice {len(te):,} de "
              f"{prueba['pitcher_id'].nunique():,} pitchers que no vio | {time.time() - t:.0f}s", flush=True)
    return P, fold, cortes


def pitcher_holdout(df, P, fold, cortes, iters):
    """Metricas en pitchers que el modelo no vio. Devuelve (tabla por fold,
    resumen con intervalos, xRV fuera de muestra de stuff y pitching con su
    objetivo, para la calibracion)."""
    filas, todo = [], {}
    for k, c in enumerate(cortes):
        idx = np.flatnonzero(fold == k)
        f_k, por_fila = _comparar(df.iloc[idx], {nm: M[idx] for nm, M in P.items()}, c["valores"], c["frec"])
        filas += [{**f, "fold": k} for f in f_k]
        for nm, a in por_fila.items():
            todo.setdefault(nm, np.zeros((4, len(df)), dtype=np.float32))[:, idx] = a
    return (pd.DataFrame(filas, columns=COLS_HOLDOUT), resumir(filas, todo, df["pitcher_id"], iters),
            {nm: (todo[nm][2], todo[nm][3]) for nm in ("stuff", "pitching")})


def _curva(pred, obs, n_bins=10, pid=None, W=None):
    """Confiabilidad: grupos de igual tamano ordenados por la prediccion.
    Devuelve (n, prediccion media, observado medio) por grupo y el resumen:
      ece        promedio de |observado - predicho|
      ece_ruido  el ECE que saldria por puro azar con un modelo bien calibrado:
                 la media observada de un grupo de n pitcheos se aparta de la
                 predicha ~ sd / raiz(n) aunque no haya sesgo. Un ECE parecido
                 a este piso no indica mala calibracion; con menos pitcheos
                 (una banda de altitud) el piso sube y el ECE con el
      pendiente  recta de observado contra predicho entre los grupos (1 = bien
                 calibrado; menos de 1 = el modelo exagera las diferencias)
      pendiente_ci_lo, _hi   IC 95% de la pendiente remuestreando PITCHERS con
                 los grupos fijos. pid = pitcher (entero) de cada fila; W =
                 cuantas veces entra cada pitcher en cada replica. Sin ellos, NaN"""
    pred, obs = np.asarray(pred, dtype=float), np.asarray(obs, dtype=float)
    orden = np.argsort(pred, kind="stable")
    b = np.empty(len(pred), dtype=int)
    b[orden] = np.arange(len(pred)) * n_bins // len(pred)
    n = np.bincount(b, minlength=n_bins)
    p, o = np.bincount(b, pred, n_bins) / n, np.bincount(b, obs, n_bins) / n
    var = np.maximum(np.bincount(b, (obs - pred) ** 2, n_bins) / n - (o - p) ** 2, 0.0)   # de obs - pred, por grupo
    lo = hi = np.nan
    if pid is not None:
        # sumas por (pitcher, grupo) una sola vez; cada replica es una suma pesada de pitchers
        N, SP, SO = (W @ np.bincount(pid * n_bins + b, w, W.shape[1] * n_bins).reshape(-1, n_bins)
                     for w in (None, pred, obs))
        with np.errstate(divide="ignore", invalid="ignore"):     # un grupo sin pitcheos en una replica queda en NaN
            dp, do = (S / N - (S.sum(1) / N.sum(1))[:, None] for S in (SP, SO))
            sxx = np.nansum(N * dp ** 2, axis=1)                 # la misma recta pesada de _recta, replica por replica
            rep = np.where(sxx > 1e-12, np.nansum(N * dp * do, axis=1) / sxx, np.nan)
        if np.isfinite(rep).mean() > 0.9:
            lo, hi = (float(q) for q in np.percentile(rep[np.isfinite(rep)], [2.5, 97.5]))
    return (n, p, o), {"n": int(len(pred)), "tasa_base": float(obs.mean()),
                       "ece": float(np.average(np.abs(o - p), weights=n)),
                       "ece_ruido": float(np.average(np.sqrt(2 / np.pi * var / n), weights=n)),
                       "pendiente": _recta(p, o, n), "pendiente_ci_lo": lo, "pendiente_ci_hi": hi,
                       "brier": float(np.mean((pred - obs) ** 2))}


def calibracion(df, P, xr):
    """Confiabilidad de las probabilidades y del xRV FUERA DE MUESTRA (cross-fit
    por pitcher), en total y por banda de altitud. Objetivos del reto:
      whiff, called_strike, swing, gb    por pitcheo, de Stuff+ y de Pitching+
      contacto_duro   linea o elevado duro (P(ip_ld) + P(ip_fb_hard)). OJO: es
                      un TIPO de batazo, no la velocidad de salida: una linea
                      cuenta aunque salga suave y un rodado de 95+ mph no cuenta
                      (score.py lo llama p_linea_o_elevado_duro)
      chase           P(swing) de Pitching+ en los pitcheos fuera de zona
      xrv             xRV contra el run value realizado (Stuff+: a cuenta promedio)
    Devuelve (tabla por grupo de prediccion, resumen). El resumen trae el IC 95%
    de la pendiente (remuestreo de pitchers) y el piso de ruido del ECE: ver _curva."""
    y = _y(df)
    pid = pd.factorize(df["pitcher_id"])[0]
    n_p = int(pid.max()) + 1
    W = np.random.default_rng(RANDOM_STATE).multinomial(n_p, np.full(n_p, 1.0 / n_p), size=N_BOOT)
    todos = np.ones(len(df), dtype=bool)
    swing = lambda M: M[:, SWING].sum(1)
    objetivos = []                              # (objetivo, modelo, prediccion, observado, filas que cuentan)
    for nm in ("stuff", "pitching"):
        M = P[nm]
        objetivos += [("whiff", nm, M[:, I["whiff"]], y == I["whiff"], todos),
                      ("called_strike", nm, M[:, I["called_strike"]], y == I["called_strike"], todos),
                      ("swing", nm, swing(M), np.isin(y, SWING), todos),
                      ("gb", nm, M[:, I["ip_gb"]], y == I["ip_gb"], todos),
                      ("contacto_duro", nm, M[:, I["ip_ld"]] + M[:, I["ip_fb_hard"]],
                       np.isin(y, [I["ip_ld"], I["ip_fb_hard"]]), todos),
                      ("xrv", nm, xr[nm][0], xr[nm][1], todos)]
    objetivos.append(("chase", "pitching", swing(P["pitching"]), df["y_chase"].to_numpy(),
                      df["in_zone"].to_numpy() == 0))
    banda = df["banda"].to_numpy()
    curvas, resumen = [], []
    for objetivo, nm, pred, obs, filas in objetivos:
        for grupo in ("todos", "alta", "media", "mar"):
            m = filas if grupo == "todos" else filas & (banda == grupo)
            if m.sum() < 200:
                continue
            (n, p, o), res = _curva(pred[m], obs[m], pid=pid[m], W=W)
            llave = {"objetivo": objetivo, "modelo": nm, "grupo": grupo}
            curvas += [{**llave, "bin": i + 1, "n": int(n[i]), "pred_media": p[i], "obs_media": o[i]}
                       for i in range(len(n))]
            resumen.append({**llave, **res})
    return pd.DataFrame(curvas, columns=COLS_CAL), pd.DataFrame(resumen, columns=COLS_CAL_RES)


def season_holdout(df, iters):
    """Temporada fuera de muestra: se entrena con las temporadas anteriores a
    la ultima y se prueba en la ultima. Mismas columnas que el resumen por
    pitcher (los pitchers SI se repiten entre temporadas: mide el paso del
    tiempo, no pitchers nuevos)."""
    ultima = int(df["season"].max())
    antes = (df["season"] < ultima).to_numpy()
    cols = COLS_RESUMEN + ["test_season", "n_train"]
    if antes.sum() < MIN_CORTE or (~antes).sum() < MIN_CORTE:
        print("  temporada fuera de muestra omitida: el dataset trae una sola temporada (o casi)")
        return pd.DataFrame(columns=cols)
    tr, valores, frec = _entrenamiento(df, antes, 50)
    te = df[~antes]
    P = {nm: proba(fit(tr, feats, iters[nm]), te) for nm, feats in SUBMODELOS.items()}
    filas, por_fila = _comparar(te, P, valores, frec)
    return resumir(filas, por_fila, te["pitcher_id"], iters).assign(test_season=ultima, n_train=len(tr))[cols]


def _sesgo(dif, grupo):
    """Media de dif por 100 pitcheos e IC 95% remuestreando GRUPOS completos
    (juegos): los pitcheos de un mismo juego no son independientes."""
    g = pd.factorize(grupo)[0]
    G = int(g.max()) + 1
    W = np.random.default_rng(RANDOM_STATE).multinomial(G, np.full(G, 1.0 / G), size=N_BOOT)
    rep = (W @ np.bincount(g, weights=dif, minlength=G)) / (W @ np.bincount(g, minlength=G))
    return (100 * float(dif.mean()), *(100 * float(q) for q in np.percentile(rep, [2.5, 97.5])), G)


def _valores_remuestreados(base, grupo):
    """outcome_values recalculado en N_BOOT_VALORES remuestreos de GRUPOS
    completos (juegos) de las filas de entrenamiento `base`: el ruido de
    muestreo que traen los valores de los resultados y la recta de los batazos."""
    g = pd.factorize(grupo)[0]
    G = int(g.max()) + 1
    orden = np.argsort(g, kind="stable")
    corte = np.searchsorted(g[orden], np.arange(G + 1))                # filas del grupo k: orden[corte[k]:corte[k + 1]]
    rng = np.random.default_rng(RANDOM_STATE)
    return [outcome_values(base.iloc[np.concatenate([orden[corte[k]:corte[k + 1]] for k in rng.integers(0, G, G)])])
            for _ in range(N_BOOT_VALORES)]


def park_holdout(df, iters):
    """¿Que tan bien se predice un parque de altitud que el modelo NUNCA vio?
      sin_casa        se entrena sin los pitcheos de casa y se prueba en ellos
      sin_banda_alta  se entrena sin NINGUN parque de la banda alta y se prueba
                      en ellos (con categorias de altitud es el mismo corte que
                      el anterior y se escribe una sola vez)
    Un ajuste de Stuff y uno de Pitching por escenario. Los valores de los
    resultados, incluida la recta de los batazos en la densidad, salen SOLO de
    las filas de entrenamiento. Se reporta el sesgo (xRV predicho menos run
    value realizado, por 100 pitcheos) con tres valuaciones de los batazos:
      valor_neutral       sin informacion de altitud (densidad media del entrenamiento)
      valor_nivel_mar     como si el parque estuviera al nivel del mar (RHO_MAR)
      valor_por_densidad  la nuestra: la recta EXTRAPOLADA a la densidad del parque
    Stuff: pitcheo tal como se midio, valores a cuenta promedio, contra el run
    value a cuenta promedio. Pitching: valores por cuenta contra el run value real.
    OJO: los pitchers SI se repiten entre entrenamiento y prueba (lanzan en casa
    y de visita); lo que el modelo no vio es el parque.
    El IC junta dos ruidos independientes, sumados en cuadratura: el de los
    juegos del parque de prueba (remuestreo de juegos) y el de los valores y la
    recta, que salen de una muestra de juegos de entrenamiento (se remuestrean,
    se recalculan los valores y se vuelve a valuar con las MISMAS
    probabilidades; su desviacion estandar va en sesgo_ee_valores). No incluye
    el ruido de reajustar el clasificador."""
    casa, alta = (df["es_casa"] == 1).to_numpy(), (df["banda"] == "alta").to_numpy()
    escenarios = {"sin_casa": casa}
    if (alta != casa).any():
        escenarios["sin_banda_alta"] = alta
    else:
        print("  estadio fuera de muestra: 'sin_banda_alta' es el mismo corte que 'sin_casa' "
              "(la banda alta es solo la categoria de casa): se escribe una vez")
    por_juego = "game_id" in df and df["game_id"].notna().all()
    filas = []
    for i, (escenario, prueba) in enumerate(escenarios.items()):
        if prueba.sum() < MIN_CORTE or (~prueba).sum() < MIN_CORTE:
            print(f"  estadio fuera de muestra '{escenario}' omitido: no hay datos suficientes")
            continue
        tr, valores, _ = _entrenamiento(df, ~prueba, 100 + i)
        te = df[prueba]
        rho = te["air_density"].to_numpy(float)
        col_grupo = "game_id" if por_juego else "pitcher_id"
        grupo = te[col_grupo]
        den = valores["densidad"]
        entrena = np.flatnonzero(~prueba)                               # todas, como los valores de _entrenamiento
        replicas = _valores_remuestreados(df[COLS_VALOR].iloc[entrena], df[col_grupo].iloc[entrena])
        valuaciones = (("valor_neutral", None), ("valor_nivel_mar", RHO_MAR), ("valor_por_densidad", rho))
        for sub in ("stuff", "pitching"):
            P = _normalizar(proba(fit(tr, SUBMODELOS[sub], iters[sub]), te))
            logloss = float(-np.log(P[np.arange(len(te)), _y(te)]).mean())
            cuentas = None if sub == "stuff" else te["count"]
            rv = te["run_value"].to_numpy(float)
            # el objetivo de Stuff (run value a cuenta promedio) tambien cambia con los valores
            objetivo = (lambda v: rv_cuenta_media(te, v)) if sub == "stuff" else (lambda v: rv)
            real = objetivo(valores)
            medias = []                                                 # sesgo medio de cada replica x valuacion
            for v in replicas:
                real_v = objetivo(v)
                medias.append([np.mean(xrv(P, v, rho=aire, counts=cuentas, clip=False) - real_v)
                               for _, aire in valuaciones])
            ee_valores = 100 * np.std(medias, axis=0, ddof=1)
            for (valuacion, aire), ee in zip(valuaciones, ee_valores):
                pred = xrv(P, valores, rho=aire, counts=cuentas, clip=False)
                sesgo, lo, hi, G = _sesgo(pred - real, grupo)
                lo, hi = sesgo - np.hypot(sesgo - lo, 1.96 * ee), sesgo + np.hypot(hi - sesgo, 1.96 * ee)
                filas.append({"escenario": escenario, "submodelo": sub, "valuacion": valuacion, "n": int(len(te)),
                              "n_train": int(len(tr)), "logloss": logloss,
                              "xrv_medio_pred_x100": 100 * float(pred.mean()),
                              "rv_medio_real_x100": 100 * float(real.mean()), "sesgo_por_100": sesgo,
                              "sesgo_ci_lo": float(lo), "sesgo_ci_hi": float(hi), "sesgo_ee_valores": float(ee),
                              "n_juegos": G,
                              "agrupado_por": "juego" if por_juego else "pitcher", "rho_test": float(rho.mean()),
                              "rho_train_min": den["rho_obs"][0], "rho_train_max": den["rho_obs"][1],
                              "n_puntos": den["n_puntos"]})
    return pd.DataFrame(filas, columns=COLS_PARQUE)


def ablacion_h4(df, max_swings=100_000):
    """H4, "el spin solo no basta": ¿que tan bien se predice el swing fallado
    (dado que hubo swing) con cada conjunto de variables de H4? GroupKFold(5)
    por pitcher. `delta_*` compara cada conjunto con su `referencia` en los
    MISMOS folds; el IC usa la correccion de Nadeau y Bengio (los folds
    comparten datos de entrenamiento: sin ella el intervalo sale angosto).
    `delta_*_vs_spin` compara igual, pero contra H4_BASE (tipo + spin), cada
    conjunto que va despues: la fila "+diferenciales" es el contraste de H4
    (todo lo que Stuff+ agrega al spin) y no depende del orden de los pasos."""
    y = _y(df)
    idx = np.flatnonzero(np.isin(y, SWING))
    if len(idx) > max_swings:
        idx = np.sort(np.random.default_rng(RANDOM_STATE).choice(idx, max_swings, replace=False))
    d = df.iloc[idx]
    if len(d) < MIN_CORTE or d["pitcher_id"].nunique() < 10:
        print("  ablacion H4 omitida: no hay swings suficientes")
        return pd.DataFrame(columns=COLS_H4)
    whiff, pitcher = (y[idx] == I["whiff"]).astype(int), d["pitcher_id"].to_numpy()
    K = 5
    cortes = list(GroupKFold(n_splits=K).split(pitcher, groups=pitcher))
    medida, filas = {}, []
    for conjunto, (variables, referencia) in H4.items():
        variables = [v for v in variables if v in d.columns and d[v].notna().any()]
        X = _prep(d, variables)
        auc, ll = [], []
        for tr, te in cortes:
            m = HistGradientBoostingClassifier(
                max_iter=50 if RAPIDO else 150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                l2_regularization=1.0, categorical_features="from_dtype", early_stopping=False,
                random_state=RANDOM_STATE).fit(X.iloc[tr], whiff[tr])
            p = np.clip(m.predict_proba(X.iloc[te])[:, 1], 1e-6, 1 - 1e-6)
            auc.append(roc_auc_score(whiff[te], p))
            ll.append(-np.mean(np.where(whiff[te] == 1, np.log(p), np.log1p(-p))))
        medida[conjunto] = np.array([auc, ll])
        fila = {"conjunto": conjunto, "n_vars": len(variables), "auc": np.mean(auc), "logloss": np.mean(ll),
                "auc_sd": np.std(auc, ddof=1), "logloss_sd": np.std(ll, ddof=1), "n": int(len(d)),
                "referencia": referencia}
        despues_del_spin = H4_BASE in medida and conjunto != H4_BASE
        for contra, sufijo in ((referencia, ""), (H4_BASE if despues_del_spin else None, "_vs_spin")):
            if not contra:
                continue
            dif = medida[conjunto] - medida[contra]                           # (2, folds), pareado
            ee = dif.std(axis=1, ddof=1) * np.sqrt(1 / K + 1 / (K - 1))       # 1/k + n_prueba/n_entrena
            medio, margen = dif.mean(axis=1), stats.t.ppf(0.975, K - 1) * ee
            for j, nombre in enumerate(("delta_auc" + sufijo, "delta_logloss" + sufijo)):
                fila.update(zip((nombre, nombre + "_lo", nombre + "_hi"),
                                (medio[j], medio[j] - margen[j], medio[j] + margen[j])))
        filas.append(fila)
    return pd.DataFrame(filas, columns=COLS_H4)


def importancia_stuff(df, fold, n_max=30_000, repeticiones=2):
    """Importancia por permutacion del modelo Stuff del fold 0 en pitcheos que
    no vio: cuanto sube el log-loss al revolver una variable. OJO: variables
    muy relacionadas (ivb, mov_total, los diferenciales) se reparten el efecto."""
    m = joblib.load(config.MODELS / "stuff_fold0.joblib")
    idx = np.flatnonzero(fold == 0)
    rng = np.random.default_rng(RANDOM_STATE)
    if len(idx) > n_max:
        idx = np.sort(rng.choice(idx, n_max, replace=False))
    te = df[m.feats_ + ["outcome"]].iloc[idx].copy()
    y, fila = _y(te), np.arange(len(idx))
    logloss = lambda: float(-np.log(_normalizar(proba(m, te))[fila, y]).mean())
    base, filas = logloss(), []
    for f in m.feats_:
        original = te[f].to_numpy()
        subidas = []
        for _ in range(repeticiones):
            te[f] = rng.permutation(original)
            subidas.append(logloss() - base)
        te[f] = original
        filas.append({"variable": f, "delta_logloss": np.mean(subidas),
                      "delta_logloss_sd": np.std(subidas, ddof=1)})
    return pd.DataFrame(filas, columns=COLS_IMP).sort_values("delta_logloss", ascending=False)


# ── etapas ──────────────────────────────────────────────────────────────────
def etapa_valores(df):
    v = outcome_values(df)
    (config.MODELS / "outcome_values.json").write_text(json.dumps(v, indent=1))
    den, f = v["densidad"], df["outcome"].value_counts(normalize=True)
    print("valor de cada resultado a cuenta promedio (carreras, + = bateador):",
          {c: round(x, 3) for c, x in v["neutral"].items()})
    print(f"run value medio x100: real {100 * df['run_value'].mean():+.2f} | con valores a cuenta promedio "
          f"{100 * sum(f.get(c, 0.0) * x for c, x in v['neutral'].items()):+.2f} (por eso el xRV de Stuff+ se "
          "valida contra el run value a cuenta promedio)")
    print(f"batazos segun la densidad del aire: recta sobre {den['n_puntos']} categorias de altitud"
          + ("" if den["ic_fiable"] else " (menos de 6: el intervalo es solo orientativo)"))
    mar, casa = valores_en(v, RHO_MAR), valores_en(v, RHO_CDMX)
    for c in IN_PLAY:
        lo, hi = den["pendiente_ic"][c]
        print(f"  {c:11s} nivel del mar {mar[c]:+.3f} | Harp Helu {casa[c]:+.3f} | pendiente "
              f"{den['pendiente'][c]:+.2f} [{lo:+.2f}, {hi:+.2f}] carreras por unidad de densidad")


def _guardar(tabla, nombre, t0, **formato):
    tabla.to_csv(config.OUT / nombre, index=False)
    print(f"\n{nombre} ({time.time() - t0:.0f}s)")
    print(tabla.to_string(index=False, **formato) if len(tabla) else "  (sin filas: no estimable con estos datos)",
          flush=True)


def etapa_validar(df):
    t = time.time()
    print("numero de arboles (validacion agrupada por pitcher):", flush=True)
    iters = iteraciones(df, recalcular=True)
    print(f"cross-fit por pitcher: {N_FOLDS} folds", flush=True)
    P, fold, cortes = crossfit(df, iters)
    ph, resumen, xr = pitcher_holdout(df, P, fold, cortes, iters)
    ph.to_csv(config.OUT / "val_pitcher_holdout.csv", index=False)
    ver = ["modelo", "logloss", "mejora_vs_global_pct", "mejora_vs_global_ci_lo", "mejora_vs_global_ci_hi",
           "mejora_vs_cuenta_pct", "mejora_vs_cuenta_ci_lo", "mejora_vs_cuenta_ci_hi", "rmse_xrv", "corr_xrv"]
    _guardar(resumen, "val_resumen.csv", t, columns=ver + CONDICIONALES, float_format="{:.4f}".format)
    t = time.time()
    curvas, cal = calibracion(df, P, xr)
    curvas.to_csv(config.OUT / "val_calibracion.csv", index=False)
    _guardar(cal, "val_calibracion_resumen.csv", t, float_format="{:.4f}".format)
    t = time.time()
    _guardar(season_holdout(df, iters), "val_season_holdout.csv", t, columns=ver + ["test_season"],
             float_format="{:.4f}".format)
    t = time.time()
    _guardar(park_holdout(df, iters), "val_park_holdout.csv", t, columns=COLS_PARQUE[:13],
             float_format="{:.3f}".format)
    t = time.time()
    _guardar(ablacion_h4(df), "val_h4_ablation.csv", t, float_format="{:.4f}".format)
    t = time.time()
    _guardar(importancia_stuff(df, fold), "val_importancia_stuff.csv", t, float_format="{:.5f}".format)


def etapa_final(df):
    t = time.time()
    iters = iteraciones(df, ["stuff"])["stuff"]
    joblib.dump(fit(df, SUBMODELOS["stuff"], iters), config.MODELS / "stuff.joblib", compress=3)
    print(f"modelo final de Stuff (solo para el simulador): backend {BACKEND} | {iters} arboles | "
          f"{len(df):,} pitcheos | {time.time() - t:.0f}s", flush=True)


def main():
    etapa = sys.argv[1] if len(sys.argv) > 1 else "todo"
    etapas = {"valores": etapa_valores, "validar": etapa_validar, "final": etapa_final}
    if etapa != "todo" and etapa not in etapas:
        raise SystemExit("Uso: python model.py [valores | validar | final | todo]")
    t0 = time.time()
    config.MODELS.mkdir(exist_ok=True)
    config.OUT.mkdir(exist_ok=True)
    df = cargar()
    print(f"backend: {BACKEND} | {len(df):,} pitcheos | {df['pitcher_id'].nunique():,} pitchers | "
          f"etapa: {etapa}" + (" | MODO RAPIDO" if RAPIDO else ""), flush=True)
    for nombre, correr in etapas.items():
        if etapa in ("todo", nombre):
            correr(df)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
