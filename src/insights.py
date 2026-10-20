"""
INSIGHTS — de los scores a decisiones para Diablos Rojos. Todo sale de lo que
escribio score.py: aqui no se ajusta ni se evalua ningun modelo.

  FORMAS   dentro de cada tipo de pitcheo, que version rinde mas en el Harp Helu
  PERFIL   tipo por tipo, como son los arsenales con mejor Stuff+ en CDMX
  H2       el cambio y el split (CH, FS) ganan valor relativo en la altitud?
  H3       el pitcher de rodados sufre menos la altitud?
  LISTAS   los mejores en CDMX, los mas afectados y los mas consistentes

INCERTIDUMBRE. Ninguna diferencia sale sin intervalo. Para una cifra <x>:
  <x>_lo, <x>_hi              IC 95% remuestreando PITCHERS completos (ruido de
                              muestreo; los mismos remuestreos que score.py)
  <x>_lo_folds, <x>_hi_folds  IC 95% ENTRE MODELOS: la misma cifra con los
                              pitchers de cada modelo de fold por separado y una
                              t de Student sobre esos valores. Otro modelo,
                              entrenado con otros pitchers, da otra cifra, y eso
                              el remuestreo de pitchers no lo ve
  <x>_sig                     True solo si los DOS intervalos dejan fuera el
                              valor de referencia (0 en una diferencia, 100 en
                              un nivel) y hay MIN_PITCHERS pitchers. Lo que solo
                              usa datos observados (la prueba de H3) no depende
                              del modelo: lleva nada mas el primer intervalo
Ningun intervalo incluye la incertidumbre de las elasticidades (physics.py).
Los veredictos de H2 y H3 salen de esos intervalos con una regla escrita
(`regla`) y pueden ser "apoyada", "contraria", "no concluyente" o "sin datos".

ESCALAS. Formas y H2: puntos de Stuff+ POR PITCHEO (1 punto = 1/10 de la
desviacion entre pitcheos). Listas: puntos ENTRE PITCHERS. No se comparan.

ARCHIVOS (en out/; siempre se escriben, con filas vacias o NaN si no alcanza):

insights_formas.csv — por tipo y forma, cuatro filas: tercil bajo, medio, alto
  y "alto-bajo". Los terciles se cortan sobre la magnitud con sentido de
  pitcheo, de modo que "alto" SIEMPRE es mas de lo que dice el nombre, y el
  quiebre es el llevado a la densidad media de la liga (no el medido en cada
  parque, que mezclaria "mucho ride" con "lanzado al nivel del mar").
    pt, forma        ride, hundimiento, caida, corrida, barrido, corte,
                     velocidad, dif_velo (recta menos pitcheo), spin
    etiqueta, unidad texto para el documento; pulg, mph o rpm
    columna, signo   de donde sale: magnitud = signo x columna (asi el
                     documento puede imprimir la columna misma, con su signo)
    tercil           bajo | medio | alto | alto-bajo
    valor_medio      magnitud media del tercil (en alto-bajo, la diferencia)
    pitcheos, pitchers   (en alto-bajo, los de los dos terciles juntos)
    stuff_cdmx, stuff_mar   NIVEL medio de Stuff+ en el Harp Helu y al nivel
                     del mar (100 = el pitcheo promedio lanzado en ESE aire)
    delta_altitud    stuff_cdmx - stuff_mar: cambio RELATIVO frente a la liga
    cada una con sus intervalos: stuff_cdmx_lo/_hi/_lo_folds/_hi_folds/_sig,
    stuff_mar_... y delta_lo, delta_hi, delta_lo_folds, delta_hi_folds, delta_sig
  En la fila alto-bajo las tres cifras son DIFERENCIAS (tercil alto menos
  bajo) y _sig las compara contra 0; en las otras, los niveles contra 100.

insights_perfil.csv — por tipo de pitcheo y rasgo, entre los arsenales
  (pitcher-temporada-tipo) con MIN_PERFIL+ pitcheos: el 25% con mayor
  stuff_cdmx_plus_enc ("top") contra el resto y contra todos ("liga").
    pt, rasgo        velo, dif_velo, spin, ivb_cdmx, hb_cdmx, ext, rel_h,
                     rel_s y rho (control, ver abajo). dif_velo = velocidad
                     de la recta primaria del pitcher menos la del pitcheo;
                     no cuenta los arsenales que SON esa recta (su diferencial
                     es 0 por construccion) y no se escribe para FF ni SI
    etiqueta, unidad mph, rpm, pulg, pies. IVB y HB son los del pitcheo
                     llevado al aire del Harp Helu: lo que mediria el radar
                     ahi. HB y lado de salida: + = lado del brazo (en un
                     zurdo, HorzBreak y RelSide del dataset traen el otro signo)
    control          True en la fila rho (densidad media del aire donde lanzo
                     el arsenal): no es un rasgo; si top y resto difieren ahi,
                     el perfil todavia depende de donde lanzo cada quien
    arsenales, n_top, n_resto, pitchers_top, corte_top (el Stuff+ CDMX
                     encogido que separa al top; puntos ENTRE ARSENALES)
    top_mediana, top_p25, top_p75, resto_mediana, resto_p25, resto_p75,
    liga_mediana, liga_p25, liga_p75
    dif              top_mediana - resto_mediana, con dif_lo, dif_hi,
                     dif_lo_folds, dif_hi_folds, dif_sig (en cada replica se
                     vuelve a elegir el 25% mejor)
    top_mar_mediana  mediana del rasgo en el 25% mejor AL NIVEL DEL MAR
                     (stuff_mar_plus) de los mismos arsenales
    dif_cdmx_mar     top_mediana - top_mar_mediana, con dif_cdmx_mar_lo, _hi,
                     _lo_folds, _hi_folds, _sig: en que se distingue el mejor
                     en el Harp Helu del mejor en cualquier parque
  Con menos de MIN_ARSENALES arsenales solo se llenan las columnas liga_*.

insights_mix.csv — mezcla de pitcheos de las pitcher-temporadas calificadas
  (scores_pitcher.csv): el 25% mejor en stuff_cdmx_plus_enc contra el resto.
  El uso es sobre TODOS los pitcheos y vale 0 si el pitcher no lanza ese tipo,
  asi que cada columna de uso suma 1.
    pt, uso_top, uso_resto, uso_liga (fracciones), n_top, n_resto,
    pitchers_top, dif (uso_top - uso_resto) con dif_lo, dif_hi,
    dif_lo_folds, dif_hi_folds, dif_sig

insights_top_cdmx.csv, insights_afectados.csv, insights_consistentes.csv —
  pitcher-temporadas de scores_pitcher.csv, 10 filas como maximo, mismas
  columnas (COLS_LISTA): pitcher_id, pitcher_name, season, pitcheos, lefty,
  pct_alta, velo_recta, stuff_plus_enc, stuff_lo, stuff_hi,
  stuff_cdmx_plus_enc, stuff_cdmx_lo, stuff_cdmx_hi, stuff_mar_plus,
  delta_altitud, delta_lo, delta_hi, delta_abs_max, costo_cdmx_rv100,
  location_plus_enc, pitching_plus_enc (valua los batazos EN SU PARQUE: sigue
  al parque del pitcher), pitching_neutral_plus_enc, pitching_neutral_lo,
  pitching_neutral_hi (el Pitching+ comparable entre parques), swstr_pct,
  whiff_swing_pct, gb_pct, rv100, fold (el modelo de la validacion cruzada que
  califico a ese pitcher: entre folds distintos hay un desfase de nivel que
  score.py mide en val_stability.json)
    top_cdmx       mayor stuff_cdmx_plus_enc entre quienes tienen MIN_TOP+ pitcheos
    afectados      delta_altitud mas negativo entre aquellos cuyo intervalo
                   completo queda bajo 0 (delta_hi < 0)
    consistentes   menor delta_abs_max = max(|delta_lo|, |delta_hi|): el cambio
                   mas grande que su intervalo todavia permite
  Los intervalos de estas tablas (de score.agregar) son solo ruido de muestreo
  (juegos y pitcheos): no incluyen el error del modelo.

insights.json
  H2  hipotesis (la frase que el veredicto prueba: la del reto);
      por_tipo {pt: fila de altitude_by_pitch_type.csv};
      contraste_liga = delta_altitud medio de CH+FS menos el de TODOS los
        pitcheos (delta_altitud promedia 0 en la liga: es lo que CH+FS ganan
        o pierden frente a los demas). Es el que decide (llave `decide`):
        a, b, pitcheos_a, pitcheos_b, pitchers_a, pitchers_b, delta (+ _lo,
        _hi, _lo_folds, _hi_folds, _sig), delta_mov (+ lo mismo): la parte
        que viene solo del cambio de movimiento, y veredicto
      contraste, contraste_rectas = lo mismo para CH+FS menos SL+CU (los
        quiebres por spin) y menos FF+SI, cada uno con SU veredicto; no
        cambian el de H2. Las familias son las de physics.FAMILIAS
      decide, veredicto (el de contraste_liga), regla
  H3  hipotesis;
      pitcher: la prueba dentro del mismo pitcher, con resultados OBSERVADOS.
        Pitcher-temporadas con MIN_BANDA_H3+ pitcheos en banda alta y en banda
        mar; regresion de (carreras por 100 arriba menos abajo) contra el % de
        rodados de la temporada, pesada por pitcheos. pendiente = carreras por
        100 por cada 10 puntos de rodados (negativa = el de rodados sufre
        menos), con pendiente_lo, _hi, _sig; dif_media (+ _lo, _hi, _sig) = lo
        que sube el run value arriba, en promedio; gb_medio (fraccion);
        pitcher_temporadas, pitchers, min_pitcheos_por_banda, unidad,
        veredicto. Con menos de MIN_PITCHERS pitchers: NaN y "sin datos".
        Dos cifras para leer un "no concluyente": semiancho (la mitad del
        intervalo de la pendiente: la prueba no distingue pendientes mas
        chicas que eso) y fiabilidad_gb (parte de la varianza del % de
        rodados que no es ruido binomial; el ruido del regresor acerca la
        pendiente a 0) con pendiente_corregida = pendiente / fiabilidad_gb,
        solo de referencia (no entra al veredicto)
      sensibilidad {"100": ..., "50": ...}: la misma prueba con umbrales mas
        bajos de pitcheos por banda (no decide nada)
      batazos: cuanto mas cuesta un elevado (ip_fb_soft + ip_fb_hard) que un
        rodado en CDMX que al nivel del mar, en carreras por batazo. En las
        dos lecturas el elevado es la MISMA mezcla de suaves y duros, la de
        la liga: que en un parque haya mas elevados duros (95+ mph al salir
        del bat) no es algo que haga el aire con el batazo
          segun_recta  rodado y elevado {mar, cdmx, cambio (+ _lo, _hi, _sig)}
                       con la recta de valor contra densidad de
                       outcome_values.json, y contraste (+ _lo, _hi, _sig) =
                       cambio del elevado menos el del rodado. Intervalos
                       quitando una categoria de altitud a la vez; ic_fiable
                       (False con menos de 6: "orientativo") y n_puntos
          observado    rodado y elevado {mar, casa, cambio (+ _lo, _hi, _sig),
                       batazos_mar, batazos_casa}: run value medio de los
                       batazos en casa y en banda mar (por clase, con la
                       mezcla fija); contraste (+ _lo, _hi, _sig) y pitchers.
                       Intervalos remuestreando pitchers.
                       Incluye a los bateadores locales: no separa el aire del
                       equipo de casa
          veredicto, regla
      segun_modelo: pendiente del costo de lanzar en CDMX (costo_cdmx_rv100,
        del contrafactual) contra el % de rodados, entre pitcher-temporadas
        calificadas: pendiente (+ _lo, _hi, _lo_folds, _hi_folds, _sig),
        pitcher_temporadas, pitchers, unidad, veredicto. Es lo que IMPLICA el
        modelo, no una observacion
      veredicto, regla
  parametros  los umbrales de este modulo, para citarlos sin escribirlos a mano
"""
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

# config.OUT y config.MODELS se leen al usarse, para que una prueba pueda cambiarlos
import config
from config import RHO_CDMX, RHO_MAR, MIN_PITCHES_PITCHER, MIN_PITCHERS_BETA
from features import TIPOS
from model import outcome_values, valores_en, COLS_VALOR
from physics import FAMILIAS
from score import remuestreo, sumas, razon, ic, N_BOOT

MIN_PITCHERS = MIN_PITCHERS_BETA   # pitchers para afirmar una diferencia (el mismo umbral que score.por_tipo)
MIN_FORMA = 1500                   # pitcheos de un tipo para partirlo en terciles (500 por tercil)
MIN_PERFIL = 100                   # pitcheos de un arsenal para entrar al perfil
MIN_ARSENALES = 16                 # arsenales (o pitcher-temporadas) para separar un 25% mejor: al menos 4 arriba
TOP = 0.25
MIN_TOP = 400                      # pitcheos de la temporada para la lista de los mejores en CDMX
N_LISTA = 10
MIN_BANDA_H3 = 150                 # pitcheos en CADA banda (alta y mar) para la prueba de H3
SENSIBILIDAD_H3 = (100, 50)        # umbrales mas bajos que se reportan aparte

TERCILES = ("bajo", "medio", "alto")
# clave -> (columna de scores_pitch, signo, etiqueta para el documento, unidad). magnitud = signo x columna
FORMA = {
    "ride":        ("ivb_neutral", 1, "Ride (IVB)", "pulg"),
    "hundimiento": ("ivb_neutral", -1, "Hundimiento (IVB con signo contrario)", "pulg"),
    "caida":       ("ivb_neutral", -1, "Caída (IVB con signo contrario)", "pulg"),
    "corrida":     ("hb_neutral", 1, "Corrida al lado del brazo (HB)", "pulg"),
    "barrido":     ("hb_neutral", -1, "Barrido al lado del guante (HB con signo contrario)", "pulg"),
    "corte":       ("hb_neutral", -1, "Corte al lado del guante (HB con signo contrario)", "pulg"),
    "velocidad":   ("velocity", 1, "Velocidad", "mph"),
    "dif_velo":    ("velo_diff_vs_fb", -1, "Diferencial de velocidad con la recta", "mph"),
    "spin":        ("spin_rate", 1, "Spin", "rpm"),
}
FORMAS_DE = {"FF": ("ride", "velocidad", "corrida"), "SI": ("corrida", "hundimiento", "velocidad"),
             "FC": ("corte", "ride", "velocidad"), "SL": ("barrido", "caida", "velocidad", "spin"),
             "CU": ("caida", "barrido", "velocidad", "spin"), "CH": ("dif_velo", "corrida", "hundimiento"),
             "FS": ("dif_velo", "hundimiento", "corrida")}
# (cifra, prefijo de sus intervalos, columna por pitcheo, centro del nivel)
MEDIDAS = (("stuff_cdmx", "stuff_cdmx", "stuff_cdmx_plus", 100.0), ("stuff_mar", "stuff_mar", "stuff_mar_plus", 100.0),
           ("delta_altitud", "delta", "delta_altitud", 0.0))
# (rasgo = columna de scores_arsenal, etiqueta para el documento, unidad, es control)
RASGOS = (("velo", "Velocidad", "mph", False),
          ("dif_velo", "Diferencial de velocidad con la recta", "mph", False),
          ("spin", "Spin", "rpm", False),
          ("ivb_cdmx", "IVB en el Harp Helú", "pulg", False),
          ("hb_cdmx", "HB en el Harp Helú (+ = lado del brazo)", "pulg", False),
          ("ext", "Extensión", "pies", False),
          ("rel_h", "Altura de salida", "pies", False),
          ("rel_s", "Lado de salida (+ = lado del brazo)", "pies", False),
          ("rho", "Densidad media del aire donde lanzó (control)", "densidad relativa", True))
SIN_DIFERENCIAL = ("FF", "SI")     # casi siempre SON la recta de referencia: quedarian dos o tres arsenales
# las mismas familias que physics.py y que las filas FAM_* de val_contrafactual.csv (el cutter no esta en ninguna)
CAMBIO, QUIEBRE, RECTAS = FAMILIAS["FAM_CAMBIO"], FAMILIAS["FAM_QUIEBRE"], FAMILIAS["FAM_RECTAS"]
DECIDE_H2 = "contraste_liga"       # cual de los tres contrastes de H2 da el veredicto
BATAZOS = {"rodado": ("ip_gb",), "elevado": ("ip_fb_soft", "ip_fb_hard")}

_INTERVALO = ("_lo", "_hi", "_lo_folds", "_hi_folds", "_sig")
COLS_FORMAS = (["pt", "forma", "etiqueta", "unidad", "columna", "signo", "tercil", "valor_medio", "pitcheos",
                "pitchers"] + [c for nombre, p, _, _ in MEDIDAS for c in [nombre] + [p + s for s in _INTERVALO]])
COLS_PERFIL = (["pt", "rasgo", "etiqueta", "unidad", "control", "arsenales", "n_top", "n_resto", "pitchers_top",
                "corte_top"] + [f"{g}_{q}" for g in ("top", "resto", "liga") for q in ("mediana", "p25", "p75")]
               + ["dif"] + ["dif" + s for s in _INTERVALO]
               + ["top_mar_mediana", "dif_cdmx_mar"] + ["dif_cdmx_mar" + s for s in _INTERVALO])
COLS_MIX = (["pt", "uso_top", "uso_resto", "uso_liga", "n_top", "n_resto", "pitchers_top", "dif"]
            + ["dif" + s for s in _INTERVALO])
COLS_LISTA = ["pitcher_id", "pitcher_name", "season", "pitcheos", "lefty", "pct_alta", "velo_recta",
              "stuff_plus_enc", "stuff_lo", "stuff_hi", "stuff_cdmx_plus_enc", "stuff_cdmx_lo", "stuff_cdmx_hi",
              "stuff_mar_plus", "delta_altitud", "delta_lo", "delta_hi", "delta_abs_max", "costo_cdmx_rv100",
              "location_plus_enc", "pitching_plus_enc", "pitching_neutral_plus_enc", "pitching_neutral_lo",
              "pitching_neutral_hi", "swstr_pct", "whiff_swing_pct", "gb_pct", "rv100", "fold"]
COLS_NUEVAS = ("delta_abs_max", "fold")     # las dos de COLS_LISTA que no vienen en scores_pitcher.csv
COLS_TIPO = ["pitcheos", "pitchers", "stuff_mar", "stuff_cdmx", "delta", "delta_lo", "delta_hi", "delta_lo_folds",
             "delta_hi_folds", "delta_mov", "delta_valor", "costo_cdmx_rv100", "significativo"]
# lo que se lee de cada archivo de score.py (si falta algo se dice que y se detiene)
COLS_PITCH = ["pitcher_id", "season", "pt", "fb_type", "fold", "banda", "es_casa", "air_density", "count", "outcome",
              "run_value", "y_gb", "y_bip", "velocity", "spin_rate", "velo_diff_vs_fb", "ivb_neutral",
              "hb_neutral", "stuff_cdmx_plus", "stuff_mar_plus", "delta_altitud", "delta_mov"]
COLS_ARSENAL = ["pitcher_id", "season", "pt", "pitcheos", "velo", "velo_recta", "spin", "ivb_cdmx", "hb_cdmx",
                "ext", "rel_h", "rel_s", "rho", "stuff_cdmx_plus_enc", "stuff_mar_plus"]

# H2 se decide con la frase del reto ("ganan valor relativo"): contra la liga. El contraste contra los quiebres
# prueba otra cosa (el mecanismo: diferencial de velocidad contra spin) y puede salir con el signo contrario.
HIPOTESIS_H2 = ("Los pitcheos que viven del diferencial de velocidad (cambio y split) ganan valor relativo en la "
                "altitud.")
HIPOTESIS_H3 = "El pitcher de rodados sufre menos la altitud: su run value sube menos al lanzar arriba."
REGLA_H2 = ("Contraste = cambio relativo medio (delta_altitud, puntos de Stuff+ por pitcheo) de "
            f"{'+'.join(CAMBIO)} menos el de todos los pitcheos de la liga, donde ese cambio promedia 0. Apoyada si "
            "es positivo y significativo; contraria si es negativo y significativo; no concluyente si alguno de "
            "sus dos intervalos (pitchers, modelos de fold) contiene 0; sin datos con menos de "
            f"{MIN_PITCHERS} pitchers. Los contrastes contra {'+'.join(QUIEBRE)} y contra {'+'.join(RECTAS)} "
            "llevan su propio veredicto con la misma regla y no cambian el de H2.")
REGLA_H3 = ("Decide la prueba dentro del mismo pitcher: pendiente de (carreras por 100 pitcheos en banda alta menos "
            "en banda mar) contra el % de rodados de la temporada. Apoyada si la pendiente es negativa y su "
            "intervalo excluye 0; contraria si es positiva y lo excluye; no concluyente si lo contiene; sin datos "
            f"si no hay {MIN_PITCHERS} pitchers con {MIN_BANDA_H3}+ pitcheos en cada banda. El contraste de batazos "
            "y la pendiente según el modelo se reportan aparte y no cambian el veredicto.")
REGLA_BATAZOS = ("Contraste = lo que sube el valor de un elevado menos lo que sube el de un rodado al pasar del "
                 "nivel del mar a CDMX, con la mezcla de elevados suaves y duros fija en la de la liga. Apoyada si "
                 "el contraste observado (casa contra banda mar) es positivo y significativo y el de la recta de "
                 "valor contra densidad tiene el mismo signo y un intervalo que también excluye 0 (orientativo con "
                 "menos de 6 categorías de altitud). Contraria, lo mismo con signo negativo. El observado incluye a "
                 "los bateadores del equipo local: por sí solo no separa el aire del equipo.")


# ── intervalos ──────────────────────────────────────────────────────────────
def pesos(t):
    """(cod, WF, k) para remuestrear las filas de `t` (pide pitcher_id y fold).
    cod es el codigo del pitcher de cada fila. Cada renglon de WF da el peso de
    cada pitcher: [0] la muestra real, luego N_BOOT replicas de pitchers
    completos (score.remuestreo) y al final k renglones, uno por modelo de
    fold, con 1 en los pitchers que califico ese modelo. Asi una sola cuenta
    devuelve la cifra, sus replicas y lo que da cada modelo por separado."""
    if not len(t):
        return np.zeros(0, dtype=int), np.zeros((N_BOOT + 1, 0)), 0
    cod, _, W = remuestreo(t["pitcher_id"])
    fold = np.zeros(W.shape[1], dtype=int)
    fold[cod] = t["fold"].to_numpy(int)
    folds = np.unique(fold)
    return cod, np.vstack([W, (fold[None, :] == folds[:, None]).astype(float)]), len(folds)


def _fuera(lo, hi, ref=0.0):
    """El intervalo deja fuera a ref (con NaN, False)."""
    return bool(lo > ref or hi < ref)


def medida(nombre, v, pitchers, k=0, ref=0.0, prefijo=None):
    """Columnas de una cifra v = [muestra real, replicas..., k modelos de fold]:
    la cifra, sus intervalos y <prefijo>_sig (ver el encabezado del modulo)."""
    p = prefijo or nombre
    valor, lo, hi = ic(v[:len(v) - k], pitchers)
    out, sig = {nombre: valor, p + "_lo": lo, p + "_hi": hi}, pitchers >= MIN_PITCHERS and _fuera(lo, hi, ref)
    if k:
        f = v[len(v) - k:]
        f = f[np.isfinite(f)]
        margen = stats.t.ppf(0.975, len(f) - 1) * f.std(ddof=1) / np.sqrt(len(f)) if len(f) > 1 else np.nan
        out.update({p + "_lo_folds": valor - margen, p + "_hi_folds": valor + margen})
        sig = sig and _fuera(valor - margen, valor + margen, ref)
    return {**out, p + "_sig": bool(sig)}


def medias(cod, WF, filas, *vs):
    """Media de cada arreglo de vs en `filas` (mascara o posiciones), cada una
    como [muestra real, replicas..., modelos de fold...]."""
    c = cod[filas]
    n = sumas(c, WF, np.ones(len(c)))
    return [razon(sumas(c, WF, v[filas]), n) for v in vs]


def cuantil(x, w, q):
    """Cuantil q de x con cada renglon de pesos enteros de w (renglones x
    filas). Es np.quantile sobre la muestra con cada fila repetida w veces;
    NaN en un renglon sin peso."""
    orden = np.argsort(x, kind="stable")
    xs, acum = x[orden], np.cumsum(w[:, orden], axis=1)
    n = acum[:, -1]
    h = (n - 1) * q
    j = np.floor(h)
    a = xs[(acum > j[:, None]).argmax(1)]
    b = xs[(acum > np.minimum(j + 1, n - 1)[:, None]).argmax(1)]
    return np.where(n > 0, a + (h - j) * (b - a), np.nan)


def pendiente(x, y, w):
    """Pendiente de minimos cuadrados de y en x con cada renglon de pesos de w."""
    n = w.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = w @ x / n, w @ y / n
        sxx = w @ (x * x) / n - mx ** 2
        return np.where(sxx > 1e-9, (w @ (x * y) / n - mx * my) / sxx, np.nan)


def veredicto(a_favor, significativo, hay_datos):
    """La unica regla de veredicto: `a_favor` es la cifra orientada (positiva =
    en el sentido de la hipotesis) y `significativo` sale de sus intervalos."""
    if not hay_datos or not np.isfinite(a_favor):
        return "sin datos"
    if not significativo:
        return "no concluyente"
    return "apoyada" if a_favor > 0 else "contraria"


# ── formas ──────────────────────────────────────────────────────────────────
def _cifras(vs, pitchers, k, refs):
    """Las tres cifras de MEDIDAS de una fila de insights_formas.csv con sus intervalos."""
    out = {}
    for (nombre, p, _, _), v, ref in zip(MEDIDAS, vs, refs):
        out.update(medida(nombre, v, pitchers, k, ref, p))
    return out


def formas(sp):
    """insights_formas.csv: dentro de cada tipo, terciles de cada forma y la
    diferencia del tercil alto menos el bajo (columnas en el encabezado).
    Los terciles tienen el mismo numero de pitcheos y sus cortes son los de la
    muestra real. El nivel de un tercil se mide contra la liga: en cada replica
    o modelo de fold, contra la liga de esa replica o de ese fold."""
    cod, WF, k = pesos(sp)
    por_pitcheo = [sp[col].to_numpy(float) for _, _, col, _ in MEDIDAS]
    centro = [c for _, _, _, c in MEDIDAS]
    liga = medias(cod, WF, slice(None), *por_pitcheo)
    pt, filas = sp["pt"].to_numpy(), []
    for tipo, claves in FORMAS_DE.items():
        for clave in claves:
            col, signo, etiqueta, unidad = FORMA[clave]
            x = signo * sp[col].to_numpy(float)
            del_tipo = np.flatnonzero((pt == tipo) & np.isfinite(x))
            if len(del_tipo) < MIN_FORMA:
                continue
            tercil = pd.qcut(pd.Series(x[del_tipo]).rank(method="first"), 3, labels=False).to_numpy()
            base = {"pt": tipo, "forma": clave, "etiqueta": etiqueta, "unidad": unidad, "columna": col, "signo": signo}
            parte = {}
            for i, nombre in enumerate(TERCILES):
                f = del_tipo[tercil == i]
                parte[nombre] = f, medias(cod, WF, f, *por_pitcheo), len(np.unique(cod[f]))
                nivel = [c + m - todos for c, m, todos in zip(centro, parte[nombre][1], liga)]
                filas.append({**base, "tercil": nombre, "valor_medio": x[f].mean(), "pitcheos": len(f),
                              "pitchers": parte[nombre][2], **_cifras(nivel, parte[nombre][2], k, centro)})
            (fa, alto, pa), (fb, bajo, pb) = parte["alto"], parte["bajo"]
            filas.append({**base, "tercil": "alto-bajo", "valor_medio": x[fa].mean() - x[fb].mean(),
                          "pitcheos": len(fa) + len(fb), "pitchers": len(np.unique(cod[np.concatenate([fa, fb])])),
                          **_cifras([a - b for a, b in zip(alto, bajo)], min(pa, pb), k, [0.0] * len(MEDIDAS))})
    return pd.DataFrame(filas, columns=COLS_FORMAS)


# ── perfil por pitcheo y mezcla ─────────────────────────────────────────────
def _top_y_resto(enc, w, hay):
    """Pesos (renglones x filas) del 25% con mayor `enc` y del resto. El corte
    se vuelve a calcular en cada renglon de w: cada replica y cada fold eligen
    a su propio 25% mejor. Sin filas suficientes los dos grupos quedan vacios."""
    if not hay:
        return w * 0.0, w * 0.0
    arriba = enc[None, :] >= cuantil(enc, w, 1 - TOP)[:, None]
    return w * arriba, w * ~arriba


def perfil_por_pitcheo(ars):
    """insights_perfil.csv: por tipo de pitcheo, los arsenales con mejor Stuff+
    en CDMX (25% superior de stuff_cdmx_plus_enc) contra el resto y contra
    todos, rasgo por rasgo, en mediana y P25-P75 (columnas en el encabezado).
    Ademas, contra el 25% mejor AL NIVEL DEL MAR (stuff_mar_plus): lo que
    distingue a los dos grupos es lo propio del Harp Helu; lo demas es ser
    bueno en cualquier parque. `ars` es scores_arsenal.csv con las columnas
    fold y fb_type (la recta primaria de la pitcher-temporada, de cargar())."""
    a = ars[ars["pitcheos"] >= MIN_PERFIL].reset_index(drop=True)
    # un cutter que es la recta primaria de su pitcher tiene diferencial 0 por construccion: no es un rasgo
    a["dif_velo"] = (a["velo_recta"] - a["velo"]).where(a["pt"] != a["fb_type"])
    cod, WF, k = pesos(a)
    filas = []
    for tipo in TIPOS:
        idx = np.flatnonzero((a["pt"] == tipo).to_numpy())
        if not len(idx):
            continue
        d, w = a.iloc[idx], WF[:, cod[idx]]
        enc = d["stuff_cdmx_plus_enc"].to_numpy(float)
        hay = len(idx) >= MIN_ARSENALES
        top, resto = _top_y_resto(enc, w, hay)
        top_mar = _top_y_resto(d["stuff_mar_plus"].to_numpy(float), w, hay)[0]
        pitchers_top = len(np.unique(cod[idx][top[0] > 0]))
        base = {"pt": tipo, "arsenales": len(idx), "n_top": int(top[0].sum()), "n_resto": int(resto[0].sum()),
                "pitchers_top": pitchers_top, "corte_top": float(np.quantile(enc, 1 - TOP)) if hay else np.nan}
        for rasgo, etiqueta, unidad, control in RASGOS:
            x = d[rasgo].to_numpy(float)
            ok = np.isfinite(x)
            if (rasgo == "dif_velo" and tipo in SIN_DIFERENCIAL) or not ok.any():
                continue
            x, grupos = x[ok], {"top": top[:, ok], "resto": resto[:, ok], "liga": w[:, ok]}
            fila = {**base, "rasgo": rasgo, "etiqueta": etiqueta, "unidad": unidad, "control": control}
            for g, wg in grupos.items():
                fila.update({f"{g}_{nombre}": float(cuantil(x, wg[:1], q)[0])
                             for nombre, q in (("mediana", 0.5), ("p25", 0.25), ("p75", 0.75))})
            en_top, en_resto, en_top_mar = (cuantil(x, wg, 0.5)
                                            for wg in (grupos["top"], grupos["resto"], top_mar[:, ok]))
            filas.append({**fila, **medida("dif", en_top - en_resto, pitchers_top, k),
                          "top_mar_mediana": float(en_top_mar[0]),
                          **medida("dif_cdmx_mar", en_top - en_top_mar, pitchers_top, k)})
    return pd.DataFrame(filas, columns=COLS_PERFIL)


def mezcla(sp, pit):
    """insights_mix.csv: uso de cada tipo de pitcheo (sobre TODOS los pitcheos
    de la pitcher-temporada, 0 si no lo lanza) en el 25% mejor en CDMX contra
    el resto, a nivel pitcher-temporada. `pit` es scores_pitcher.csv con fold."""
    uso = (pd.crosstab([sp["pitcher_id"], sp["season"]], sp["pt"], normalize="index")
           .reindex(columns=list(TIPOS), fill_value=0.0))
    U = uso.reindex(pd.MultiIndex.from_frame(pit[["pitcher_id", "season"]])).to_numpy(float)
    assert np.allclose(U.sum(1), 1.0), "el uso de cada pitcher-temporada debe sumar 1"
    cod, WF, k = pesos(pit)
    w = WF[:, cod]
    top, resto = _top_y_resto(pit["stuff_cdmx_plus_enc"].to_numpy(float), w, len(pit) >= MIN_ARSENALES)
    media = lambda wg: razon(wg @ U, wg.sum(1)[:, None])           # (renglones, tipos)
    u_top, u_resto, u_liga = media(top), media(resto), media(w[:1])
    pitchers_top = len(np.unique(cod[top[0] > 0]))
    filas = [{"pt": pt, "uso_top": u_top[0, j], "uso_resto": u_resto[0, j], "uso_liga": u_liga[0, j],
              "n_top": int(top[0].sum()), "n_resto": int(resto[0].sum()), "pitchers_top": pitchers_top,
              **medida("dif", u_top[:, j] - u_resto[:, j], pitchers_top, k)} for j, pt in enumerate(TIPOS)]
    return pd.DataFrame(filas, columns=COLS_MIX)


# ── hipotesis ───────────────────────────────────────────────────────────────
def h2(sp, tipo):
    """H2: los pitcheos de diferencial de velocidad (CH, FS) ganan valor
    relativo en la altitud? Con el delta_altitud del contrafactual; `tipo` es
    altitude_by_pitch_type.csv. Tres contrastes, cada uno con su veredicto;
    decide el de DECIDE_H2 (contra la liga). Llaves en el encabezado."""
    cod, WF, k = pesos(sp)
    pt = sp["pt"].to_numpy()
    delta, mov = sp["delta_altitud"].to_numpy(float), sp["delta_mov"].to_numpy(float)

    def contraste(a, b):
        fa, fb = (np.flatnonzero(np.isin(pt, g)) for g in (a, b))
        pa, pb = (len(np.unique(cod[f])) for f in (fa, fb))
        dif = [x - y for x, y in zip(medias(cod, WF, fa, delta, mov), medias(cod, WF, fb, delta, mov))]
        c = {"a": list(a), "b": list(b), "pitcheos_a": len(fa), "pitcheos_b": len(fb), "pitchers_a": pa,
             "pitchers_b": pb, **medida("delta", dif[0], min(pa, pb), k),
             **medida("delta_mov", dif[1], min(pa, pb), k)}
        return {**c, "veredicto": veredicto(c["delta"], c["delta_sig"], min(pa, pb) >= MIN_PITCHERS)}

    c = {"contraste_liga": contraste(CAMBIO, TIPOS), "contraste": contraste(CAMBIO, QUIEBRE),
         "contraste_rectas": contraste(CAMBIO, RECTAS)}
    return {"hipotesis": HIPOTESIS_H2, "por_tipo": tipo.set_index("pt")[COLS_TIPO].to_dict("index"), **c,
            "decide": DECIDE_H2, "veredicto": c[DECIDE_H2]["veredicto"], "regla": REGLA_H2}


def _prueba_rodados(t, W, minimo):
    """H3 dentro del mismo pitcher: entre las pitcher-temporadas de `t` con
    `minimo`+ pitcheos en CADA banda, pendiente de (carreras por 100 en banda
    alta menos en banda mar) contra el % de rodados de la temporada. Cada una
    pesa n_alta * n_mar / (n_alta + n_mar): la precision de su diferencia.
    semiancho, fiabilidad_gb y pendiente_corregida (ver el encabezado) dicen
    que tan chica es la pendiente que esta prueba puede ver."""
    e = t[(t["n_alta"] >= minimo) & (t["n_mar"] >= minimo) & (t["bip"] > 0)]
    pitchers, nada = int(e["cod"].nunique()), np.full(len(W), np.nan)
    out = {"min_pitcheos_por_banda": minimo, "pitcher_temporadas": len(e), "pitchers": pitchers,
           "unidad": "carreras por 100 pitcheos por cada 10 puntos de % de rodados"}
    b = dif = gb = nada
    fiab = np.nan
    if pitchers >= MIN_PITCHERS:
        n_a, n_m = e["n_alta"].to_numpy(float), e["n_mar"].to_numpy(float)
        p, bip = (e["gb"] / e["bip"]).to_numpy(float), e["bip"].to_numpy(float)
        x = 10 * p
        y = (e["rv_alta"] / e["n_alta"] - e["rv_mar"] / e["n_mar"]).to_numpy(float)
        w = W[:, e["cod"].to_numpy()] * (n_a * n_m / (n_a + n_m))
        b, dif, gb = pendiente(x, y, w), razon(w @ y, w.sum(1)), razon(w @ x, w.sum(1)) / 10
        # el % de rodados de una temporada trae ruido binomial: la pendiente sale multiplicada por la parte de
        # su varianza que SI es del pitcher (atenuacion de un regresor medido con error)
        var = float(np.average((p - np.average(p, weights=w[0])) ** 2, weights=w[0]))
        if var > 0:
            fiab = 1.0 - float(np.average(p * (1 - p) / bip, weights=w[0])) / var
    out.update({**medida("pendiente", b, pitchers), **medida("dif_media", dif, pitchers), "gb_medio": float(gb[0])})
    fiab = fiab if fiab > 0 else np.nan              # sin varianza propia no hay nada que corregir (NaN -> NaN)
    out.update({"semiancho": (out["pendiente_hi"] - out["pendiente_lo"]) / 2, "fiabilidad_gb": fiab,
                "pendiente_corregida": out["pendiente"] / fiab})
    return {**out, "veredicto": veredicto(-out["pendiente"], out["pendiente_sig"], pitchers >= MIN_PITCHERS)}


def _contraste_batazos(sp, valores, cod, W):
    """H3 por el lado del batazo: (lo que sube el valor de un elevado) menos
    (lo que sube el de un rodado) al pasar del nivel del mar a CDMX. Positivo =
    el aire delgado castiga mas al elevado. Dos lecturas (ver el encabezado),
    las dos con la mezcla de elevados suaves y duros de la liga (`parte`)."""
    clase, rv = sp["outcome"].to_numpy(), sp["run_value"].to_numpy(float)
    casa, mar = (sp["es_casa"] == 1).to_numpy(), (sp["banda"] == "mar").to_numpy()
    n = sp["outcome"].value_counts()
    parte = {g: {c: n.get(c, 0) / max(1, sum(n.get(x, 0) for x in clases)) for c in clases}
             for g, clases in BATAZOS.items()}

    def con_recta(v):
        """{grupo: (valor al nivel del mar, valor en CDMX)} con la recta de valor contra densidad de v."""
        aires = valores_en(v, RHO_MAR), valores_en(v, RHO_CDMX)
        return {g: tuple(sum(p * aire[c] for c, p in parte[g].items()) for aire in aires) for g in BATAZOS}

    def cambios(r):
        """[lo que sube el rodado, lo que sube el elevado, contraste] de un con_recta()."""
        rodado, elevado = (r[g][1] - r[g][0] for g in ("rodado", "elevado"))
        return np.array([rodado, elevado, elevado - rodado])

    recta = con_recta(valores)
    d, real, margen = sp[COLS_VALOR], cambios(recta), np.full(3, np.nan)
    estadios = d["stadium"].unique()
    K = len(estadios)
    if K >= 3:      # la recta se vuelve a ajustar quitando una categoria de altitud a la vez, como en physics.py
        sin_uno = np.array([cambios(con_recta(outcome_values(d[d["stadium"] != e]))) for e in estadios])
        margen = stats.t.ppf(0.975, K - 1) * np.sqrt((K - 1) / K * ((sin_uno - sin_uno.mean(0)) ** 2).sum(0))
    lo, hi = real - margen, real + margen
    con_margen = lambda nombre, i: {nombre: real[i], nombre + "_lo": lo[i], nombre + "_hi": hi[i],
                                    nombre + "_sig": _fuera(lo[i], hi[i])}
    segun_recta = {"rodado": {"mar": recta["rodado"][0], "cdmx": recta["rodado"][1], **con_margen("cambio", 0)},
                   "elevado": {"mar": recta["elevado"][0], "cdmx": recta["elevado"][1], **con_margen("cambio", 1)},
                   **con_margen("contraste", 2), "ic_fiable": bool(K >= 6), "n_puntos": int(K)}

    def en_banda(g, banda):
        """(run value medio del grupo g en esa banda, pitchers de la clase que menos tiene). Clase por clase y
        con la mezcla fija de la recta: juntando suaves y duros con la mezcla de cada parque, que en casa haya
        mas elevados duros contaba como "el elevado vale mas", y eso no lo hace el aire. NaN si falta una clase."""
        con = [(clase == c, p) for c, p in parte[g].items() if p > 0]
        v = sum((p * medias(cod, W, m & banda, rv)[0] for m, p in con), np.full(len(W), 0.0 if con else np.nan))
        return v, min([len(np.unique(cod[m & banda])) for m, _ in con], default=0)

    observado, cambio, pitchers = {}, {}, []
    for g, clases in BATAZOS.items():
        (en_casa, p_casa), (en_mar, p_mar) = en_banda(g, casa), en_banda(g, mar)
        cambio[g], del_grupo, m = en_casa - en_mar, min(p_casa, p_mar), np.isin(clase, clases)
        pitchers.append(del_grupo)
        observado[g] = {"mar": en_mar[0], "casa": en_casa[0], **medida("cambio", cambio[g], del_grupo),
                        "batazos_mar": int((m & mar).sum()), "batazos_casa": int((m & casa).sum())}
    observado.update({"pitchers": min(pitchers),
                      **medida("contraste", cambio["elevado"] - cambio["rodado"], min(pitchers))})
    # casa contra banda mar compara dos grupos de parques y de equipos: solo concluye si la recta de valor
    # contra densidad, que usa todos los niveles, va en el mismo sentido Y su intervalo tambien excluye 0
    firme = (observado["contraste_sig"] and observado["contraste"] * segun_recta["contraste"] > 0
             and segun_recta["contraste_sig"])
    return {"unidad": "carreras por batazo", "segun_recta": segun_recta, "observado": observado,
            "veredicto": veredicto(observado["contraste"], firme, min(pitchers) >= MIN_PITCHERS),
            "regla": REGLA_BATAZOS}


def h3(sp, pit, valores):
    """H3: el pitcher de rodados sufre menos la altitud? Tres piezas (llaves en
    el encabezado): la prueba dentro del mismo pitcher con resultados
    OBSERVADOS, que es la UNICA que decide (sin ella el veredicto es "sin
    datos"); el contraste de valor de los batazos; y lo que implica el modelo.
    Ver REGLA_H3."""
    cod, WF, k = pesos(sp)
    W = WF[:len(WF) - k]              # resultados observados: no dependen del modelo de fold
    alta, mar, rv = sp["banda"] == "alta", sp["banda"] == "mar", 100 * sp["run_value"].astype(float)
    temporada = lambda s: s.groupby([sp["pitcher_id"], sp["season"]], sort=False)
    t = pd.DataFrame({"n_alta": temporada(alta).sum(), "n_mar": temporada(mar).sum(),
                      "rv_alta": temporada(rv.where(alta, 0.0)).sum(), "rv_mar": temporada(rv.where(mar, 0.0)).sum(),
                      "gb": temporada(sp["y_gb"]).sum(), "bip": temporada(sp["y_bip"]).sum(),
                      "cod": temporada(pd.Series(cod, index=sp.index)).first()})
    prueba = _prueba_rodados(t, W, MIN_BANDA_H3)
    batazos = _contraste_batazos(sp, valores, cod, W)

    q = pit[pit["gb_pct"].notna()]
    cod_q, WF_q, k_q = pesos(q)
    b = pendiente(10 * q["gb_pct"].to_numpy(float), q["costo_cdmx_rv100"].to_numpy(float),
                  WF_q[:, cod_q] * q["pitcheos"].to_numpy(float))
    modelo = {"pitcher_temporadas": len(q), "pitchers": int(q["pitcher_id"].nunique()),
              "unidad": "carreras por 100 pitcheos de costo en CDMX por cada 10 puntos de % de rodados",
              **medida("pendiente", b, q["pitcher_id"].nunique(), k_q)}
    modelo["veredicto"] = veredicto(-modelo["pendiente"], modelo["pendiente_sig"], modelo["pitchers"] >= MIN_PITCHERS)

    return {"hipotesis": HIPOTESIS_H3, "pitcher": prueba,
            "sensibilidad": {str(m): _prueba_rodados(t, W, m) for m in SENSIBILIDAD_H3},
            "batazos": batazos, "segun_modelo": modelo, "veredicto": prueba["veredicto"], "regla": REGLA_H3}


# ── listas ──────────────────────────────────────────────────────────────────
def listas(pit):
    """Las tres listas de pitcher-temporadas (reglas y columnas en el encabezado)."""
    p = pit.assign(delta_abs_max=np.maximum(pit["delta_lo"].abs(), pit["delta_hi"].abs()))
    return {"top_cdmx": p[p["pitcheos"] >= MIN_TOP].nlargest(N_LISTA, "stuff_cdmx_plus_enc")[COLS_LISTA],
            "afectados": p[p["delta_hi"] < 0].nsmallest(N_LISTA, "delta_altitud")[COLS_LISTA],
            "consistentes": p.nsmallest(N_LISTA, "delta_abs_max")[COLS_LISTA]}


# ── corrida ─────────────────────────────────────────────────────────────────
def cargar():
    """Lo que escribio score.py: (pitcheos, pitcher-temporadas, arsenales, efecto
    por tipo, valores de cada resultado). De scores_pitch.parquet solo se leen
    las columnas que se usan (el archivo completo trae muchas mas). La
    categoria de altitud se lee con el nombre stadium, que es como la pide
    model.COLS_VALOR (hace de "estadio")."""
    rutas = {n: config.OUT / n for n in ("scores_pitch.parquet", "scores_pitcher.csv", "scores_arsenal.csv",
                                         "altitude_by_pitch_type.csv")}
    rutas["outcome_values.json"] = config.MODELS / "outcome_values.json"
    faltan = [n for n, ruta in rutas.items() if not ruta.exists()]
    if faltan:
        # outcome_values.json lo escribe 'valores'; lo demas, 'score': no se manda repetir la validacion sin necesidad
        raise SystemExit(f"Faltan {faltan}: corre primero  "
                         + config.comando("valores" if "outcome_values.json" in faltan else "score"))
    hay = pq.read_schema(rutas["scores_pitch.parquet"]).names
    pit, ars, tipo = (pd.read_csv(rutas[n], dtype={"pitcher_id": str})
                      for n in ("scores_pitcher.csv", "scores_arsenal.csv", "altitude_by_pitch_type.csv"))
    pedido = (("scores_pitch.parquet", hay, COLS_PITCH + ["altitude_category"]),
              ("scores_pitcher.csv", pit, [c for c in COLS_LISTA if c not in COLS_NUEVAS]),
              ("scores_arsenal.csv", ars, COLS_ARSENAL), ("altitude_by_pitch_type.csv", tipo, ["pt"] + COLS_TIPO))
    for nombre, tabla, cols in pedido:
        falta = [c for c in cols if c not in tabla]
        if falta:
            raise SystemExit(f"A out/{nombre} le faltan columnas {falta}: vuelve a correr  {config.comando('score')}")
    sp = pd.read_parquet(rutas["scores_pitch.parquet"], columns=COLS_PITCH + ["altitude_category"])
    sp["pitcher_id"] = sp["pitcher_id"].astype(str)
    sp = sp.rename(columns={"altitude_category": "stadium"})      # la categoria de altitud hace de "estadio"
    # el fold y la recta primaria de cada pitcher-temporada; de paso comprueba que las tres tablas salieron
    # de la misma corrida
    fold = sp.groupby(["pitcher_id", "season"], as_index=False)[["fold", "fb_type"]].first()
    pit, ars = (t.merge(fold, on=["pitcher_id", "season"], how="left") for t in (pit, ars))
    if sp.empty or pit["fold"].isna().any() or ars["fold"].isna().any():
        raise SystemExit("scores_pitch.parquet esta vacio o no corresponde a scores_pitcher.csv / scores_arsenal.csv: "
                         f"vuelve a correr  {config.comando('score')}")
    return sp, pit, ars, tipo, json.loads(rutas["outcome_values.json"].read_text())


def _linea(nombre, d, llave):
    """Una cifra con sus intervalos, para la consola."""
    v = {c: d.get(llave + c, np.nan) for c in ("",) + _INTERVALO}
    if not np.isfinite(v[""]):
        return f"  {nombre}: sin datos"
    folds = f" | entre folds [{v['_lo_folds']:+.2f}, {v['_hi_folds']:+.2f}]" if llave + "_lo_folds" in d else ""
    return (f"  {nombre}: {v['']:+.2f} [{v['_lo']:+.2f}, {v['_hi']:+.2f}]{folds} | "
            f"significativo: {'si' if v['_sig'] else 'no'}")


def main():
    t0 = time.time()
    pd.set_option("display.width", 250)
    sp, pit, ars, tipo, valores = cargar()
    tablas = {"formas": formas(sp), "perfil": perfil_por_pitcheo(ars), "mix": mezcla(sp, pit), **listas(pit)}
    for nombre, tabla in tablas.items():
        tabla.to_csv(config.OUT / f"insights_{nombre}.csv", index=False)
    ins = {"H2": h2(sp, tipo), "H3": h3(sp, pit, valores),
           "parametros": {"remuestreos": N_BOOT, "modelos_de_fold": int(sp["fold"].nunique()),
                          "min_pitchers": MIN_PITCHERS, "min_pitcheos_forma": MIN_FORMA,
                          "min_pitcheos_arsenal_perfil": MIN_PERFIL, "min_arsenales_perfil": MIN_ARSENALES,
                          "top": TOP, "min_pitcheos_pitcher": MIN_PITCHES_PITCHER, "min_pitcheos_top_cdmx": MIN_TOP,
                          "min_pitcheos_por_banda_h3": MIN_BANDA_H3}}
    (config.OUT / "insights.json").write_text(json.dumps(ins, indent=1, default=lambda o: o.item()))

    f, p, mix = tablas["formas"], tablas["perfil"], tablas["mix"]
    print(f"{len(sp):,} pitcheos | {sp['pitcher_id'].nunique():,} pitchers | {ins['parametros']['modelos_de_fold']} "
          f"modelos de fold | {N_BOOT} remuestreos de pitchers")
    print("\nFORMAS (puntos de Stuff+ por pitcheo; quiebre a densidad media de la liga; 'alto' = mas de esa forma):")
    print(f[["pt", "forma", "tercil", "valor_medio", "pitcheos", "pitchers", "stuff_cdmx", "stuff_cdmx_lo",
             "stuff_cdmx_hi", "stuff_cdmx_sig", "stuff_mar", "delta_altitud", "delta_lo", "delta_hi",
             "delta_lo_folds", "delta_hi_folds", "delta_sig"]].round(2).to_string(index=False))
    sin_terciles = [t for t in FORMAS_DE if t not in set(f["pt"])]
    if sin_terciles:
        print(f"  sin terciles (menos de {MIN_FORMA:,} pitcheos): {', '.join(sin_terciles)}")
    print(f"\nPERFIL POR PITCHEO (arsenales con {MIN_PERFIL}+ pitcheos; top = 25% mejor en Stuff+ CDMX; IVB y HB en "
          "el aire del Harp Helu):")
    print(p.loc[p["n_top"] > 0, ["pt", "rasgo", "arsenales", "n_top", "top_mediana", "top_p25", "top_p75",
                                 "resto_mediana", "liga_mediana", "dif", "dif_lo", "dif_hi", "dif_sig",
                                 "top_mar_mediana", "dif_cdmx_mar", "dif_cdmx_mar_lo", "dif_cdmx_mar_hi",
                                 "dif_cdmx_mar_sig"]].round(2).to_string(index=False))
    pocos = p[p["n_top"] == 0].drop_duplicates("pt")
    if len(pocos):
        print(f"  solo mediana de la liga (menos de {MIN_ARSENALES} arsenales): "
              + ", ".join(f"{r.pt} ({r.arsenales})" for r in pocos.itertuples()))
    print("\nMEZCLA de las pitcher-temporadas calificadas (uso de cada tipo, top 25% en CDMX contra el resto):")
    print(mix.round(3).to_string(index=False))

    h, a = ins["H2"], "+".join(CAMBIO)
    print(f"\nH2, {a} ganan valor relativo en la altitud (delta en puntos por pitcheo): {h['veredicto'].upper()}")
    print(_linea(f"{a} contra la liga (decide)", h["contraste_liga"], "delta"))
    print(_linea("  de eso, por movimiento", h["contraste_liga"], "delta_mov"))
    for llave, b in (("contraste", QUIEBRE), ("contraste_rectas", RECTAS)):
        print(_linea(f"{a} menos {'+'.join(b)} ({h[llave]['veredicto']}, no decide)", h[llave], "delta"))
    h = ins["H3"]
    print(f"\nH3, el pitcher de rodados sufre menos la altitud: {h['veredicto'].upper()}")
    for d in [h["pitcher"]] + list(h["sensibilidad"].values()):
        print(_linea(f"mismo pitcher arriba y abajo, {d['min_pitcheos_por_banda']}+ pitcheos por banda "
                     f"({d['pitcher_temporadas']} pitcher-temporadas): carreras por 100 por cada 10 puntos de rodados",
                     d, "pendiente"))
    d = h["pitcher"]
    if np.isfinite(d["semiancho"]):
        print(f"    esa prueba no distingue pendientes de menos de {d['semiancho']:.2f}; el % de rodados tiene "
              f"fiabilidad {d['fiabilidad_gb']:.2f} (pendiente corregida por ese ruido: "
              f"{d['pendiente_corregida']:+.2f})")
    print(_linea(f"batazos ({h['batazos']['veredicto']}): elevado menos rodado, casa contra banda mar, observado",
                 h["batazos"]["observado"], "contraste"))
    print(_linea(f"  lo mismo con la recta de valor ({h['batazos']['segun_recta']['n_puntos']} puntos)",
                 h["batazos"]["segun_recta"], "contraste"))
    print(_linea(f"segun el modelo ({h['segun_modelo']['veredicto']}, no decide): costo en CDMX por cada 10 puntos "
                 "de rodados", h["segun_modelo"], "pendiente"))
    ver = ["pitcher_name", "season", "pitcheos", "stuff_cdmx_plus_enc", "stuff_cdmx_lo", "stuff_cdmx_hi",
           "stuff_plus_enc", "delta_altitud", "delta_lo", "delta_hi"]
    for nombre, titulo in (("top_cdmx", f"MEJORES EN CDMX ({MIN_TOP}+ pitcheos)"),
                           ("afectados", "MAS AFECTADOS (intervalo del delta completo bajo 0)"),
                           ("consistentes", "MAS CONSISTENTES (menor cambio posible dentro de su intervalo)")):
        print(f"\n{titulo}, puntos entre pitchers:")
        # los nombres pueden traer acentos y la consola de Windows no siempre los aguanta
        print(tablas[nombre][ver].round(1).to_string(index=False).encode("ascii", "replace").decode()
              if len(tablas[nombre]) else "  (ninguno)")
    print(f"\ninsights.py listo en {time.time() - t0:.0f}s: insights.json y insights_{{formas, perfil, mix, top_cdmx, "
          "afectados, consistentes}.csv")


if __name__ == "__main__":
    main()
