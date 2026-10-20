"""
DOCUMENTO TECNICO (6 a 10 paginas, carta) generado 100% de out/ y models/.
Si cambian los datos se vuelve a correr y todas las cifras, y todas las frases,
se actualizan solas.

REGLAS DEL TEXTO
  - Ninguna frase que afirme una direccion o un juicio ("gana", "pierde",
    "supera", "apoyada"...) es un texto fijo: sale de una funcion de la seccion
    FRASES que recibe la cifra y su intervalo (o una bandera de aceptacion que
    ya calculo el pipeline) y tiene salida para los dos sentidos, para "sin
    diferencia medible" / "no concluyente" y para "sin datos".
  - Cuando una cifra tiene dos intervalos (remuestreo de pitchers y entre
    modelos de fold) la frase usa el mas ancho (envolvente).
  - Cada cifra que decide algo lleva su intervalo o su n. Los puntos Plus dicen
    su escala: "por pitcheo" o "entre pitchers" (no se comparan).
  - Los intervalos de la fisica con pocos niveles de densidad (ic_fiable False:
    las 3 categorias de altitud del dataset oficial) se llaman "orientativos".
  - Un archivo que falta, viene vacio o es de otra version NO tumba el
    documento: ese bloque imprime "No estimable con estos datos" y se sigue.
  - Corrida rapida (out/run_info.json): marca bajo el titulo y en cada pie.

LEE (solo lo que usa)
  out/run_info.json                hash_codigo, rapido, pasos.*.{hash_codigo, rapido}
  out/altitude_elasticities.csv    var (ivb, hb_n, mov_total, velo_loss, spin_rate, velocity, sustentacion),
                                   pt, beta, ci_lo, ci_hi, ic_fiable, loo_min, loo_max, signo_estable,
                                   estimable, significativo, pitchers, media_mar, media_cdmx,
                                   delta_cdmx_vs_mar, delta_lo, delta_hi, beta_usada, beta_usada_lo,
                                   beta_usada_hi, ic_fiable_usada, fuente, motivo
  out/altitude_summary.json        rho_cdmx, rho_mar, n_estadios (niveles de densidad: categorias), carry
  out/park_table.csv               stadium (la categoria), banda, altitud, densidad, pitcheos, rv_por_100, ivb_ff
  out/valores_carrera.json         metodo, medias_entradas, juegos, ajuste.residuo_max, pesos (como se estimo
                                   el valor en carreras con el propio dataset: features.estimar_valores_carrera)
  out/adaptacion.csv               tipo, variable, unidad, diferencia, ic_lo, ic_hi, pitchers, significativo,
                                   nota (la fila es la misma cantidad que un placebo)
  out/chequeo_etiquetas.csv        derivada, n, acuerdo, usada, nota
  out/limpieza.json                batazos, sin_rastreo, sin_rastreo_por_evento, rastreados, rastreados_por_evento
  out/val_resumen.csv, val_season_holdout.csv   modelo, n, pitchers, logloss, corr_xrv, auc_*,
                                   mejora_vs_{global,cuenta}_pct con _ci_lo/_ci_hi, delta_brier_vs_cuenta,
                                   iters, backend (+ test_season, n_train)
  out/val_pitcher_holdout.csv      fold (cuantos folds)
  out/val_park_holdout.csv         escenario, submodelo, valuacion, n, n_juegos, rv_medio_real_x100,
                                   sesgo_por_100, sesgo_ci_lo, sesgo_ci_hi, rho_test, rho_train_min,
                                   rho_train_max, n_puntos
  out/val_calibracion.csv          objetivo, modelo, grupo, pred_media, obs_media
  out/val_calibracion_resumen.csv  objetivo, modelo, grupo, n, tasa_base, ece, ece_ruido, pendiente,
                                   pendiente_ci_lo, pendiente_ci_hi
  out/val_contrafactual.csv        bloque (neutralidad, transporte, transporte_beta_1, resultados,
                                   entre_pitchers), modelo, pt, variable, celdas, pitchers, nivel_mar,
                                   nivel_alta, predicho, observado, valor, ic_lo, ic_hi, error_abs, pasa,
                                   ic_dentro_de_tolerancia, con_potencia, factor_escala_pitcher, nota
  out/val_stability.json           split_half_*, estabilidad_*, predice_rv_*, dif_*_vs_rv, pares, pitchers,
                                   acuerdo_modelos_*, error_modelo_*_pts, desfase_modelos_*_pts
  out/val_h4_ablation.csv          conjunto, referencia, n_vars, n, auc, delta_auc (+ _lo, _hi),
                                   delta_auc_vs_spin (+ _lo, _hi)
  out/val_importancia_stuff.csv    variable, delta_logloss, delta_logloss_sd
  out/altitude_by_pitch_type.csv   pt, pitcheos, pitchers, stuff_mar, stuff_cdmx, delta, delta_lo/_hi,
                                   delta_lo_folds/_hi_folds, delta_mov, costo_cdmx_rv100, significativo
  out/altitude_park_effect.json    costo_cdmx_vs_mar_rv100, ic95_recta, recta_ic_fiable, recta_n_puntos,
                                   por_movimiento_rv100 (+_ic95), por_valor_batazos_rv100
                                   (+_ic95_recta), observado_casa_menos_mar_rv100 (+ observado_ic95),
                                   costo_entre_bandas_rv100, rho_casa_obs, rho_mar_obs,
                                   rango_pitchers_p10_p90
  out/park_scores.csv, scores_pitcher.csv, folds.csv
  out/scores_pitch.parquet         SOLO pt, season, ivb_mar, hb_mar, ivb_cdmx, hb_cdmx, stuff_mar_plus,
                                   stuff_cdmx_plus (y el numero de filas, de los metadatos)
  out/predicciones_por_pitcheo.csv.gz   SOLO el encabezado: que columnas trae el entregable por pitcheo
  out/insights_formas.csv, insights_perfil.csv, insights_mix.csv, insights_top_cdmx.csv, insights.json
  models/outcome_values.json, iters.json, {stuff,location,pitching}_fold0.joblib (solo feats_)
Escribe out/documento_tecnico.pdf y las figuras en out/fig/.
"""
import json
import os
import time
from xml.sax.saxutils import escape as esc

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                CondPageBreak, KeepTogether)

# config.OUT y config.MODELS se leen al usarse, para que una prueba pueda cambiarlos
import config

# Paleta ejecutiva: negro / grises; verde, rojo y ambar SOLO para marcar ganancia / perdida / duda
NEGRO, GRIS, GRIS_CLARO = colors.HexColor("#111111"), colors.HexColor("#555555"), colors.HexColor("#E8E8E8")
VERDE, ROJO, AMBAR = colors.HexColor("#C6E9C9"), colors.HexColor("#F4C7C3"), colors.HexColor("#FBE8B0")
C_GANA, C_PIERDE, C_NEUTRO, C_AZUL = "#15803D", "#B91C1C", "#9CA3AF", "#2563A8"

MARGEN = 0.7 * inch
ANCHO = letter[0] - 2 * MARGEN            # 7.1 pulgadas utiles
PAGINAS = (6, 10)                         # rango que pide el reto
META_PAGINAS = 9                          # meta del documento: si sale mas largo se intenta compactar
RAYA = "–"
NO_ESTIMABLE = "No estimable con estos datos"
MARCA_RAPIDA = "CORRIDA DE PRUEBA (30% de los pitchers)"
PLACEHOLDERS = {"Nombre del equipo", "Integrante 1", "Integrante 2", "Universidad"}
PASOS_DATOS = ("features", "physics", "valores", "validar", "final", "score", "insights")
COLS_PITCH = ["pt", "season", "ivb_mar", "hb_mar", "ivb_cdmx", "hb_cdmx", "stuff_mar_plus", "stuff_cdmx_plus"]
N_FIGURA = 40_000                         # pitcheos al azar que se dibujan en el mapa
CAL_OK = 0.15                             # una pendiente de calibracion a menos de 0.15 de 1 se llama "calibrada"

NOMBRE_PT = {"FF": "Recta de 4 costuras", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
             "CU": "Curva", "CH": "Cambio", "FS": "Split"}
ARTICULO_PT = {"FF": "la", "CU": "la"}    # los demas llevan "el"
ORDEN_PT = list(NOMBRE_PT)
FAMILIAS = {"FAM_RECTAS": ("FF", "SI"), "FAM_QUIEBRE": ("SL", "CU"), "FAM_CAMBIO": ("CH", "FS")}
# de que familia toma prestada la beta cada tipo (el cutter no esta en la media de ninguna: usa la de las rectas)
FAMILIA_DE = {"FF": "FAM_RECTAS", "SI": "FAM_RECTAS", "FC": "FAM_RECTAS", "SL": "FAM_QUIEBRE",
              "CU": "FAM_QUIEBRE", "CH": "FAM_CAMBIO", "FS": "FAM_CAMBIO"}
NOMBRE_FAM = {"FAM_RECTAS": "Rectas", "FAM_QUIEBRE": "Quiebres", "FAM_CAMBIO": "Cambios"}
TOLERANCIA = {"neutralidad": 1.0, "transporte": 0.5}      # puntos de Stuff+ por pitcheo y pulgadas (score.py)
MIN_CELDA_FISICA = MIN_CELDA_CF = None
try:                                      # las definiciones vigentes mandan; si no se pueden leer, las de arriba
    import physics as _physics
    FAMILIAS, MIN_CELDA_FISICA = dict(_physics.FAMILIAS), int(_physics.MIN_CELDA)
    FAMILIA_DE = dict(_physics.FAMILIA_DE)
except (Exception, SystemExit):           # noqa: BLE001
    pass
try:
    import score as _score
    TOLERANCIA, MIN_CELDA_CF = dict(_score.TOLERANCIA), int(_score.MIN_CELDA)
except (Exception, SystemExit):           # noqa: BLE001
    pass

ABREV_FUENTE = {"propia": "prop.", "propia_encogida": "prop.", "familia": "fam.", "liga": "liga",
                "mov_total": "mov.", "fisica": "fís.", "no_estimable": ""}
# columna `motivo` de physics.py: por que un tipo no usa su propia beta
MOTIVO = {"pocos_pitchers": "por pocos pitchers con el mismo pitcheo en aires distintos",
          "igual_a_familia": "porque no se distinguen de la de su familia",
          "no_estimable": "porque el quiebre medio es casi 0 o cambia de signo",
          "sin_error_estandar": "sin error estándar", "recortada": "recortadas al límite"}
# variable del modelo -> (familia, nombre en el documento). Una variable que no este aqui sale con su nombre crudo
VARIABLES = {
    "velocity": ("Física del pitcheo en la salida", "velocidad"), "ivb": ("Física del pitcheo en la salida", "IVB"),
    "hb_n": ("Física del pitcheo en la salida", "HB"), "spin_rate": ("Física del pitcheo en la salida", "spin"),
    "release_height": ("Física del pitcheo en la salida", "altura de salida"),
    "rel_side_n": ("Física del pitcheo en la salida", "lado de salida"),
    "extension": ("Física del pitcheo en la salida", "extensión"),
    "mov_total": ("Física del pitcheo en la salida", "movimiento total"),
    "mov_per_krpm": ("Física del pitcheo en la salida", "movimiento por 1,000 rpm"),
    "velo_diff_vs_fb": ("Relación con la recta primaria del pitcher", "diferencial de velocidad"),
    "ivb_diff_vs_fb": ("Relación con la recta primaria del pitcher", "diferencial de IVB"),
    "hb_diff_vs_fb": ("Relación con la recta primaria del pitcher", "diferencial de HB"),
    "tunnel_dist": ("Relación con la recta primaria del pitcher", "túnel en la salida: distancia entre puntos de salida medios"),
    "is_fastball": ("Relación con la recta primaria del pitcher", "es recta"),
    "pt": ("Tipo y mano", "tipo de pitcheo"), "lefty": ("Tipo y mano", "pitcher zurdo"),
    "plate_x_n": ("Ubicación y cuenta", "lado en el plato"), "plate_z": ("Ubicación y cuenta", "altura en el plato"),
    "zone_dist": ("Ubicación y cuenta", "distancia al centro de la zona"), "in_zone": ("Ubicación y cuenta", "en zona"),
    "balls": ("Ubicación y cuenta", "bolas"), "strikes": ("Ubicación y cuenta", "strikes"),
    "same_side": ("Ubicación y cuenta", "bateador del mismo lado"),
    "usage_pct": ("Repertorio y situación", "% de uso del pitcheo"),
    "n_pitch_types": ("Repertorio y situación", "tamaño del repertorio"),
    "stand_l": ("Repertorio y situación", "bateador zurdo"), "base_out": ("Repertorio y situación", "outs"),
    "vra": ("Ángulos de salida y de llegada", "vertical de salida"),
    "hra_n": ("Ángulos de salida y de llegada", "horizontal de salida"),
    "vaa": ("Ángulos de salida y de llegada", "vertical de llegada"),
    "haa_n": ("Ángulos de salida y de llegada", "horizontal de llegada"),
}
# lo que NO entra a Stuff+ a proposito (features.py lo explica): se imprime solo si de verdad no esta en feats_
FUERA_DE_STUFF = [
    (("air_density",), "la densidad del aire (identificaría el parque; la altitud entra por el transporte del "
                       "movimiento y por el valor de los batazos)"),
    (("eff_velo",), "la velocidad efectiva (depende del aire)"),
    (("usage_pct", "n_pitch_types"), "el % de uso y el tamaño del repertorio (decisiones del pitcher, no física)"),
    (("spin_axis_n", "axis_dev"), "el eje de giro (el sistema de rastreo lo infiere del movimiento: repite al IVB y al HB)"),
    (("vra", "hra_n"), "los ángulos de salida (con el punto de salida reconstruyen la ubicación)"),
]
FORMA_CORTA = {"ride": "Ride (IVB)", "hundimiento": "Hundimiento", "caida": "Caída",
               "corrida": "Corrida al lado del brazo (HB)", "barrido": "Barrido", "corte": "Corte",
               "velocidad": "Velocidad", "dif_velo": "Dif. de velocidad con la recta", "spin": "Spin"}
FORMA_FRASE = {"ride": "ride", "hundimiento": "hundimiento", "caida": "caída", "corrida": "corrida al lado del brazo",
               "barrido": "barrido", "corte": "corte", "velocidad": "velocidad",
               "dif_velo": "diferencial de velocidad con la recta", "spin": "spin"}
RASGO_CORTO = {"velo": "Velocidad (mph)", "dif_velo": "Dif. de velocidad con la recta (mph)", "spin": "Spin (rpm)",
               "ivb_cdmx": "IVB en el Harp Helú (pulg)", "hb_cdmx": "HB en el Harp Helú (pulg)",
               "ext": "Extensión (pies)", "rel_h": "Altura de salida (pies)", "rel_s": "Lado de salida (pies)"}
RASGO_DEC = {"spin": 0}
# la llave 'contacto_duro' es la de model.py; el objetivo es un TIPO de batazo (linea o elevado duro), no el contacto
# duro por velocidad de salida que este mismo documento reporta como tasa observada ("Contacto duro %")
OBJETIVOS = [("whiff", "P(swing fallado)"), ("called_strike", "P(strike cantado)"), ("swing", "P(swing)"),
             ("chase", "P(chase)"), ("gb", "P(rodado)"),
             ("contacto_duro", "P(línea o elevado duro)"), ("xrv", "xRV (carreras)")]
P_DERIVADAS = ("p_swing", "p_chase", "p_linea_o_elevado_duro")     # como las llama score.py si las escribe
MODELOS_VAL = [("nulo_global", "Nulo global"), ("nulo_cuenta", "Nulo por cuenta"),
               ("stuff", "Stuff+ (no ve la cuenta)"), ("stuff_cuenta", "Stuff+ con cuenta"),
               ("location", "Location+"), ("pitching", "Pitching+")]
CONJUNTO_H4 = {"tipo": "Solo el tipo de pitcheo", "+spin": "+ spin", "+velocidad": "+ velocidad",
               "+movimiento": "+ movimiento (IVB, HB, total, por rpm)", "forma_sin_spin": "Forma sin el spin",
               "+release": "+ punto de salida y extensión", "+diferenciales": "+ diferenciales con la recta"}
VEREDICTO = {"apoyada": "Apoyada", "contraria": "Contraria", "no concluyente": "No concluyente",
             "sin datos": "Sin datos"}

H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=NEGRO, spaceAfter=4,
                    spaceBefore=7, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=NEGRO, spaceAfter=3,
                    spaceBefore=6, keepWithNext=1)
H2T = ParagraphStyle("h2t", parent=H2, keepWithNext=0)
P = ParagraphStyle("p", fontName="Helvetica", fontSize=8.8, leading=11.1, textColor=NEGRO, spaceAfter=4, alignment=4)
NOTA = ParagraphStyle("n", parent=P, fontSize=7.5, leading=9.1, textColor=GRIS, spaceAfter=4)
CELDA = ParagraphStyle("c", parent=P, fontSize=7.8, leading=9.6, alignment=0, spaceAfter=0)
FORMULA = ParagraphStyle("f", parent=P, alignment=1, spaceAfter=4)

# Si el documento pasa de la meta de paginas se vuelve a armar mas compacto (main): letra, tablas y figuras un
# poco mas chicas, y las tablas ya pueden partirse entre paginas. (texto, tablas, figuras, saltos) Los dos ultimos
# niveles no achican mas la letra ni las figuras (quedarian ilegibles): solo encogen el hueco minimo que exige cada
# salto de pagina condicional, que es donde se pierden paginas.
NIVELES = [(1.0, 1.0, 1.0, 1.0), (.975, .98, .93, .98), (.955, .96, .88, .96), (.91, .92, .76, .92), (.88, .9, .7, .9),
           (.88, .9, .7, .6), (.88, .9, .7, .35)]
BASE = {e.name: (e.fontSize, e.leading) for e in (H1, H2, H2T, P, NOTA, CELDA, FORMULA)}
AJUSTE = {"tabla": 1.0, "figura": 1.0, "saltos": 1.0, "entera": True}


def compactar(nivel):
    """Aplica un nivel de NIVELES a los estilos base (0 = tamano normal)."""
    tx, tb, fg, sl = NIVELES[nivel]
    AJUSTE.update(tabla=tb, figura=fg, saltos=sl, entera=(nivel == 0))
    for est, k in ((H1, tx), (H2, tx), (H2T, tx), (P, tx), (NOTA, tx), (FORMULA, tx), (CELDA, tb)):
        est.fontSize, est.leading = BASE[est.name][0] * k, BASE[est.name][1] * k


AVISOS = []                               # bloques que fallaron de forma inesperada (se listan al final)
_YA_DICHO = set()                         # un aviso se imprime una sola vez aunque el documento se arme dos veces


class NoEstimable(Exception):
    """El bloque no se puede escribir con lo que hay en out/ (archivo ausente, vacio o de otra version)."""


# ── lectura a prueba de faltantes ───────────────────────────────────────────
def aviso(msg, grave=False):
    """Consola solo en ASCII: la de Windows puede fallar con acentos."""
    txt = ("AVISO GRAVE: " if grave else "AVISO: ") + str(msg)
    if txt not in _YA_DICHO:
        _YA_DICHO.add(txt)
        print(txt.encode("ascii", "replace").decode("ascii"), flush=True)


def leer_csv(nombre, **kw):
    try:
        return pd.read_csv(config.OUT / nombre, encoding="utf-8", **kw)
    except Exception:                      # noqa: BLE001  (falta, vacio o ilegible: tabla sin filas)
        return pd.DataFrame()


def leer_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:                      # noqa: BLE001
        return {}


def fin(*xs):
    """True si todos son numeros finitos (None, texto o NaN -> False)."""
    for x in xs:
        try:
            if isinstance(x, str) or not np.isfinite(float(x)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def verdadero(x):
    """Las banderas llegan como bool, 'True', 1 o NaN segun la version de pandas y si la columna trae vacios."""
    return str(x).strip().lower() in ("true", "1", "1.0")


def falso(x):
    """False explicito. Una bandera vacia (NaN) no es ni verdadero() ni falso(): es 'sin veredicto'."""
    return str(x).strip().lower() in ("false", "0", "0.0")


def pide(df, nombre, *cols):
    """Detiene el BLOQUE (no el documento) si la tabla no trae lo que se necesita."""
    if df is None or not len(df):
        raise NoEstimable(f"falta out/{nombre} o no tiene filas")
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise NoEstimable(f"out/{nombre} no trae: {', '.join(faltan)}")


def fila(df, **cond):
    """Primera fila que cumple las condiciones, como dict ({} si no hay tabla, columna o fila)."""
    if df is None or not len(df) or any(c not in df.columns for c in cond):
        return {}
    m = np.ones(len(df), dtype=bool)
    for c, v in cond.items():
        m &= (df[c].astype(str) == str(v)).to_numpy()
    return df[m].iloc[0].to_dict() if m.any() else {}


def par(d, clave):
    """[lo, hi] de un JSON -> (lo, hi) o (nan, nan)."""
    v = (d or {}).get(clave)
    return (v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) == 2 and fin(*v) else (np.nan, np.nan)


def seguro(fn, *a, defecto=None, nombre=None):
    """Una frase que no se puede construir no tumba el parrafo: se omite (o sale `defecto`)."""
    try:
        return fn(*a)
    except NoEstimable:
        return defecto
    except Exception as e:                 # noqa: BLE001
        AVISOS.append(nombre or fn.__name__)
        aviso(f"'{nombre or fn.__name__}' fallo de forma inesperada ({type(e).__name__}: {e})", grave=True)
        return defecto


def leer_feats():
    """Variables de cada sub-modelo, leidas del modelo AJUSTADO (feats_). Si el modelo no abre se usa la
    lista del codigo y el documento lo dice."""
    feats, origen = {}, {}
    for nm in ("stuff", "location", "pitching"):
        try:
            import joblib
            feats[nm], origen[nm] = [str(f) for f in joblib.load(config.MODELS / f"{nm}_fold0.joblib").feats_], "modelo"
        except Exception:                  # noqa: BLE001
            try:
                import features as F
                lista = {"stuff": F.STUFF_FEATS, "location": F.LOCATION_FEATS, "pitching": F.PITCHING_FEATS}[nm]
                feats[nm], origen[nm] = list(lista) + list(F.CATEGORICAL), "codigo"
            except Exception:              # noqa: BLE001
                feats[nm], origen[nm] = [], "sin datos"
    return feats, origen


def leer_pitcheos():
    """(muestra con las columnas que usa el documento, numero de pitcheos calificados, temporadas)."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(config.OUT / "scores_pitch.parquet")
        cols = [c for c in COLS_PITCH if c in pf.schema_arrow.names]
        n = int(pf.metadata.num_rows)
        sp = pf.read(columns=cols).to_pandas() if cols else pd.DataFrame()
    except Exception:                      # noqa: BLE001
        return pd.DataFrame(), 0, []
    temporadas = sorted(int(t) for t in pd.to_numeric(sp["season"], errors="coerce").dropna().unique()) \
        if "season" in sp else []
    if len(sp) > N_FIGURA:
        sp = sp.sample(N_FIGURA, random_state=config.RANDOM_STATE)
    return sp.reset_index(drop=True), n, temporadas


def filas_parquet(nombre):
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(config.OUT / nombre).metadata.num_rows)
    except Exception:                      # noqa: BLE001
        return None


def columnas_predicciones():
    """Encabezado del entregable por pitcheo ([] si no esta): el documento solo dice que trae lo que trae."""
    try:
        import gzip
        with gzip.open(config.OUT / "predicciones_por_pitcheo.csv.gz", "rt", encoding="utf-8") as f:
            return [c.strip() for c in f.readline().split(",") if c.strip()]
    except Exception:                      # noqa: BLE001
        return []


class Datos:
    """Todo lo que lee el documento. Lo que falte queda como tabla sin filas o dict vacio."""

    def __init__(self):
        O, M = config.OUT, config.MODELS
        self.info = leer_json(O / "run_info.json")
        self.el = leer_csv("altitude_elasticities.csv")
        self.alt = leer_json(O / "altitude_summary.json")
        self.parks = leer_csv("park_table.csv")
        self.adapt = leer_csv("adaptacion.csv")
        self.chequeo = leer_csv("chequeo_etiquetas.csv")
        self.res = leer_csv("val_resumen.csv")
        self.ph = leer_csv("val_pitcher_holdout.csv")
        self.sh = leer_csv("val_season_holdout.csv")
        self.pk = leer_csv("val_park_holdout.csv")
        self.cal = leer_csv("val_calibracion.csv")
        self.calr = leer_csv("val_calibracion_resumen.csv")
        self.h4 = leer_csv("val_h4_ablation.csv")
        self.imp = leer_csv("val_importancia_stuff.csv")
        self.cf = leer_csv("val_contrafactual.csv")
        self.stab = leer_json(O / "val_stability.json")
        self.limpieza = leer_json(O / "limpieza.json")
        self.tipo = leer_csv("altitude_by_pitch_type.csv")
        self.parque = leer_json(O / "altitude_park_effect.json")
        self.park_scores = leer_csv("park_scores.csv")
        self.jug = leer_csv("scores_pitcher.csv")
        self.ars = leer_csv("scores_arsenal.csv")
        self.folds = leer_csv("folds.csv", dtype=str)
        self.formas = leer_csv("insights_formas.csv")
        self.perfil = leer_csv("insights_perfil.csv")
        self.mix = leer_csv("insights_mix.csv")
        self.top = leer_csv("insights_top_cdmx.csv")
        self.ins = leer_json(O / "insights.json")
        self.val = leer_json(M / "outcome_values.json")
        self.vc = leer_json(O / "valores_carrera.json")
        self.iters = leer_json(M / "iters.json")
        self.feats, self.feats_origen = leer_feats()
        self.sp, self.n_pitcheos, self.temporadas = leer_pitcheos()
        self.n_features = filas_parquet("features.parquet")
        self.pred_cols = columnas_predicciones()
        # el dataset oficial solo trae la categoria de altitud: cada nivel de densidad es una categoria
        self.rho_c = self.alt.get("rho_cdmx", config.RHO_CDMX)
        self.rho_m = self.alt.get("rho_mar", config.RHO_MAR)
        self.k = self.alt.get("n_estadios")
        # huella y modo de la corrida
        pasos = self.info.get("pasos") or {}
        h = str(self.info.get("hash_codigo") or "")
        self.hash = h[:8]
        self.rapido = bool(self.info.get("rapido")) if "rapido" in self.info else bool(config.RAPIDO)
        self.rapido = self.rapido or any(bool((pasos.get(p) or {}).get("rapido")) for p in PASOS_DATOS)
        self.otra_version = bool(h) and any((pasos.get(p) or {}).get("hash_codigo") not in (None, h)
                                            for p in ("validar", "score"))

    # como se llama un nivel de densidad en el texto
    def nivel(self, n=1):
        return "categoría" if n == 1 else "categorías"

    def un_nivel(self):
        return "una categoría"

    def casa(self):
        return f"la categoría {config.HOME_ALTITUDE}"


# ── formato ─────────────────────────────────────────────────────────────────
def nf(v, d=2, s=False):
    """numero -> texto ('–' si no hay dato); s: con signo. Nunca '-0.00'."""
    if not fin(v):
        return RAYA
    v = float(v)
    if round(v, d) == 0:
        v = 0.0
    return f"{v:+,.{d}f}" if s else f"{v:,.{d}f}"


def ic(lo, hi, d=2, s=False):
    """'[lo, hi]'. Un extremo que no es 0 pero se redondearia a 0 lleva mas decimales: el lector debe poder ver
    de que lado de 0 queda el intervalo."""
    if not fin(lo, hi):
        return "[sin IC]"

    def extremo(v):
        k = d
        while k < d + 2 and float(v) != 0 and round(float(v), k) == 0:
            k += 1
        return nf(v, k, s) if round(float(v), k) != 0 or float(v) == 0 else f"{float(v):+.0e}"
    return f"[{extremo(lo)}, {extremo(hi)}]"


def vic(v, lo, hi, d=2, s=True):
    return f"{nf(v, d, s)} {ic(lo, hi, d, s)}"


def entero(v):
    return f"{int(round(float(v))):,}" if fin(v) else RAYA


def juntar(xs, y="y"):
    xs = [x for x in xs if x]
    return xs[0] if len(xs) == 1 else (", ".join(xs[:-1]) + f" {y} " + xs[-1]) if xs else ""


def nombre_pt(pt, corto=False, articulo=False):
    """Nombre del tipo (o familia) para tablas; corto: en minusculas para una frase; articulo: 'el slider'."""
    pt = str(pt)
    if pt in NOMBRE_FAM:
        n = NOMBRE_FAM[pt] + (" (" + "+".join(FAMILIAS.get(pt, ())) + ")" if FAMILIAS.get(pt) else "")
        return ("las " if pt == "FAM_RECTAS" else "los ") + n[0].lower() + n[1:] if articulo else n
    if pt == "TODOS":
        return "todos los pitcheos" if corto or articulo else "Todos los pitcheos"
    if pt == "REPERTORIO":
        return "el repertorio completo" if articulo else "Repertorio completo"
    n = NOMBRE_PT.get(pt, pt)
    if articulo:
        return f"{ARTICULO_PT.get(pt, 'el')} {n.lower() if pt in NOMBRE_PT else n}"
    return n.lower() if corto and pt in NOMBRE_PT else n


def minuscula(s):
    """Primera letra en minuscula para meter un rotulo dentro de una frase; las siglas (IVB, HB) y los nombres de
    los scores (Stuff+, Pitching+) se respetan."""
    s = str(s)
    return s if s[:2].isupper() or s.split(" ")[0].rstrip(",:").endswith("+") else s[:1].lower() + s[1:]


def plural(n, uno, varios):
    return f"{entero(n)} {uno if fin(n) and int(round(float(n))) == 1 else varios}"


def rango(a, b, d=1):
    """'91.7–94.1'; con negativos, '-3.9 a -2.4' (una raya junto a un signo menos se lee mal)."""
    if not fin(a, b):
        return RAYA
    return f"{nf(a, d)} a {nf(b, d)}" if min(float(a), float(b)) < 0 else f"{nf(a, d)}{RAYA}{nf(b, d)}"


def orient(fiable):
    return "" if verdadero(fiable) else ", orientativo"


def envolvente(*pares):
    """El mas ancho de varios intervalos: minimo de los inferiores y maximo de los superiores."""
    ps = [(float(lo), float(hi)) for lo, hi in pares if fin(lo, hi)]
    return (min(p[0] for p in ps), max(p[1] for p in ps)) if ps else (np.nan, np.nan)


def lado(lo, hi, ref=0.0):
    """+1 si el intervalo queda arriba de ref, -1 si queda abajo, 0 si lo contiene, None si no hay intervalo."""
    if not fin(lo, hi):
        return None
    return 1 if lo > ref else -1 if hi < ref else 0


def color_lado(lo, hi):
    return {1: VERDE, -1: ROJO}.get(lado(lo, hi))


# ── FRASES: toda afirmacion con direccion o juicio sale de una de estas funciones ──
def frase_aire(D):
    r = D.rho_c / D.rho_m if fin(D.rho_c, D.rho_m) and D.rho_m else np.nan
    if not fin(r):
        return "No hay densidades de referencia para comparar el Harp Helú con el nivel del mar."
    cifra = f"densidad relativa {nf(D.rho_c, 3)} contra {nf(D.rho_m, 3)}"
    if r < 1:
        return (f"En el Harp Helú (2,236 m) el aire tiene {100 * (1 - r):.0f}% menos densidad que en los parques de la "
                f"LMB al nivel del mar ({cifra}).")
    if r > 1:
        return f"El aire de referencia del Harp Helú es {100 * (r - 1):.0f}% más denso que el del nivel del mar ({cifra})."
    return f"Las dos densidades de referencia son iguales ({cifra})."


def frase_movimiento(D):
    """Resumen de H1: decide el intervalo de la beta del IVB de las rectas (fila de familia: no depende de la
    etiqueta automatica del pitcheo); si no existe, la de la recta de 4 costuras."""
    r = fila(D.el, var="ivb", pt="FAM_RECTAS") or fila(D.el, var="ivb", pt="FF")
    b, lo, hi = r.get("beta"), r.get("ci_lo"), r.get("ci_hi")
    if not r or not verdadero(r.get("estimable", True)) or lado(lo, hi) is None or not fin(b):
        return "No se pudo estimar cómo cambia el movimiento con la densidad del aire."
    cifra = f"β del IVB de las rectas = {nf(b)}, IC 95% {ic(lo, hi)}{orient(r.get('ic_fiable'))}"
    ff = fila(D.el, var="ivb", pt="FF")
    nivel = (f": con esa elasticidad una recta de 4 costuras pasa de {nf(ff.get('media_mar'), 1)} a "
             f"{nf(ff.get('media_cdmx'), 1)} pulgadas de IVB" if fin(ff.get("media_mar"), ff.get("media_cdmx")) else "")
    s = lado(lo, hi)
    if s > 0:
        return f"Comparando a cada pitcher contra sí mismo, el quiebre baja con la densidad del aire ({cifra}){nivel}."
    if s < 0:
        return f"Comparando a cada pitcher contra sí mismo, el quiebre sube cuando baja la densidad ({cifra})."
    return f"Comparando a cada pitcher contra sí mismo, no se distingue un cambio del quiebre con la densidad ({cifra})."


def frase_costo(p, corta=False):
    """Costo ABSOLUTO de lanzar en el Harp Helu: decide ic95_recta, el intervalo que incluye la incertidumbre de
    la recta del valor de los batazos (el de remuestreo de pitchers es mucho mas angosto y no la incluye)."""
    costo, (lo, hi) = (p or {}).get("costo_cdmx_vs_mar_rv100"), par(p, "ic95_recta")
    if not fin(costo) or lado(lo, hi) is None:
        return "El costo absoluto de lanzar en el Harp Helú no es estimable con estos datos."
    cifra = (f"{nf(costo, 2, True)} carreras por 100 pitcheos, IC 95% {ic(lo, hi, 2, True)}"
             + ("" if verdadero(p.get("recta_ic_fiable")) else
                f", orientativo: la recta de valor de los batazos sale de {entero(p.get('recta_n_puntos'))} "
                "puntos de densidad"))
    s = lado(lo, hi)
    if s > 0:
        txt = f"El mismo pitcheo permite más carreras en el Harp Helú que al nivel del mar ({cifra})."
    elif s < 0:
        txt = f"El mismo pitcheo permite menos carreras en el Harp Helú que al nivel del mar ({cifra})."
    else:
        txt = f"No se distingue un costo de lanzar en el Harp Helú frente al nivel del mar ({cifra})."
    if corta:
        return txt
    mov, (mlo, mhi) = p.get("por_movimiento_rv100"), par(p, "por_movimiento_ic95")
    val, (vlo, vhi) = p.get("por_valor_batazos_rv100"), par(p, "por_valor_batazos_ic95_recta")
    if fin(mov, val):
        def parte(v, a, b, que):                             # cada parte, con lo que dice su propio intervalo
            return f"{vic(v, a, b)} {que}" + (", intervalo que incluye 0" if lado(a, b) == 0 else "")
        txt += (" Descomposición: "
                + parte(mov, mlo, mhi, "por el cambio de movimiento, con el valor de los batazos fijo (IC de "
                                       "remuestreo de pitchers") + ") y "
                + parte(val, vlo, vhi, "por el cambio de valor de los batazos con la densidad (IC de la recta") + ").")
    obs, (olo, ohi) = p.get("observado_casa_menos_mar_rv100"), par(p, "observado_ic95")
    if fin(obs):
        # lo observado solo cubre las densidades que se midieron (casa y banda mar): se compara con el costo modelado
        # entre ESAS dos, no con el de los dos aires de referencia
        eb, ra, rb = p.get("costo_entre_bandas_rv100"), p.get("rho_casa_obs"), p.get("rho_mar_obs")
        igual = (f" (densidades {nf(ra, 3)} y {nf(rb, 3)}; entre esas dos el modelo da {nf(eb, 2, True)})"
                 if fin(eb, ra, rb) else "")
        txt += (f" Referencia observada: los pitcheos lanzados en casa permitieron {vic(obs, olo, ohi)} carreras por "
                f"100 frente a los de la banda mar{igual}; esa diferencia mezcla parque, equipos y bateadores y no "
                "prueba nada por sí sola.")
    p10, p90 = par(p, "rango_pitchers_p10_p90")
    if fin(p10, p90):
        txt += f" Entre pitchers el costo va de {nf(p10, 2, True)} a {nf(p90, 2, True)} (percentiles 10 y 90)."
    return txt


def frase_vs_nulo(nombre, r, contra, corta=False):
    """Un modelo contra un nulo, en log-loss fuera de muestra: decide el IC 95% de la mejora (remuestreo de
    pitchers) que escribio model.py. corta: la cifra sin repetir 'de log-loss, IC 95%' (ya lo dice la tabla)."""
    v, lo, hi = (r.get(f"mejora_vs_{contra}_pct"), r.get(f"mejora_vs_{contra}_ci_lo"),
                 r.get(f"mejora_vs_{contra}_ci_hi"))
    nulo = "nulo global" if contra == "global" else "nulo por cuenta"
    nombre = (nombre + " ") if nombre else ""
    s = lado(lo, hi)
    if s is None or not fin(v):
        return f"{nombre}no tiene datos contra el {nulo}"
    d = 2 if abs(v) < 10 else 1
    cifra = (f"{nf(v, d, True)}% {ic(lo, hi, d, True)}" if corta else
             f"{nf(v, d, True)}% de log-loss, IC 95% {ic(lo, hi, d, True)}")
    if s > 0:
        return f"{nombre}supera al {nulo} ({cifra})"
    if s < 0:
        return f"{nombre}queda por debajo del {nulo} ({cifra})"
    return f"{nombre}no se distingue del {nulo} ({cifra})"


def frase_brier(r):
    """Stuff+ contra el nulo por cuenta en Brier de swing fallado (negativo = menos error)."""
    v, lo, hi = (r.get("delta_brier_vs_cuenta"), r.get("delta_brier_vs_cuenta_ci_lo"),
                 r.get("delta_brier_vs_cuenta_ci_hi"))
    s = lado(lo, hi)
    if s is None or not fin(v):
        return ""
    cifra = f"diferencia de Brier × 1,000 = {vic(1e3 * v, 1e3 * lo, 1e3 * hi)}; negativo = menos error"
    if s < 0:
        return f" En el error cuadrático de P(swing fallado), en cambio, Stuff+ sí mejora a ese nulo ({cifra})."
    if s > 0:
        return f" En el error cuadrático de P(swing fallado) Stuff+ también es peor que ese nulo ({cifra})."
    return f" En el error cuadrático de P(swing fallado) Stuff+ no se distingue de ese nulo ({cifra})."


def frase_sesgo(v, lo, hi):
    """Sesgo = carreras predichas menos reales por 100 pitcheos en un parque que el modelo no vio. La frase
    continua a un sujeto: 'Pitching+ ' + frase_sesgo(...)."""
    s = lado(lo, hi)
    if s is None or not fin(v):
        return "no tiene sesgo estimable"
    cifra = vic(v, lo, hi) + " carreras por 100"
    if s > 0:
        return f"sobreestima las carreras ({cifra})"
    if s < 0:
        return f"subestima las carreras ({cifra})"
    return f"no muestra un sesgo distinguible de 0 ({cifra})"


def clase_calibracion(pend, lo=None, hi=None):
    """Regla escrita sobre la pendiente de observado contra predicho y su IC 95% (remuestreo de pitchers):
      calibrada       todo el intervalo esta a menos de CAL_OK de 1
      exagera         el intervalo queda bajo 1 (y no cabe en esa tolerancia): el modelo exagera las diferencias
      comprime        el intervalo queda sobre 1: las comprime
      no concluyente  el intervalo contiene 1 y es mas ancho que la tolerancia, o el archivo no trae intervalo"""
    if not fin(pend):
        return "sin datos"
    if not fin(lo, hi):
        return "no concluyente"
    if lo >= 1 - CAL_OK and hi <= 1 + CAL_OK:
        return "calibrada"
    return "exagera" if hi < 1 else "comprime" if lo > 1 else "no concluyente"


def calibracion_de(r):
    """clase_calibracion de una fila de val_calibracion_resumen.csv."""
    return clase_calibracion(r.get("pendiente"), r.get("pendiente_ci_lo"), r.get("pendiente_ci_hi"))


def frase_predictiva(nombre, stab, clave):
    """¿El score de una temporada predice las carreras de la siguiente mejor que las propias carreras?
    Decide el IC de la DIFERENCIA de correlaciones (remuestreo de pitchers), no la comparacion de dos numeros."""
    r_m, r_rv = stab.get(f"predice_rv_t1__con_{clave}_plus_t"), stab.get("predice_rv_t1__con_rv_observado_t")
    dif, (lo, hi) = stab.get(f"dif_{clave}_vs_rv"), par(stab, f"dif_{clave}_vs_rv_ci")
    s = lado(lo, hi)
    if s is None or not fin(r_m, r_rv, dif):
        return f"{nombre}: no hay suficientes pitchers con dos temporadas para medir su poder predictivo."
    cifra = f"r = {nf(r_m)} contra {nf(r_rv)}; diferencia {vic(dif, lo, hi)}"
    base = f"{nombre} de una temporada"
    if s > 0:
        return f"{base} predice las carreras de la siguiente mejor que las propias carreras permitidas ({cifra})."
    if s < 0:
        return f"{base} predice las carreras de la siguiente peor que las propias carreras permitidas ({cifra})."
    return (f"{base} y las propias carreras permitidas no se distinguen al predecir las carreras de la siguiente "
            f"({cifra}).")


def clase_tipo(r):
    """(clase, lo, hi) del cambio RELATIVO de un tipo de pitcheo; misma regla que el dashboard:
      gana / pierde            solo si score.py lo marco `significativo` (los dos intervalos excluyen 0 y hay
                               pitchers suficientes)
      sin diferencia medible   si el intervalo mas ancho contiene 0
      evidencia insuficiente   si excluye 0 pero faltan pitchers o un intervalo"""
    d = r.get("delta")
    lo, hi = envolvente((r.get("delta_lo"), r.get("delta_hi")), (r.get("delta_lo_folds"), r.get("delta_hi_folds")))
    if not fin(d):
        return "sin datos", lo, hi
    if fin(lo, hi) and verdadero(r.get("significativo")):
        return ("gana" if d > 0 else "pierde"), lo, hi
    if fin(lo, hi) and lo <= 0 <= hi:
        return "sin diferencia medible", lo, hi
    return "evidencia insuficiente", lo, hi


def frase_tipos(tipo, cifras=True):
    """Que tipos de pitcheo ganan o pierden valor RELATIVO en el Harp Helu, segun clase_tipo."""
    pide(tipo, "altitude_by_pitch_type.csv", "pt", "delta")
    g, p, resto = [], [], []
    for _, r in tipo.iterrows():
        clase, lo, hi = clase_tipo(r)
        nom = nombre_pt(r["pt"], articulo=True)
        con = nom + (f" ({vic(r['delta'], lo, hi, 1)})" if cifras else "")
        if clase == "gana":
            g.append(con)
        elif clase == "pierde":
            p.append(con)
        else:
            resto.append(nom)
    if not g and not p:
        return ("Frente a los demás pitcheos, ningún tipo cambia de valor relativo en el Harp Helú de forma medible "
                "con los dos intervalos (pitchers y modelos).")
    partes = []
    if g:
        partes.append(("ganan" if len(g) > 1 else "gana") + " valor " + juntar(g))
    if p:
        partes.append(("lo pierden " if len(p) > 1 else "lo pierde ") if g else
                      ("pierden valor " if len(p) > 1 else "pierde valor ") + "")
        partes[-1] += juntar(p)
    txt = "Frente a los demás pitcheos, en el Harp Helú " + juntar(partes) + " (puntos por pitcheo, IC 95% más ancho)"
    if resto:
        txt += f"; para {juntar(resto)} no hay diferencia medible o la evidencia es insuficiente"
    return txt + "."


def frase_aceptacion(bloque, v, lo, hi, dentro=None, corta=False):
    """Pruebas de neutralidad y de transporte (ADENDA 2.d): pasa si |diferencia| < tolerancia. El intervalo
    distingue 'equivalente dentro de la tolerancia' de 'pasa por la estimacion puntual'."""
    tol = TOLERANCIA.get(bloque)
    if not fin(v) or tol is None:
        return "sin datos"
    uni = "punto" if bloque == "neutralidad" else "pulg"
    if abs(v) >= tol:
        return "fuera de tolerancia" if corta else f"fuera de la tolerancia de {tol:g} {uni}"
    if not fin(lo, hi):
        return "dentro de tolerancia (sin IC)" if corta else f"dentro de la tolerancia de {tol:g} {uni} (sin intervalo)"
    if dentro is None or not str(dentro).strip() or str(dentro).lower() == "nan":
        dentro = -tol < lo and hi < tol
    if verdadero(dentro):
        return "equivalente" if corta else f"equivalente dentro de la tolerancia de {tol:g} {uni}"
    hasta = max(abs(lo), abs(hi))
    if corta:
        return f"pasa; IC hasta {nf(hasta)}"
    return (f"dentro de la tolerancia de {tol:g} {uni} por la estimación puntual; el intervalo no descarta "
            f"diferencias de hasta {nf(hasta)}")


def frase_resultado(lo, hi, corta=False):
    """Bloque 'resultados' del contrafactual: valor = cambio predicho menos cambio observado. Los dos cambios
    parten del mismo nivel del mar, asi que el valor es tambien el nivel predicho EN ALTITUD menos el observado
    ahi, y asi se dice: 'un cambio mayor / menor' se leia al reves cuando los dos cambios son negativos."""
    s = lado(lo, hi)
    if s is None:
        return "sin datos"
    if s == 0:
        return "no se distinguen" if corta else "lo predicho y lo observado no se distinguen"
    if corta:
        return "predice más alto" if s > 0 else "predice más bajo"
    return ("en altitud el modelo predice un valor más alto que el observado" if s > 0 else
            "en altitud el modelo predice un valor más bajo que el observado")


def lectura_resultado(r):
    """(pasa, lectura corta, color) de una fila del bloque 'resultados', con la regla de score.veredicto_fila
    (columnas pasa y con_potencia). pasa: False si el IC de predicho menos observado excluye 0; True si lo
    contiene Y la prueba podia fallar; None = sin veredicto (sin potencia: tampoco habria rechazado la prediccion
    de que el aire no mueve el pitcheo). Sin veredicto no hay color."""
    lo, hi = r.get("ic_lo"), r.get("ic_hi")
    if falso(r.get("pasa")):
        return False, (frase_resultado(lo, hi, corta=True) if lado(lo, hi) else "no pasa"), ROJO
    if verdadero(r.get("pasa")):
        if not verdadero(r.get("con_potencia")):             # archivo anterior a con_potencia: no dice si podia fallar
            return None, "sin veredicto", None
        return True, "no se distinguen", VERDE
    if falso(r.get("con_potencia")):
        return None, "sin potencia", GRIS_CLARO
    return None, ("sin veredicto" if lado(lo, hi) is not None else "sin datos"), None


def frase_placebo(nombre, r, unidad, d, corta=False):
    """Placebo: variable que el aire no deberia mover. Una beta cuyo intervalo excluye 0 es un sesgo entre
    parques. Lo contrario NO es un aprobado cuando el intervalo es orientativo (pocos niveles de densidad): ahi se
    da el tamano del sesgo que ese intervalo todavia permite (delta_lo / delta_hi, en las unidades de la variable).
    corta: la misma regla sin repetir las cifras (para la tabla de evidencia; las cifras estan en la seccion 2)."""
    b, lo, hi = r.get("beta"), r.get("ci_lo"), r.get("ci_hi")
    delta, dlo, dhi = r.get("delta_cdmx_vs_mar"), r.get("delta_lo"), r.get("delta_hi")
    if not fin(b, lo, hi):
        return f"{nombre}: no estimable"
    cifra = f"β = {nf(b, 3)} {ic(lo, hi, 3)}{orient(r.get('ic_fiable'))}"
    if fin(delta):
        cifra += f"; {nf(delta, d, True)} {ic(dlo, dhi, d, True) + ' ' if fin(dlo, dhi) else ''}{unidad} en el Harp Helú"
    if lado(lo, hi) != 0:
        return (f"{nombre}: su β no es cero ({cifra})" if corta else
                f"{nombre}: su β no es cero ({cifra}): hay una diferencia de medición o de uso entre parques de ese tamaño")
    if verdadero(r.get("ic_fiable")) or not fin(dlo, dhi):
        return f"{nombre}: no se distingue de cero" + ("" if corta else f" ({cifra})")
    hasta = f"{nf(max(abs(dlo), abs(dhi)), d)} {unidad}"
    return (f"{nombre}: no se descarta un sesgo de hasta {hasta}" if corta else
            f"{nombre}: el intervalo incluye 0 pero no descarta un sesgo de hasta {hasta} ({cifra})")


def frase_beta_vs_1(lo, hi):
    s = lado(lo, hi, 1.0)
    if s is None:
        return "sin intervalo para compararla con 1"
    if s > 0:
        return "su intervalo queda por encima de 1: el quiebre cae más que en proporción a la densidad"
    if s < 0:
        return "su intervalo queda por debajo de 1: el quiebre cae menos que en proporción a la densidad"
    return "su intervalo incluye 1: no se distingue de la proporcionalidad"


def frase_cambia(r, d=2):
    """Fila de adaptacion.csv -> 'sube', 'baja' o 'sin cambio medible' segun su IC."""
    s = lado(r.get("ic_lo"), r.get("ic_hi"))
    if s is None:
        return "sin datos"
    return "sin cambio medible" if s == 0 else ("sube" if s > 0 else "baja")


def frase_mezcla(mez):
    if not len(mez):
        return ("Mezcla de pitcheos: no estimable con estos datos (no hay suficientes pitchers con pitcheos en las "
                "dos bandas).")
    mas, menos = [], []
    for _, r in mez.iterrows():
        s = lado(r.get("ic_lo"), r.get("ic_hi"))
        txt = f"{nombre_pt(r['variable'], True)} ({vic(r['diferencia'], r.get('ic_lo'), r.get('ic_hi'), 1)})"
        if s == 1:
            mas.append(txt)
        elif s == -1:
            menos.append(txt)
    n = entero(pd.to_numeric(mez["pitchers"], errors="coerce").max()) if "pitchers" in mez else RAYA
    base = f"Mezcla de pitcheos (puntos porcentuales de uso, banda alta menos banda mar; {n} pitchers): "
    if not mas and not menos:
        return base + "ningún tipo cambia su uso de forma medible."
    partes = ([f"se usa más {juntar(mas)}"] if mas else []) + ([f"se usa menos {juntar(menos)}"] if menos else [])
    return base + juntar(partes) + "; el resto, sin cambio medible."


def frase_etiqueta(et):
    """Si la etiqueta automatica del pitcheo fuera estable con la altitud, estas diferencias serian ~0."""
    if not len(et):
        return "Estabilidad de la etiqueta del pitcheo: no estimable con estos datos."
    partes, migra = [], False
    for _, r in et.iterrows():
        s, cifra = lado(r.get("ic_lo"), r.get("ic_hi")), vic(r["diferencia"], r.get("ic_lo"), r.get("ic_hi"), 1)
        que = {"FF dentro de FF+SI": "la proporción de rectas de 4 costuras dentro de FF+SI",
               "Velocidad SL menos FF": "la velocidad del slider menos la de la recta"}.get(str(r["variable"]),
                                                                                         str(r["variable"]))
        uni = str(r.get("unidad", "")).replace("puntos %", "puntos")
        if s is None:
            continue
        if s == 0:
            partes.append(f"{que} no cambia de forma medible ({cifra} {uni})")
        else:
            migra = True
            partes.append(f"{que} {'sube' if s > 0 else 'baja'} en altitud ({cifra} {uni}; {entero(r.get('pitchers'))} "
                          "pitchers)")
    # la conclusion se dice una vez, despues de todas las cifras (antes se repetia tras cada una)
    return ("Estabilidad de la etiqueta automática (AutoPitchType): " + juntar(partes, "y")
            + (": la etiqueta migra con el aire." if migra else ".")
            if partes else "Estabilidad de la etiqueta del pitcheo: no estimable con estos datos.")


def veredicto_h1(D):
    """H1 con regla escrita: decide la elasticidad del MOVIMIENTO TOTAL de todos los pitcheos (fila TODOS /
    mov_total de altitude_elasticities.csv): apoyada si es positiva, su intervalo excluye 0 y physics.py la marco
    significativa (pitchers suficientes); contraria si es negativa y lo excluye; no concluyente si lo contiene."""
    pide(D.el, "altitude_elasticities.csv", "var", "pt", "beta", "ci_lo", "ci_hi")
    r = fila(D.el, var="mov_total", pt="TODOS")
    b, lo, hi = r.get("beta"), r.get("ci_lo"), r.get("ci_hi")
    s = lado(lo, hi) if verdadero(r.get("estimable", True)) and fin(b) else None
    sig = verdadero(r.get("significativo", True))
    if s is None:
        return "sin datos", "No hay elasticidad estimable del movimiento total."
    ver = "apoyada" if s > 0 and sig else "contraria" if s < 0 and sig else "no concluyente"
    sentido = {1: "el movimiento baja cuando baja la densidad", -1: "el movimiento sube cuando baja la densidad",
               0: "no se distingue un cambio del movimiento con la densidad"}[s]
    txt = ("" if verdadero(r.get("ic_fiable")) else "Con intervalos orientativos (pocos niveles de densidad). ")
    txt += (f"Movimiento total, todos los pitcheos: β = {vic(b, lo, hi, 2, False)} "
            f"({entero(r.get('pitchers'))} pitchers): {sentido}"
            + ("" if sig or s == 0 else ", pero con menos pitchers de los que pide la regla") + ".")
    extra = []
    for var, nom in (("ivb", "IVB"), ("hb_n", "HB")):
        f = fila(D.el, var=var, pt="FAM_RECTAS")
        if verdadero(f.get("estimable")) and fin(f.get("beta"), f.get("ci_lo"), f.get("ci_hi")):
            extra.append(f"{nom} {vic(f['beta'], f['ci_lo'], f['ci_hi'], 2, False)}")
    if extra:
        txt += " Rectas (FF+SI): β de " + juntar(extra) + "."
    txt += (" Los placebos están en la sección 2. Se mide el quiebre total inducido: no se separa la fuerza de Magnus "
            "de otros efectos del aire.")
    return ver, txt


def veredicto_h4(ab):
    """H4 ('el spin solo no basta') con regla escrita sobre val_h4_ablation.csv (AUC de swing fallado dado swing).
    Decide UNA fila: el conjunto completo ('+diferenciales') contra tipo + spin (delta_auc_vs_spin y su IC):
    apoyada si ese intervalo queda sobre 0; contraria si no queda sobre 0 y agregar el spin al tipo si sube el AUC
    (el spin solo bastaria); no concluyente en otro caso. Los pasos uno a uno de la tabla dependen del orden en
    que entran las variables (lo que aporta el movimiento cambia si entra antes o despues de la velocidad): se
    describen contra la misma base, tipo + spin, y no deciden. Un archivo sin esas columnas da 'sin datos'."""
    pide(ab, "val_h4_ablation.csv", "conjunto", "auc")
    paso = {str(r["conjunto"]): r for _, r in ab.iterrows()}
    cambia = {1: "sube el AUC", -1: "baja el AUC", 0: "no cambia el AUC de forma medible"}

    def delta(k, sufijo=""):
        """(lado del intervalo, 'valor [IC]') de una fila; sufijo '_vs_spin' = contra tipo + spin."""
        r = paso.get(k)
        v, lo, hi = ((r.get(f"delta_auc{sufijo}{c}") for c in ("", "_lo", "_hi")) if r is not None else (None,) * 3)
        return (lado(lo, hi), vic(v, lo, hi, 3)) if fin(v) else (None, None)
    s_todo, c_todo = delta("+diferenciales", "_vs_spin")
    s_spin, c_spin = delta("+spin")
    if s_todo is None:
        return "sin datos", "La ablación no trae el modelo completo contra el de tipo + spin con su intervalo."
    ver = "apoyada" if s_todo == 1 else "contraria" if s_spin == 1 else "no concluyente"
    n = paso["+diferenciales"].get("n_vars")
    txt = ("AUC de swing fallado dado swing (tabla de la sección 5). El modelo completo"
           + (f", con {entero(n)} variables," if fin(n) else "")
           + f" contra el de tipo + spin {cambia[s_todo]} ({c_todo}). ")
    if s_spin is not None:
        txt += f"Agregar el spin al tipo de pitcheo {cambia[s_spin]} ({c_spin}). "
    nom = {"+velocidad": "con la velocidad", "+movimiento": "con velocidad y movimiento",
           "+release": "con el punto de salida además"}
    medio = [f"{nom[k]} {c}" for k, (s, c) in ((k, delta(k, "_vs_spin")) for k in nom) if s is not None]
    if medio:
        txt += "Conjuntos intermedios, también contra tipo + spin: " + "; ".join(medio) + ". "
    s_f, c_f = delta("forma_sin_spin")
    if s_f is not None:
        txt += f"Quitar el spin de un conjunto que ya tiene velocidad y movimiento {cambia[s_f]} ({c_f})."
    return ver, txt.strip()


def veredicto_coherente(ver, s, apoya, nombre):
    """`ver` viene de insights.json. Si afirma una direccion ('apoyada' / 'contraria') que el intervalo impreso a su
    lado no respalda (s = lado() de ese intervalo; `apoya` = el lado que apoya la hipotesis), se rebaja a
    'no concluyente' y se avisa: el documento nunca imprime un veredicto que su propia cifra contradice."""
    exige = {"apoyada": apoya, "contraria": -apoya}.get(ver)
    if exige is not None and s != exige:
        aviso(f"{nombre}: insights.json dice '{ver}' pero el intervalo que se imprime no lo respalda; el documento "
              "pone 'no concluyente'")
        return "no concluyente"
    return ver


def contraste_h2(h2):
    """(llave, contraste) del contraste de insights.json que decide H2. insights.py lo dice en `decide`
    ('contraste_liga': cambio y split contra todos los pitcheos, la frase del reto); un archivo de la version
    anterior no trae esa llave y decidia con 'contraste' (contra slider y curva)."""
    llave = str(h2.get("decide") or "contraste")
    return llave, (h2.get(llave) or {})


def nombres_pt(pts):
    """['CH', 'FS'] -> 'cambio y split'."""
    return juntar([nombre_pt(p, True) for p in pts or []])


def regla_h2(h2):
    llave, c = contraste_h2(h2)
    a = nombres_pt(c.get("a")) or "cambio y split"
    contra = ("frente a todos los pitcheos de la liga" if llave == "contraste_liga" else
              f"menos el de {nombres_pt(c.get('b')) or 'slider y curva'}")
    return (f"Regla: apoyada si el cambio relativo de {a} {contra} es positivo y sus dos intervalos excluyen 0; "
            "contraria si es negativo y lo excluyen.")


def texto_h2(h2):
    """El veredicto viene de insights.json y se imprime con la cifra del contraste que lo decide (contraste_h2);
    apoyada pide que el intervalo mas ancho (el de pitchers y el de modelos) quede sobre 0, contraria que quede
    bajo 0 (veredicto_coherente). Los otros contrastes se leen con la misma regla y no cambian el veredicto."""
    def envuelve(m):
        return envolvente((m.get("delta_lo"), m.get("delta_hi")), (m.get("delta_lo_folds"), m.get("delta_hi_folds")))
    decide, c = contraste_h2(h2)
    ver = str(h2.get("veredicto") or "sin datos")
    lo, hi = envuelve(c)
    ver = veredicto_coherente(ver, lado(lo, hi) if fin(c.get("delta")) else None, 1, "H2")
    a = "+".join(c.get("a", []))
    if not fin(c.get("delta")):
        return ver, "No hay contraste estimable para el cambio y el split."
    contra = ("el de todos los pitcheos de la liga, que promedia 0" if decide == "contraste_liga" else
              "el de " + "+".join(c.get("b", [])))
    txt = (f"Cambio relativo de {a} menos {contra}: {vic(c['delta'], lo, hi, 1)} puntos por pitcheo (IC 95% más "
           f"ancho de pitchers y modelos; {entero(c.get('pitchers_a'))} pitchers).")
    otros = []
    for llave, nom in (("contraste_liga", "toda la liga"), ("contraste", "los quiebres"),
                       ("contraste_rectas", "las rectas")):
        m = h2.get(llave) or {}
        if llave == decide or not fin(m.get("delta")):
            continue
        mlo, mhi = envuelve(m)
        s = lado(mlo, mhi)
        if s is None:
            lectura = ""
        elif s == 0:
            lectura = ", sin diferencia medible"
        elif verdadero(m.get("delta_sig")):
            lectura = f", a favor de {a}" if s > 0 else f", en contra de {a}"
        else:
            lectura = ", evidencia insuficiente"
        otros.append(f"contra {nom} ({'+'.join(m.get('b', []))}) {vic(m['delta'], mlo, mhi, 1)}{lectura}")
    if otros:
        txt += " No entran al veredicto: " + "; ".join(otros) + "."
    return ver, txt


def texto_h3(h3):
    """Veredicto de insights.json. Decide SOLO la prueba dentro del mismo pitcher: sin ella el veredicto es 'sin
    datos' (tambien con un insights.json anterior, que ahi decia 'no concluyente'). El valor de los batazos se
    reporta aparte: su direccion solo se afirma si su propio veredicto (que pide la recta de valor contra
    densidad) la respalda."""
    ver = str(h3.get("veredicto") or "sin datos")
    p, b = h3.get("pitcher") or {}, h3.get("batazos") or {}
    con_prueba = fin(p.get("pendiente"))
    # apoyada pide una pendiente con intervalo bajo 0 (el de rodados sufre menos); contraria, sobre 0
    ver = veredicto_coherente(ver, lado(p.get("pendiente_lo"), p.get("pendiente_hi")) if con_prueba else None, -1, "H3")
    if con_prueba:
        s = lado(p.get("pendiente_lo"), p.get("pendiente_hi"))
        txt = (f"Dentro del mismo pitcher ({plural(p.get('pitcher_temporadas'), 'pitcher-temporada', 'pitcher-temporadas')}, "
               f"{entero(p.get('pitchers'))} pitchers con {entero(p.get('min_pitcheos_por_banda'))}+ pitcheos en cada "
               f"banda): por cada 10 puntos más de % de rodados, la diferencia de run value entre banda alta y banda mar cambia "
               f"{vic(p['pendiente'], p.get('pendiente_lo'), p.get('pendiente_hi'))} carreras por 100 pitcheos"
               + {1: " (el de rodados sufre más)", -1: " (el de rodados sufre menos)", 0: " (sin diferencia medible)",
                  None: ""}[s] + ".")
        m = h3.get("segun_modelo") or {}
        if s == 0 and fin(p.get("semiancho")):               # que tan chica es la pendiente que la prueba puede ver
            mlo, mhi = envolvente((m.get("pendiente_lo"), m.get("pendiente_hi")),
                                  (m.get("pendiente_lo_folds"), m.get("pendiente_hi_folds")))
            txt += (f" Con ese intervalo la prueba solo distingue pendientes de más de {nf(p['semiancho'])}"
                    + (f"; la que implica el modelo es {vic(m['pendiente'], mlo, mhi)}" if fin(m.get("pendiente")) else "")
                    + (f". El % de rodados de una temporada se mide con ruido (fiabilidad {nf(p['fiabilidad_gb'])}), "
                       "lo que acerca la pendiente a 0" if fin(p.get("fiabilidad_gb")) else "") + ".")
    else:
        ver = "sin datos" if ver == "no concluyente" else ver
        txt = ("Dentro del mismo pitcher: sin datos (solo "
               f"{plural(p.get('pitcher_temporadas'), 'pitcher-temporada', 'pitcher-temporadas')} con "
               f"{entero(p.get('min_pitcheos_por_banda'))}+ pitcheos en cada banda).")
    o = b.get("observado") or {}
    if fin(o.get("contraste")):
        s = lado(o.get("contraste_lo"), o.get("contraste_hi"))
        el, ro = o.get("elevado") or {}, o.get("rodado") or {}
        lectura = {1: ", el elevado se encarece más", -1: ", el rodado se encarece más", 0: ", sin diferencia medible",
                   None: ""}[s]
        if s in (1, -1) and str(b.get("veredicto")) not in ("apoyada", "contraria"):
            lectura += " en casa, pero la recta de valor contra densidad no lo confirma"
        txt += (f" Valor de los batazos (no decide el veredicto): de la banda mar a casa un elevado cambia "
                f"{vic(el.get('cambio'), el.get('cambio_lo'), el.get('cambio_hi'))} carreras y un rodado "
                f"{vic(ro.get('cambio'), ro.get('cambio_lo'), ro.get('cambio_hi'))}; contraste "
                f"{vic(o['contraste'], o.get('contraste_lo'), o.get('contraste_hi'))}{lectura} (observado, con la "
                "misma mezcla de elevados suaves y duros en las dos bandas; incluye a los bateadores locales).")
    r = b.get("segun_recta") or {}
    if fin(r.get("contraste")):
        txt += (f" Con la recta de valor contra densidad el contraste es "
                f"{vic(r['contraste'], r.get('contraste_lo'), r.get('contraste_hi'))}{orient(r.get('ic_fiable'))}.")
    return ver, txt


def formas_significativas(formas):
    """Filas alto-bajo cuyo NIVEL en el Harp Helu difiere con los dos intervalos (columna stuff_cdmx_sig)."""
    pide(formas, "insights_formas.csv", "pt", "forma", "tercil", "stuff_cdmx", "stuff_cdmx_sig")
    ab = formas[formas["tercil"].astype(str) == "alto-bajo"]
    ab = ab[[verdadero(x) for x in ab["stuff_cdmx_sig"]]]
    out = []
    for _, r in ab.iterrows():
        lo, hi = envolvente((r.get("stuff_cdmx_lo"), r.get("stuff_cdmx_hi")),
                            (r.get("stuff_cdmx_lo_folds"), r.get("stuff_cdmx_hi_folds")))
        if fin(r["stuff_cdmx"]) and lado(lo, hi) not in (None, 0):
            out.append((abs(r["stuff_cdmx"]),
                        f"{nombre_pt(r['pt'], True)} con {'más' if r['stuff_cdmx'] > 0 else 'menos'} "
                        f"{FORMA_FRASE.get(str(r['forma']), str(r['forma']))} ({vic(r['stuff_cdmx'], lo, hi, 1)})"))
    return [t for _, t in sorted(out, reverse=True)]


def error_del_modelo(D):
    """Lo que score.acuerdo_modelos midio en val_stability.json (los modelos de la validacion cruzada calificando
    a las MISMAS pitcher-temporadas), junto al semiancho medio del intervalo que se imprime por pitcher
    (scores_pitcher.csv). {'stuff' | 'delta': {error, semiancho, sd_ic, desfase, rango, acuerdo}}, en puntos entre
    pitchers: error = el error tipico de UN modelo, una desviacion estandar (score.py: desviacion de la diferencia
    entre dos modelos / raiz de 2); semiancho = el del intervalo impreso, que es de 95%; sd_ic = ese semiancho en
    desviaciones estandar (/ 1.96), que es lo que se compara con `error`; desfase = lo mas que se aparta el nivel
    medio de un modelo del promedio; rango = lo mas que difieren los niveles de DOS modelos. Solo trae las claves
    que se midieron. Ningun intervalo por pitcher incluye ese error: por eso se dice junto a cada tabla."""
    out = {}
    for clave, lo, hi in (("stuff", "stuff_lo", "stuff_hi"), ("delta", "delta_lo", "delta_hi")):
        err = D.stab.get(f"error_modelo_{clave}_pts")
        if not fin(err):
            continue
        semi = np.nan
        if {lo, hi} <= set(D.jug.columns):
            ancho = pd.to_numeric(D.jug[hi], errors="coerce") - pd.to_numeric(D.jug[lo], errors="coerce")
            semi = float((ancho / 2).mean())
        des = D.stab.get(f"desfase_modelos_{clave}_pts")
        des = [float(x) for x in des if fin(x)] if isinstance(des, (list, tuple)) else []
        # el intervalo impreso es de 95% y `error` es UNA desviacion estandar: se comparan en la misma unidad
        out[clave] = {"error": float(err), "semiancho": semi, "sd_ic": semi / 1.96,
                      "desfase": max(abs(x) for x in des) if des else np.nan,
                      "rango": max(des) - min(des) if len(des) > 1 else np.nan,
                      "acuerdo": D.stab.get(f"acuerdo_modelos_{clave}")}
    return out


def unidad_del_ruido(D):
    """De que es el ruido de muestreo de los intervalos por pitcher. score.py toma el juego como unidad si el
    dataset trae game_id (val_stability.json: split_half_por = 'juego'); si no viene, o le falta a demasiados
    pitcheos ('pitcheo'), supone pitcheos independientes y los intervalos salen angostos: el documento lo dice.
    Sin el archivo o sin la clave no se sabe cual fue la unidad y no se afirma ninguna."""
    por = str(D.stab.get("split_half_por"))
    return ("juegos y pitcheos" if por == "juego" else
            "pitcheos tomados como independientes, porque el dataset no trae el juego o le falta a demasiados "
            "pitcheos, lo que angosta el intervalo"
            if por == "pitcheo" else "juegos o pitcheos: out/val_stability.json no dice cuál")


# ── recomendaciones: cada una existe solo si su evidencia pasa su regla ──────
def rec_parque(D):
    p = D.parque
    costo, (lo, hi) = p.get("costo_cdmx_vs_mar_rv100"), par(p, "ic95_recta")
    s = lado(lo, hi)
    if s is None or not fin(costo):
        return None
    if s == 0:
        return ("<b>Evaluar por parque.</b> No hay evidencia suficiente para corregir las líneas de casa por un costo "
                f"de parque: {vic(costo, lo, hi)} carreras por 100 pitcheos{orient(p.get('recta_ic_fiable'))}.")
    cifra = f"{vic(costo, lo, hi)} carreras por 100 pitcheos{orient(p.get('recta_ic_fiable'))}"
    if s < 0:
        return ("<b>Evaluar por parque.</b> Según el modelo el mismo pitcheo permite menos carreras en el Harp Helú que "
                f"al nivel del mar ({cifra}); antes de usar esa cifra conviene revisar la validación por parque de la "
                "sección 4.")
    # Pitching+ a secas valua los batazos en el parque de cada quien: entre parques se compara su version neutral
    neutral = (" (y el Pitching+ neutral, no el de «en su parque»)" if "pitching_neutral_plus_enc" in D.jug else "")
    return ("<b>Evaluar por parque.</b> El mismo pitcheo cuesta más en el Harp Helú que al nivel del mar "
            f"({cifra}), así que las líneas crudas de casa y de gira no son comparables. Para comparar pitchers "
            f"conviene el Stuff+ neutral{neutral}, con el mismo aire para todos, y para proyectarlos en casa, el "
            "Stuff+ en el Harp Helú; están por pitcher en scores_pitcher.csv y en el tablero.")


def rec_tipos(D):
    pide(D.tipo, "altitude_by_pitch_type.csv", "pt", "delta")
    clases = [clase_tipo(r)[0] for _, r in D.tipo.iterrows()]
    if not any(c in ("gana", "pierde") for c in clases):
        return ("<b>Pitcheos para casa.</b> No hay evidencia suficiente para ordenar los tipos de pitcheo por su "
                "cambio relativo en el Harp Helú: ninguno pasa los dos intervalos (pitchers y modelos).")
    ff = fila(D.el, var="ivb", pt="FF")
    perd = (f" La recta de 4 costuras vive del ride y pierde {nf(abs(ff['delta_cdmx_vs_mar']), 1)} pulgadas de IVB."
            if fin(ff.get("delta_cdmx_vs_mar")) and ff["delta_cdmx_vs_mar"] < 0 else "")
    t = D.tipo.set_index(D.tipo["pt"].astype(str))
    costo = ""
    if "costo_cdmx_rv100" in t.columns and t["costo_cdmx_rv100"].notna().sum() >= 2:
        c = t["costo_cdmx_rv100"].astype(float).dropna().sort_values()
        costo = (f" A todos les cuesta más lanzar en el Harp Helú, de {nf(c.iloc[0], 2, True)} carreras por 100 pitcheos "
                 f"({nombre_pt(c.index[0], True)}) a {nf(c.iloc[-1], 2, True)} ({nombre_pt(c.index[-1], True)}): «gana» "
                 "quiere decir que le cuesta menos que al pitcheo promedio.")
    return ("<b>Pitcheos para casa.</b> " + frase_tipos(D.tipo) + " Ganar valor relativo no es ganar movimiento: con "
            "menos aire todos los pitcheos pierden movimiento (la fuerza de Magnus es proporcional a la densidad) y Stuff+ "
            "se recentra en cada aire." + costo + perd + " Es una hipótesis del modelo para el plan de pitcheo en casa: no "
            "se probó que ese valor se mantenga al usar más el pitcheo.")


def rec_formas(D):
    sig = formas_significativas(D.formas)
    if not sig:
        return ("<b>Forma dentro del pitcheo.</b> No hay evidencia suficiente para preferir una versión de algún "
                "pitcheo en el Harp Helú: ninguna diferencia entre tercil alto y bajo pasa los dos intervalos.")
    return ("<b>Forma dentro del pitcheo.</b> Versiones con mayor Stuff+ en el Harp Helú (tercil alto menos tercil "
            "bajo, puntos por pitcheo, IC 95% más ancho): " + "; ".join(sig[:5])
            + (f"; y {len(sig) - 5} más en la tabla de la sección 7" if len(sig) > 5 else "") + ". Sirve para priorizar "
            "desarrollo y scouting; es una diferencia entre pitcheos distintos, no el efecto de cambiar un pitcheo.")


def rec_ride(D):
    """La recta de mucho ride: fila FF / ride / alto-bajo. Nivel en CDMX y cambio por altitud, cada uno con su
    bandera de dos intervalos."""
    r = fila(D.formas, pt="FF", forma="ride", tercil="alto-bajo")
    if not r or not fin(r.get("stuff_cdmx"), r.get("delta_altitud")):
        return None
    n_lo, n_hi = envolvente((r.get("stuff_cdmx_lo"), r.get("stuff_cdmx_hi")),
                            (r.get("stuff_cdmx_lo_folds"), r.get("stuff_cdmx_hi_folds")))
    d_lo, d_hi = envolvente((r.get("delta_lo"), r.get("delta_hi")), (r.get("delta_lo_folds"), r.get("delta_hi_folds")))
    nivel, cambio = vic(r["stuff_cdmx"], n_lo, n_hi, 1), vic(r["delta_altitud"], d_lo, d_hi, 1)
    n_sig, d_sig = verdadero(r.get("stuff_cdmx_sig")), verdadero(r.get("delta_sig"))
    cab = "<b>Recta de mucho ride.</b> "
    cifras = (f"tercil alto menos bajo de ride: {nivel} puntos por pitcheo en el Harp Helú; cambio por la altitud de "
              f"esa diferencia: {cambio}")
    if d_sig and r["delta_altitud"] < 0:
        if n_sig and r["stuff_cdmx"] > 0:
            mar = r["stuff_cdmx"] - r["delta_altitud"]
            cuanto = (f"{100 * -r['delta_altitud'] / mar:.0f}% de su ventaja (de {nf(mar, 1, True)} a "
                      f"{nf(r['stuff_cdmx'], 1, True)} puntos por pitcheo)" if mar > 0 else "parte de su ventaja")
            return cab + f"Pierde {cuanto} con la altitud, pero la conserva en el Harp Helú ({cifras})."
        if n_sig and r["stuff_cdmx"] < 0:
            return cab + f"Pierde su ventaja con la altitud y en el Harp Helú rinde menos que la de poco ride ({cifras})."
        return cab + (f"Pierde ventaja con la altitud; en el Harp Helú ya no se distingue de la de poco ride ({cifras}).")
    if d_sig and r["delta_altitud"] > 0:
        return cab + f"Gana ventaja con la altitud ({cifras})."
    return cab + ("No hay evidencia suficiente para afirmar que la recta de mucho ride rinda distinto en el Harp Helú "
                  f"que al nivel del mar ({cifras}).")


def rec_rodados(D):
    h3 = D.ins.get("H3") or {}
    if not h3:
        return None
    ver = texto_h3(h3)[0]                                   # el mismo veredicto, ya revisado, de la tabla de hipotesis
    o = (h3.get("batazos") or {}).get("observado") or {}
    # el valor de los batazos solo se cita como indicio si su veredicto y su propio intervalo coinciden
    bat = ("apoyada" if str((h3.get("batazos") or {}).get("veredicto")) == "apoyada"
           and lado(o.get("contraste_lo"), o.get("contraste_hi")) == 1 else "")
    cifra = (f"contraste observado de valor elevado menos rodado, banda mar a casa: "
             f"{vic(o.get('contraste'), o.get('contraste_lo'), o.get('contraste_hi'))} carreras por batazo; incluye a "
             "los bateadores locales") if fin(o.get("contraste")) else "sin contraste de batazos"
    cab = "<b>Pitchers de rodados.</b> "
    p = h3.get("pitcher") or {}
    # la cifra que respalda el veredicto es la pendiente de ESA prueba, no el contraste de batazos
    pend = (f"por cada 10 puntos más de % de rodados, la diferencia de run value entre banda alta y banda mar cambia "
            f"{vic(p.get('pendiente'), p.get('pendiente_lo'), p.get('pendiente_hi'))} carreras por 100 pitcheos; "
            f"{entero(p.get('pitchers'))} pitchers")
    if ver == "apoyada":
        return cab + f"La prueba dentro del mismo pitcher apoya preferirlos en casa ({pend})."
    if ver == "contraria":
        return cab + f"La prueba dentro del mismo pitcher va en contra de preferirlos en casa ({pend})."
    prueba = (f"la prueba dentro del mismo pitcher es «{ver}»" if fin(p.get("pendiente")) else
              "la prueba dentro del mismo pitcher no se pudo hacer ("
              f"{plural(p.get('pitcher_temporadas'), 'pitcher-temporada', 'pitcher-temporadas')} con "
              f"{entero(p.get('min_pitcheos_por_banda'))}+ pitcheos en cada banda)")
    if bat == "apoyada":
        return cab + ("El valor de los batazos apunta a que el elevado se encarece más que el rodado en casa "
                      f"({cifra}), pero {prueba}: no hay evidencia suficiente para preferir pitchers de rodados en casa.")
    return cab + f"No hay evidencia suficiente para preferirlos en casa: {prueba}; {cifra}."


def rec_mezcla(D):
    """Lo que hace la liga en altitud (adaptacion.csv, Mezcla) contra lo que dice el modelo (clase_tipo)."""
    mez = D.adapt[D.adapt["tipo"].astype(str) == "Mezcla"] if "tipo" in D.adapt else pd.DataFrame()
    if not len(mez) or not len(D.tipo) or "pt" not in D.tipo:
        return ("<b>Mezcla en altitud.</b> No hay evidencia suficiente para comparar lo que hace la liga al lanzar en "
                "altitud con lo que dice el modelo: el cambio de mezcla no es estimable con estos datos.")
    partes, igual, n = [], 0, 0
    for _, r in D.tipo.iterrows():
        clase = clase_tipo(r)[0]
        u = fila(mez, variable=r["pt"])
        if clase not in ("gana", "pierde") or not u:
            continue
        s = lado(u.get("ic_lo"), u.get("ic_hi"))
        uso = vic(u.get("diferencia"), u.get("ic_lo"), u.get("ic_hi"), 1)
        mov = {1: "la liga lo usa más en altitud", -1: "la liga lo usa menos en altitud",
               0: "la liga no cambia su uso de forma medible", None: "sin dato de uso"}[s]
        n += 1
        igual += s == (1 if clase == "gana" else -1)
        partes.append(f"{nombre_pt(r['pt'], True)} ({clase} valor relativo según el modelo; {mov}: {uso} puntos de uso)")
    if not partes:
        return ("<b>Mezcla en altitud.</b> No hay evidencia suficiente para comparar el uso de la liga con el modelo: "
                "ningún tipo tiene a la vez un cambio relativo medible y un dato de uso.")
    return ("<b>Mezcla en altitud.</b> Uso en banda alta menos banda mar con los mismos pitchers: " + "; ".join(partes)
            + f". En {igual} de {n} el uso de la liga se mueve en la dirección del modelo. La mezcla junta a todos los "
            "equipos, Diablos incluido: no dice qué hacen los rivales, y donde modelo y uso no coinciden hay una "
            "pregunta para el cuerpo técnico, no una conclusión.")


def rec_pitchers(D):
    pide(D.jug, "scores_pitcher.csv", "delta_altitud", "delta_lo", "delta_hi")
    j = D.jug
    baja, sube = int((j["delta_hi"] < 0).sum()), int((j["delta_lo"] > 0).sum())
    if baja + sube == 0:
        return None
    return (f"<b>Uso por parque.</b> De {len(j):,} pitcher-temporadas calificadas, {baja:,} tienen un efecto altitud "
            f"con intervalo completo bajo 0 y {sube:,} sobre 0 (puntos entre pitchers; tablero, pestaña Ranking). "
            "Ese intervalo no incluye el error del modelo (Limitaciones): es una lista de casos para "
            "revisar al armar la rotación de casa y de gira, no un veredicto por pitcher.")


def rec_scouting(D):
    return ("<b>Prospectos.</b> Medir en el bullpen de cualquier parque, capturar en el simulador del tablero el "
            "promedio de cada pitcheo indicando dónde se midió (y la temperatura, si se conoce) y leer su Stuff+ "
            "proyectado en el Harp Helú.")


PLURAL_PT = {"FF": "rectas de 4 costuras", "SI": "sinkers", "FC": "cutters", "SL": "sliders",
             "CU": "curvas", "CH": "cambios", "FS": "splits"}


# ── agregados para las bases del reto (RMSE, arsenal, riesgo de jonrón, Out+, uso operativo) ──
def frase_rmse(nombre, r, contra="cuenta"):
    """RMSE del xRV contra el run value realizado (carreras por pitcheo; el criterio 1 del reto pide RMSE), contra un
    nulo de la MISMA valuacion. Decide el IC pareado por pitchers que escribio model.py (delta_rmse_vs_*)."""
    if not r:
        return ""
    v, lo, hi = (r.get(f"delta_rmse_vs_{contra}"), r.get(f"delta_rmse_vs_{contra}_ci_lo"),
                 r.get(f"delta_rmse_vs_{contra}_ci_hi"))
    rm = r.get("rmse_xrv")
    s = lado(lo, hi)
    if s is None or not fin(v, rm):
        return ""
    base = float(rm) - float(v)                              # RMSE del nulo (delta = modelo menos nulo)
    pct = -100 * float(v) / base if base > 0 else np.nan
    rel = f", {nf(pct, 2 if abs(pct) < .1 else 1)}% menos error" if fin(pct) and s < 0 else ""
    dd = 4 if abs(float(v)) >= 5e-4 else 5
    nulo = "nulo por cuenta" if contra == "cuenta" else "nulo global"
    cifra = f"{nf(rm, dd)} contra {nf(base, dd)} del {nulo}; diferencia × 1,000 = {vic(1e3 * v, 1e3 * lo, 1e3 * hi)}{rel}"
    if s < 0:
        return f"{nombre} baja el error ({cifra})"
    if s > 0:
        return f"{nombre} tiene más error que el {nulo} ({cifra})"
    return f"{nombre} no se distingue del {nulo} ({cifra})"


def _el_que(pt):
    """'la que' / 'el que' según el género del tipo de pitcheo."""
    return "la que" if ARTICULO_PT.get(str(pt)) == "la" else "el que"


def frase_arsenal_cdmx(D, corta=False):
    """Respuesta directa a la pregunta del reto '¿qué arsenal produce el Stuff+ más alto en el Harp Helú?'. Separa
    NIVEL (el tipo y los arsenales con mayor Stuff+ en el Harp Helú) de GANANCIA por la altitud (frase_tipos)."""
    t = D.tipo
    pide(t, "altitude_by_pitch_type.csv", "pt", "stuff_cdmx", "delta")
    t = t[[fin(x) for x in t["stuff_cdmx"]]].sort_values("stuff_cdmx", ascending=False)
    if len(t) < 2:
        raise NoEstimable("sin Stuff+ por tipo en el Harp Helú")
    top, resto = t.iloc[0], t.iloc[1:3]
    gana = t.sort_values("delta", ascending=False).iloc[0]
    a = getattr(D, "ars", pd.DataFrame())
    minimo = (D.ins.get("parametros") or {}).get("min_pitcheos_arsenal_perfil", 100)
    ars, mayoria = "", ""
    if len(a) and {"pt", "pitcheos", "stuff_cdmx_plus_enc"} <= set(a.columns):
        a = a[pd.to_numeric(a["pitcheos"], errors="coerce") >= float(minimo)]
        a = a[[fin(x) for x in a["stuff_cdmx_plus_enc"]]].sort_values("stuff_cdmx_plus_enc", ascending=False)
        k = min(50, len(a))
        if k >= 10:
            cuenta = a.head(k)["pt"].astype(str).value_counts()
            dom, nd = str(cuenta.index[0]), int(cuenta.iloc[0])
            p1 = a.iloc[0]
            quien = esc(str(p1.get("pitcher_name") or p1.get("pitcher_id") or ""))
            mayoria = f"; {nd} de los {k} arsenales con mayor Stuff+ ahí son {PLURAL_PT.get(dom, nombre_pt(dom, True))}"
            ars = (f" Entre los {len(a):,} arsenales (pitcher-temporada-tipo) con {entero(minimo)}+ pitcheos, {nd} de los "
                   f"{k} con mayor Stuff+ en el Harp Helú son {PLURAL_PT.get(dom, nombre_pt(dom, True))}; el primero es "
                   f"{quien} " + (f"{int(p1['season'])} " if fin(p1.get("season")) else "")
                   + f"({nombre_pt(str(p1['pt']), True)}, {vic(p1['stuff_cdmx_plus_enc'], p1.get('stuff_cdmx_lo'), p1.get('stuff_cdmx_hi'), 1, False)}"
                   " puntos entre arsenales).")
    nivel = ""
    if str(gana["pt"]) != str(top["pt"]):
        nivel = (f" {nombre_pt(gana['pt'], articulo=True)[0].upper() + nombre_pt(gana['pt'], articulo=True)[1:]} es "
                 f"{_el_que(gana['pt'])} más gana con la altitud ({nf(gana['delta'], 1, True)}), no {_el_que(gana['pt'])[:2]} "
                 "de mayor nivel.")
    if corta:
        return (f"<b>El tipo de pitcheo con mayor Stuff+ en el Harp Helú es {nombre_pt(top['pt'], articulo=True)}</b> "
                f"({nf(top['stuff_cdmx'], 1)} puntos por pitcheo{mayoria})." + nivel)
    mlb = (" En MLB el slider también encabeza el Stuff+ medio por tipo (110.8, contra 99.2 de la recta de 4 costuras, "
           "92.5 del sinker y 87.2 del cambio; McGrattan, 2023); son escalas distintas y no se comparan cifra a cifra."
           if str(top["pt"]) == "SL" else "")
    txt = ("<b>¿Qué arsenal produce el Stuff+ más alto en el Harp Helú?</b> Por tipo, "
           f"{nombre_pt(top['pt'], articulo=True)} ({nf(top['stuff_cdmx'], 1)} puntos por pitcheo), seguido de "
           + juntar([f"{nombre_pt(r['pt'], articulo=True)} ({nf(r['stuff_cdmx'], 1)})" for _, r in resto.iterrows()])
           + "; los más bajos son " + juntar([f"{nombre_pt(r['pt'], articulo=True)} ({nf(r['stuff_cdmx'], 1)})"
                                               for _, r in t.iloc[-2:].iterrows()]) + "." + nivel)
    return txt + ars + mlb


def frase_uso_operativo(D):
    """Resumen para el cuerpo tecnico (criterio 4 del reto): que decision toma con que salida, con la estabilidad
    medida de Stuff+ contra las carreras permitidas (val_stability.json)."""
    st = D.stab

    def c(k):
        v, (lo, hi) = st.get(k), par(st, f"{k}_ci")
        return f"{nf(v)} {ic(lo, hi)}" if fin(v) else RAYA
    lo_s, _ = par(st, "split_half_stuff_ci")
    _, hi_r = par(st, "split_half_run_value_observado_ci")
    estable = (f" Stuff+ es la medida más estable (confiabilidad por mitades {c('split_half_stuff')} y de una temporada a "
               f"la siguiente {c('estabilidad_stuff_plus_t_t1')}, contra {c('split_half_run_value_observado')} y "
               f"{c('estabilidad_rv_observado_t_t1')} de las carreras permitidas): describe el pitcheo de un prospecto con "
               "pocos lanzamientos, pero por sí solo predice poco las carreras de la temporada siguiente (sección 5)."
               if fin(lo_s, hi_r) and lo_s > hi_r else "")
    rp, rr = st.get("predice_rv_t1__con_pitching_plus_t"), st.get("predice_rv_t1__con_rv_observado_t")
    rs = st.get("predice_rv_t1__con_stuff_plus_t")
    dlo, dhi = par(st, "dif_pitching_vs_rv_ci")
    pred = (f" Para proyectar las carreras del año siguiente se usa Pitching+ (r = {c('predice_rv_t1__con_pitching_plus_t')},"
            f" contra {nf(rr)} de las carreras propias"
            + ("; la diferencia no es medible" if lado(dlo, dhi) == 0 else "") + ")."
            if fin(rp, rr, rs) and rp > max(rs, 0) and lado(dlo, dhi) != -1 else "")
    return ("<b>Uso para el cuerpo técnico</b> (sección 9 y tablero): (1) plan de pitcheo en casa con los tipos y las "
            "formas que ganan valor en el Harp Helú; (2) rotación de casa y de gira con el efecto altitud de cada pitcher "
            "(pestaña Ranking); (3) reclutamiento: medir a un prospecto en cualquier parque y leer en el Simulador o en la "
            "pestaña Prospecto su Stuff+ proyectado en el Harp Helú y al nivel del mar." + estable + pred)


def por_tipo_banda():
    """Por tipo y banda (alta / mar): jonrones por 100 pitcheos (crudo) y P(línea o elevado duro) media de Pitching+."""
    try:
        sp = pd.read_parquet(config.OUT / "scores_pitch.parquet", columns=["pt", "banda", "y_hr", "p_ip_ld", "p_ip_fb_hard"])
    except Exception:                      # noqa: BLE001  (un parquet de otra version no debe tumbar el documento)
        return {}
    sp = sp[sp["banda"].astype(str).isin(["alta", "mar"])].copy()
    sp["y_hr"] = pd.to_numeric(sp["y_hr"], errors="coerce").fillna(0)
    sp["p_duro"] = pd.to_numeric(sp["p_ip_ld"], errors="coerce") + pd.to_numeric(sp["p_ip_fb_hard"], errors="coerce")
    g = sp.groupby([sp["pt"].astype(str), sp["banda"].astype(str)]).agg(hr=("y_hr", "mean"), duro=("p_duro", "mean"),
                                                                         n=("y_hr", "size"))
    return {k: (100 * float(r["hr"]), 100 * float(r["duro"]), int(r["n"])) for k, r in g.iterrows()}


def rec_riesgo_contacto(D):
    """Bases: 'qué perfiles pierden Stuff+ en altitud, cuantificando el riesgo adicional de contacto duro y jonrón'."""
    pide(D.tipo, "altitude_by_pitch_type.csv", "pt", "costo_cdmx_rv100")
    clases = {str(r["pt"]): clase_tipo(r)[0] for _, r in D.tipo.iterrows()}
    t = D.tipo.set_index(D.tipo["pt"].astype(str))
    pierden = [p for p in ORDEN_PT if clases.get(p) == "pierde"]
    ganan = sorted((p for p in ORDEN_PT if clases.get(p) == "gana"), key=lambda p: float(t.loc[p, "costo_cdmx_rv100"]))
    if not pierden:
        return None
    costo = juntar([f"{nombre_pt(p, articulo=True)} {nf(t.loc[p, 'costo_cdmx_rv100'], 2, True)}" for p in pierden])
    ref = (" contra " + juntar([f"{nombre_pt(p, True)} {nf(t.loc[p, 'costo_cdmx_rv100'], 2, True)}" for p in ganan[:2]])
           if ganan else "")
    txt = (f"<b>Riesgo de contacto duro y jonrón.</b> Carreras extra por cada 100 pitcheos en el Harp Helú frente al nivel "
           f"del mar: {costo}{ref} (comparten la incertidumbre de la recta de la recomendación 1).")
    tb = por_tipo_banda()
    det = [f"{nombre_pt(p, True)} {tb[(p, 'alta')][1]:.1f}% contra {tb[(p, 'mar')][1]:.1f}% de línea o elevado duro "
           f"y {tb[(p, 'alta')][0]:.2f} contra {tb[(p, 'mar')][0]:.2f} jonrones por 100 pitcheos"
           for p in pierden if (p, "alta") in tb and (p, "mar") in tb and min(tb[(p, 'alta')][2], tb[(p, 'mar')][2]) >= 2000]
    if det:
        txt += f" En {D.casa()} contra la banda mar: " + "; ".join(det) + " (lo primero, probabilidad de Pitching+; lo segundo, crudo)."
    v = D.val
    d = v.get("densidad") or {}
    neutral, pend, rl = v.get("neutral") or {}, d.get("pendiente") or {}, d.get("rho_liga")
    if fin(rl, neutral.get("ip_fb_hard"), pend.get("ip_fb_hard")):
        mar = neutral["ip_fb_hard"] + pend["ip_fb_hard"] * (D.rho_m - rl)
        cdmx = neutral["ip_fb_hard"] + pend["ip_fb_hard"] * (D.rho_c - rl)
        pic = (d.get("pendiente_ic") or {}).get("ip_fb_hard")
        dr = D.rho_c - D.rho_m
        lo_, hi_ = (sorted(float(x) * dr for x in pic) if isinstance(pic, (list, tuple)) and len(pic) == 2
                    and fin(*pic) else (np.nan, np.nan))
        txt += (f" Un elevado duro (95+ mph) vale {nf(mar, 2, True)} carreras al nivel del mar y {nf(cdmx, 2, True)} en el "
                f"Harp Helú ({vic(cdmx - mar, lo_, hi_)}{orient(d.get('ic_fiable'))}).")
    ps = D.park_scores
    if len(ps) and {"grupo", "hr_fb"} <= set(ps.columns):
        casa, mar_ = fila(ps, grupo="casa"), fila(ps, grupo="mar")
        if fin(casa.get("hr_fb"), mar_.get("hr_fb")):
            txt += (f" Observado, crudo: los jonrones equivalen al {100 * casa['hr_fb']:.1f}% de los elevados en {D.casa()} "
                    f"contra {100 * mar_['hr_fb']:.1f}% en la banda mar.")
    return txt


# ── Out+: P(out dentro del turno) sin reentrenar ──
CLASES_OUT = ["ball", "called_strike", "whiff", "foul", "hbp", "ip_gb", "ip_ld", "ip_fb_soft", "ip_fb_hard", "ip_pu"]
BATAZOS_OUT = [c for c in CLASES_OUT if c.startswith("ip_")]


def calcular_out_turno(D):
    """P(out dentro del turno) por pitcheo, el objetivo «Out+» del reto, con las diez probabilidades de Pitching+
    (fuera de muestra por pitcher) y el valor de cada resultado en esa cuenta, igual que el xRV:
      strike cantado o fallado con 2 strikes -> fraccion de esos pitcheos de la liga que son ponche (KorBB =
      Strikeout: OutsOnPlay NO cuenta el ponche); bola con 3 bolas y golpe -> 0; batazo de la clase c -> fraccion de
      esos batazos de la liga con al menos un out en la jugada (OutsOnPlay); lo demas -> Q(cuenta siguiente), con
      Q(b, s) = P(el turno acaba en out | cuenta) de la liga, por recursion hacia atras (un foul con 2 strikes deja
      la cuenta igual). Nulo = Q(cuenta del pitcheo), el equivalente del nulo por cuenta.
    Validacion contra el turno real: turno = juego + entrada + mitad + bateador, solo si tiene UN pitcheo final.
    Escribe out/predicciones_extra_por_pitcheo.csv.gz (PitchUID, pitching_plus_neutral, p_out_turno,
    p_out_nulo_cuenta): el Pitching+ por pitcheo con los batazos en el mismo aire para todos los parques y Out+."""
    if getattr(D, "_out", None) is not None:
        return D._out
    pc = [f"p_{c}" for c in CLASES_OUT]
    try:
        sp = pd.read_parquet(config.OUT / "scores_pitch.parquet",
                             columns=["pitch_id", "pitcher_id", "balls", "strikes", "outcome", "pitching_neutral_plus"] + pc)
        fe = pd.read_parquet(config.OUT / "features.parquet",
                             columns=["pitch_id", "game_id", "inning", "half", "batter_id", "k_or_bb", "outs_on_play"])
    except Exception as e:                 # noqa: BLE001
        raise NoEstimable(f"faltan columnas para P(out en el turno): {type(e).__name__}")
    df = sp.merge(fe, on="pitch_id", how="left")
    b = pd.to_numeric(df["balls"], errors="coerce").clip(0, 3).fillna(0).astype(int).to_numpy()
    s = pd.to_numeric(df["strikes"], errors="coerce").clip(0, 2).fillna(0).astype(int).to_numpy()
    oc = df["outcome"].astype(str).to_numpy()
    kbb = df["k_or_bb"].astype(str).str.strip().str.lower().to_numpy()
    oop = pd.to_numeric(df["outs_on_play"], errors="coerce").fillna(0).to_numpy()
    # out del bateador en ese pitcheo: ponche (por KorBB) o batazo con al menos un out en la jugada
    out_j = (kbb == "strikeout") | (np.isin(oc, BATAZOS_OUT) & (oop > 0))
    ok = np.isin(oc, CLASES_OUT)
    k2 = ok & np.isin(oc, ["called_strike", "whiff"]) & (s == 2)
    r_k = out_j[k2].mean() if k2.any() else 1.0
    r_ip = {c: (out_j[oc == c].mean() if (oc == c).any() else 0.0) for c in BATAZOS_OUT}
    f = np.zeros((4, 3, len(CLASES_OUT)))
    for j, c in enumerate(CLASES_OUT):
        np.add.at(f[:, :, j], (b[ok & (oc == c)], s[ok & (oc == c)]), 1)
    f = f / np.maximum(f.sum(axis=2, keepdims=True), 1)
    jf = CLASES_OUT.index("foul")
    Q, V = np.zeros((5, 4)), np.zeros((4, 3, len(CLASES_OUT)))     # Q[b, s]; b=4 y s=3 no se usan
    for ss in (2, 1, 0):
        for bb in (3, 2, 1, 0):
            for j, c in enumerate(CLASES_OUT):
                if c == "ball":
                    V[bb, ss, j] = 0.0 if bb == 3 else Q[bb + 1, ss]
                elif c in ("called_strike", "whiff"):
                    V[bb, ss, j] = r_k if ss == 2 else Q[bb, ss + 1]
                elif c == "foul":
                    V[bb, ss, j] = 0.0 if ss == 2 else Q[bb, ss + 1]       # con 2 strikes se fija abajo
                elif c == "hbp":
                    V[bb, ss, j] = 0.0
                else:
                    V[bb, ss, j] = r_ip[c]
            otros = float(np.dot(np.delete(f[bb, ss], jf), np.delete(V[bb, ss], jf)))
            if ss == 2:
                Q[bb, ss] = otros / max(1e-9, 1 - f[bb, ss, jf])
                V[bb, ss, jf] = Q[bb, ss]
            else:
                Q[bb, ss] = otros + f[bb, ss, jf] * V[bb, ss, jf]
    if not .5 <= Q[0, 0] <= .8:
        raise NoEstimable(f"P(out en el turno) desde 0-0 = {Q[0, 0]:.2f}, fuera de lo plausible: revisar KorBB y OutsOnPlay")
    P = df[pc].to_numpy(float)
    p_out = np.nansum(P * V[b, s], axis=1) / np.maximum(np.nansum(P, axis=1), 1e-9)
    nulo = Q[b, s]
    fin_ = np.isin(oc, ["hbp"] + BATAZOS_OUT) | np.isin(kbb, ["strikeout", "walk"])
    clave = (df["game_id"].astype(str) + "|" + df["inning"].astype(str) + "|" + df["half"].astype(str) + "|"
             + df["batter_id"].astype(str)).to_numpy()
    finales = pd.Series(fin_.astype(int)).groupby(clave).transform("sum").to_numpy()
    y_turno = pd.Series((fin_ & out_j).astype(int)).groupby(clave).transform("max").to_numpy()
    val = ok & (finales == 1) & np.isfinite(p_out)
    res = {"n": int(val.sum()), "cobertura": float(val.mean()) if len(val) else np.nan, "r_k": float(r_k),
           "q00": float(Q[0, 0]), "q02": float(Q[0, 2]), "q30": float(Q[3, 0])}
    if val.sum() >= 1000:
        y, pm, pn = y_turno[val], np.clip(p_out[val], 1e-6, 1 - 1e-6), np.clip(nulo[val], 1e-6, 1 - 1e-6)
        pit = df["pitcher_id"].astype(str).to_numpy()[val]
        bm, bn = (y - pm) ** 2, (y - pn) ** 2
        lm, ln = -(y * np.log(pm) + (1 - y) * np.log(1 - pm)), -(y * np.log(pn) + (1 - y) * np.log(1 - pn))
        g = pd.DataFrame({"p": pit, "bm": bm, "bn": bn, "lm": lm, "ln": ln}).groupby("p").sum()
        rng, mej_b, mej_l = np.random.default_rng(7), [], []
        gv = g.to_numpy()
        for _ in range(300):
            m = gv[rng.integers(0, len(gv), len(gv))].sum(axis=0)
            mej_b.append(100 * (1 - m[0] / m[1]))
            mej_l.append(100 * (1 - m[2] / m[3]))
        tot = g.sum()
        cortes = pd.qcut(pm, 10, labels=False, duplicates="drop")
        cal = pd.DataFrame({"c": cortes, "p": pm, "y": y}).groupby("c").mean()
        pend = np.polyfit(cal["p"], cal["y"], 1)[0] if len(cal) >= 3 else np.nan
        res.update({"brier": float(tot["bm"] / len(y)), "brier_nulo": float(tot["bn"] / len(y)),
                    "mejora_brier": float(100 * (1 - tot["bm"] / tot["bn"])),
                    "mejora_brier_ci": (float(np.percentile(mej_b, 2.5)), float(np.percentile(mej_b, 97.5))),
                    "mejora_ll": float(100 * (1 - tot["lm"] / tot["ln"])),
                    "mejora_ll_ci": (float(np.percentile(mej_l, 2.5)), float(np.percentile(mej_l, 97.5))),
                    "pendiente": float(pend), "p_min": float(cal["p"].min()), "p_max": float(cal["p"].max()),
                    "y_min": float(cal["y"].min()), "y_max": float(cal["y"].max()), "pitchers": int(len(g))})
    try:
        pd.DataFrame({"PitchUID": df["pitch_id"], "pitching_plus_neutral": np.round(df["pitching_neutral_plus"], 2),
                      "p_out_turno": np.round(p_out, 4), "p_out_nulo_cuenta": np.round(nulo, 4)}) \
            .to_csv(config.OUT / "predicciones_extra_por_pitcheo.csv.gz", index=False,
                    compression={"method": "gzip", "compresslevel": 6, "mtime": 0})
        res["archivo"] = True
    except OSError:
        res["archivo"] = False
    D._out = res
    return res


def frase_out_turno(D):
    r = calcular_out_turno(D)
    base = ("<b>Out+: P(out dentro del turno).</b> Sale de las diez probabilidades de Pitching+ con el valor de cada "
            "resultado en esa cuenta, como el xRV pero en outs: ponche con 2 strikes, out según la clase de batazo, 0 para "
            "base por bolas y golpe, y para lo demás la probabilidad de out de la cuenta siguiente (recursión sobre las "
            f"cuentas de la liga: el turno acaba en out {100 * r['q00']:.0f}% de las veces desde 0-0, {100 * r['q02']:.0f}% "
            f"desde 0-2 y {100 * r['q30']:.0f}% desde 3-0). Nulo: esa probabilidad de la cuenta, sin ver el pitcheo.")
    if not fin(r.get("mejora_brier")):
        return base + f" Validación contra el turno real: no estimable ({entero(r.get('n'))} pitcheos en turnos completos)."
    lo, hi = r["mejora_brier_ci"]
    llo, lhi = r["mejora_ll_ci"]
    verbo = {1: "mejora al nulo por cuenta", -1: "queda por debajo del nulo por cuenta",
             0: "no se distingue del nulo por cuenta"}[lado(lo, hi)]
    return (base + f" Contra el turno real ({entero(r['n'])} pitcheos, {100 * r['cobertura']:.0f}% del total, de turnos con "
            f"un solo pitcheo final; {entero(r['pitchers'])} pitchers calificados por un modelo que no los vio), Out+ {verbo}: "
            f"Brier {nf(r['brier'], 4)} contra {nf(r['brier_nulo'], 4)}, {vic(r['mejora_brier'], lo, hi, 2)}% menos error, "
            f"y log-loss {vic(r['mejora_ll'], llo, lhi, 2)}%. Calibración por deciles: pendiente {nf(r['pendiente'])}, de "
            f"{100 * r['p_min']:.0f}% a {100 * r['p_max']:.0f}% predicho contra {100 * r['y_min']:.0f}% a "
            f"{100 * r['y_max']:.0f}% observado."
            + (" Por pitcheo en predicciones_extra_por_pitcheo.csv.gz, con el Pitching+ neutral (batazos en el mismo aire "
               "para todos los parques)." if r.get("archivo") else ""))


def frase_discrimina(D):
    """Criterio 1 del reto: ¿los scores separan pitchers buenos de malos? Correlacion (pesada por pitcheos) de Stuff+ y
    de Pitching+ con los fallados por swing y las carreras por 100 de la MISMA temporada, y cuartil alto contra bajo de
    Stuff+. Cada pitcher lo califico un modelo que no lo vio; IC 95% remuestreando pitcher-temporadas."""
    j = D.jug
    need = {"stuff_plus_enc", "pitching_plus_enc", "whiff_swing_pct", "rv100", "pitcheos"}
    if not len(j) or not need <= set(j.columns):
        return ""
    j = j[list(need)].apply(pd.to_numeric, errors="coerce").dropna()
    if len(j) < 50:
        return ""
    w = j["pitcheos"].to_numpy(float)

    def wcorr(x, y, ww):
        mx, my = np.average(x, weights=ww), np.average(y, weights=ww)
        return np.average((x - mx) * (y - my), weights=ww) / np.sqrt(np.average((x - mx) ** 2, weights=ww) *
                                                                       np.average((y - my) ** 2, weights=ww))
    rng = np.random.default_rng(11)
    out = {}
    for sc in ("stuff_plus_enc", "pitching_plus_enc"):
        for y in ("whiff_swing_pct", "rv100"):
            x_, y_ = j[sc].to_numpy(float), j[y].to_numpy(float)
            bs = [wcorr(x_[i], y_[i], w[i]) for i in (rng.integers(0, len(j), len(j)) for _ in range(300))]
            out[(sc, y)] = (wcorr(x_, y_, w), np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    q1, q4 = j["stuff_plus_enc"].quantile([.25, .75])
    alto, bajo = j[j["stuff_plus_enc"] >= q4], j[j["stuff_plus_enc"] <= q1]
    wa = lambda d, c: np.average(d[c], weights=d["pitcheos"])          # noqa: E731
    r = lambda k: vic(*out[k], 2)                                       # noqa: E731
    return ("<b>¿Discriminan?</b> En la misma temporada, entre pitcher-temporadas calificadas por un modelo que no las "
            f"vio, la correlación de Stuff+ con los fallados por swing es {r(('stuff_plus_enc', 'whiff_swing_pct'))} y con "
            f"las carreras permitidas por 100 {r(('stuff_plus_enc', 'rv100'))}; la de Pitching+, "
            f"{r(('pitching_plus_enc', 'whiff_swing_pct'))} y {r(('pitching_plus_enc', 'rv100'))} (con las carreras, "
            "negativa = el score alto permite menos: lo esperado; pesadas por pitcheos). El cuartil alto de Stuff+ tiene "
            f"{100 * wa(alto, 'whiff_swing_pct'):.1f}% de fallados por swing y {nf(wa(alto, 'rv100'), 2, True)} carreras por "
            f"100, contra {100 * wa(bajo, 'whiff_swing_pct'):.1f}% y {nf(wa(bajo, 'rv100'), 2, True)} del cuartil bajo "
            "(negativo = a favor del pitcher).")


# ── marco teórico y literatura (citas verificadas contra la fuente; ver Referencias) ──
def sec_marco(D):
    q = D.rho_c / D.rho_m if fin(D.rho_c, D.rho_m) and D.rho_m else np.nan
    aire = f" (en el Harp Helú hay {100 * (1 - q):.0f}% menos aire que en los parques de la LMB al nivel del mar)" if fin(q) else ""
    # como se estimaron los pesos solo se dice si out/valores_carrera.json lo dice (frase_valor_carrera)
    valor = ("con pesos lineales sacados de la expectativa de carreras por estado de bases y outs (Tango et al., 2007), "
             "estimada con el dataset oficial (sección 1)" if es_markov(D.vc) else
             "con el método de pesos lineales (Tango et al., 2007; sección 1)")
    return [Paragraph(
        "<b>Sabermetría.</b> Stuff+ califica un pitcheo solo por su física en la salida, Location+ por su ubicación y "
        "Pitching+ por todo junto; la escala «+» pone el promedio en 100 y una desviación estándar por pitcheo en 10 "
        "puntos, y los pitcheos secundarios se juzgan contra la recta primaria del pitcher (Appelman, 2023; McGrattan, "
        f"2023). Cada resultado se valúa en carreras {valor}. <b>Física.</b> Las fuerzas de arrastre y de Magnus son "
        "proporcionales a la densidad del aire"
        + aire + ": en Denver, con 82% del aire del nivel del mar, un quiebre de 18 pulgadas baja a 14–15, la curva cae "
        "unas 4 pulgadas menos y la recta pierde unas 4 de subida (Nathan, 2007), y los elevados vuelan alrededor de 5% "
        "más (Adair, 2002, citado en Nathan, 2007). <b>Entorno.</b> En ligas menores de EE.UU. y Canadá, de 13 a 5,150 "
        "pies, cada 1,000 pies de altitud suman unos 0.12 jonrones y 0.56 carreras por juego, contando a los dos equipos "
        "(Richardson, 2014); en Coors, en 1995–1998, el elevado promedio voló 18.3 pies más que en el resto de la Liga "
        "Nacional, sin igualar el batazo (Zaidins et al., 2003). El Harp Helú, a 2,236 m (unos 7,300 pies), queda fuera "
        "de esos rangos y ninguno de esos estudios usa la LMB (en la literatura académica con datos públicos de MLB la "
        "fuente más usada es PITCHf/x; Huang y Hsu, 2021): por eso cada efecto se mide aquí con la LMB y se contrasta "
        "con esas cifras.", P)]


def frase_literatura(D):
    """Contraste de lo medido en la LMB con Nathan (2007). Las cifras de la LMB son la media al nivel del mar llevada al
    aire del Harp Helu con la beta usada (media_mar * (q^beta - 1)), con su IC; no es una resta directa entre parques."""
    el, q = D.el, (D.rho_c / D.rho_m if fin(D.rho_c, D.rho_m) and D.rho_m else np.nan)
    if not fin(q):
        return ""
    ivb, hb, cu = fila(el, var="ivb", pt="FF"), fila(el, var="hb_n", pt="FF"), fila(el, var="mov_total", pt="CU")

    def perdida(r):
        m, d, lo, hi = r.get("media_mar"), r.get("delta_cdmx_vs_mar"), r.get("delta_lo"), r.get("delta_hi")
        if not fin(m, d):
            return None
        return (f"{nf(abs(d), 1)}" + (f" {ic(min(abs(lo), abs(hi)), max(abs(lo), abs(hi)), 1)}" if fin(lo, hi) and
                                     lo * hi > 0 else ""), f"({nf(m, 1)} → {nf(m + d, 1)})", lo, hi)
    a, b_, c_ = perdida(ivb), perdida(hb), perdida(cu)
    partes = []
    if a:
        partes.append(f"la recta de 4 costuras pierde {a[0]} pulgadas de IVB {a[1]}" + (f" y {b_[0]} de HB" if b_ else ""))
    if c_:
        partes.append(f"la curva, que en la LMB quiebra tanto de lado como hacia abajo, {c_[0]} de movimiento total {c_[1]}")
    if not partes:
        return ""
    cuatro = all(x and fin(x[2], x[3]) and min(abs(x[2]), abs(x[3])) <= 4 <= max(abs(x[2]), abs(x[3])) for x in (a, c_) if x)
    rb = fila(el, var="ivb", pt="FAM_RECTAS")
    mayor = lado(rb.get("ci_lo"), rb.get("ci_hi"), 1.0) == 1
    txt = (f" <b>Contra la literatura</b> (Nathan, 2007; marco teórico): con {100 * (1 - q):.0f}% menos aire en el Harp "
           "Helú y las β de la tabla, " + "; ".join(partes) + ". Es el mismo orden que las ~4 pulgadas que calcula Nathan "
           "para Denver" + (" y los intervalos del IVB de la recta y del movimiento de la curva las incluyen" if cuatro else "") + f": el Harp Helú tiene {100 * q:.0f}% "
           "del aire del nivel del mar, contra 82% en Denver, "
           + ("y la β medida excede 1." if mayor else "y la β medida no se distingue de 1."))
    c = D.alt.get("carry") or {}
    v, (lo, hi) = c.get("pies_extra_cdmx_vs_mar"), par(c, "ic95")
    if fin(v, lo, hi):
        comp = ("del orden de" if lo <= 20 <= hi or lo <= 18.3 <= hi else "mayor que" if lo > 20 else "menor que")
        if lo <= 0 <= hi:
            txt += (f" A igual velocidad y ángulo de salida, la estimación del vuelo extra del elevado en el Harp Helú, "
                    f"{vic(v, lo, hi, 0)} pies{orient(c.get('ic_fiable'))}, es del orden de los ~20 pies (5%) que da la "
                    "física para Denver, pero su intervalo incluye 0.")
        else:
            txt += (f" A igual velocidad y ángulo de salida, el elevado vuela {vic(v, lo, hi, 0)} pies más en el Harp Helú"
                    f"{orient(c.get('ic_fiable'))}: {comp} los ~20 pies (5%) que da la física para Denver"
                    + ("; con menos aire que en Denver se espera algo más." if comp == "mayor que" else "."))
    return txt


REFERENCIAS = [
    "Appelman, D. (2023, 10 de marzo). PitchingBot and Stuff+ pitch modeling is now on FanGraphs! <i>FanGraphs "
    "Baseball</i>. https://blogs.fangraphs.com/pitchingbot-and-stuff-pitch-modeling-are-now-on-fangraphs/",
    "Hersbach, H., et al. (2020). The ERA5 global reanalysis. <i>Quarterly Journal of the Royal Meteorological Society, "
    "146</i>(730), 1999–2049. https://doi.org/10.1002/qj.3803 · Datos públicos: Open-Meteo (ERA5 y Copernicus DEM "
    "GLO-90), https://open-meteo.com; MLB Stats API (calendario LMB 2024–2026), https://statsapi.mlb.com",
    "Huang, J.-H., y Hsu, Y.-C. (2021). A multidisciplinary perspective on publicly available sports data in the era of big "
    "data: A scoping review of the literature on Major League Baseball. <i>SAGE Open, 11</i>(4), artículo "
    "21582440211061566. https://doi.org/10.1177/21582440211061566",
    "McGrattan, O. (2023, 10 de marzo). Stuff+, Location+, and Pitching+ primer. <i>FanGraphs Sabermetrics Library</i>. "
    "https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/",
    "Nathan, A. M. (2007, act. 2008). Baseball at high altitude: Comparing baseball trajectories at Coors and Fenway. "
    "<i>The Physics of Baseball</i>, University of Illinois. https://baseball.physics.illinois.edu/Denver.html",
    "Nathan, A. M. (2011, 25 de febrero). Baseball ProGUESTus: Home runs and humidors: Is there a connection? "
    "<i>Baseball Prospectus</i>. https://www.baseballprospectus.com/news/article/13057/",
    "Richardson, E. (2014). High altitude offense: An empirical examination of the relationship between runs scored and "
    "stadium elevation. <i>Baseball Research Journal, 43</i>(2). SABR. https://sabr.org/journal/article/high-altitude-offense",
    "Tango, T. M., Lichtman, M. G., y Dolphin, A. E. (2007). <i>The Book: Playing the percentages in baseball</i>. "
    "Potomac Books.",
    "Zaidins, C., Chambers, F., y Page, B. (2003). Into thin air: What's all the fuss about Coors Field? <i>Above the "
    "Fruited Plain</i> (SABR 33). SABR. https://sabr.org/journal/article/into-thin-air-whats-all-the-fuss-about-coors-field/",
]


def sec_referencias(D):
    return [Paragraph("<br/>".join(REFERENCIAS), ParagraphStyle("ref", parent=NOTA, alignment=0, leftIndent=8,
                                                                  firstLineIndent=-8))]


# ── tabla() ─────────────────────────────────────────────────────────────────
class Tabla(Table):
    """Tabla que solo se parte entre paginas si a cada lado quedan al menos `min_filas` filas de datos (nunca deja
    una o dos filas huerfanas) y nunca justo despues de una fila de `pegadas` (el titulo de un grupo de filas).
    Si no se puede partir asi, pasa entera a la pagina siguiente."""
    min_filas = 4
    pegadas = frozenset()

    def split(self, availWidth, availHeight):
        cab, m = self.repeatRows or 0, self.min_filas
        for _ in range(6):                                   # cada vuelta corta unas filas antes
            partes = Table.split(self, availWidth, availHeight)
            if len(partes) != 2:
                return partes
            n1, n2 = partes[0]._nrows - cab, partes[1]._nrows - cab
            if (partes[0]._nrows - 1) in self.pegadas:
                sobran = 1                                   # el titulo del grupo se va con sus filas
            elif n1 >= m and n2 >= m:
                return partes
            elif n2 < m <= n1 - (m - n2):
                sobran = m - n2                              # la cola quedo corta
            else:
                return []
            try:
                availHeight -= sum(partes[0]._rowHeights[-sobran:]) + 1
            except Exception:              # noqa: BLE001
                return []
        return []


def celda(x, estilo=CELDA):
    return Paragraph(str(x), estilo)


def tabla(df, anchos=None, semaforo=None, fmt=None, semaforo_abs=None, num=None, fondo=None, tam=7.8, izq=1,
          extra=None, min_filas=4, pegadas=()):
    """DataFrame -> tabla del documento.
    semaforo      {col: (umbral_bueno, umbral_malo, mayor_es_mejor)} sobre el valor
    semaforo_abs  {col: (umbral_bueno, umbral_malo)} sobre |valor|: para sesgos importa la distancia a 0
    num           {col: valores numericos} cuando la celda es texto ("+0.03 [-0.66, +0.72]") y el color sale del numero
    fondo         {col: [color o None por fila]} colores que ya decidio una regla (p. ej. IC a un lado de 0)
    min_filas     filas de datos que deben quedar a cada lado si la tabla se parte entre paginas (ver Tabla)
    pegadas       filas (1 = la primera de datos) que no pueden quedar al final de una pagina: titulos de grupo
    Nunca se colorea un NaN."""
    fmt, cols = fmt or {}, list(df.columns)
    tam = tam * AJUSTE["tabla"]
    enc = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=tam, leading=tam + 1.6, textColor=colors.white,
                         alignment=1)
    enc_izq = ParagraphStyle("thi", parent=enc, alignment=0)

    def texto(v, f):
        if hasattr(v, "wrap") or isinstance(v, (str, list, tuple)):   # Paragraph(s) o texto ya armado
            return v
        if not fin(v):
            return RAYA
        if f is not None:
            return f(v)
        return f"{int(v):,}" if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else f"{float(v):.2f}"
    datos = [[Paragraph(str(c), enc_izq if j < izq else enc) for j, c in enumerate(cols)]]
    for _, r in df.iterrows():
        datos.append([texto(r[c], fmt.get(c)) for c in cols])
    if anchos is not None and sum(anchos) > ANCHO:           # una tabla nunca sale mas ancha que la pagina
        anchos = [a * ANCHO / sum(anchos) for a in anchos]
    t = Tabla(datos, colWidths=anchos, repeatRows=1)
    t.min_filas, t.pegadas = (min_filas if AJUSTE["entera"] else min(min_filas, 2)), frozenset(pegadas)
    est = [("FONT", (0, 1), (-1, -1), "Helvetica", tam), ("LEADING", (0, 1), (-1, -1), tam + 1.6),
           ("BACKGROUND", (0, 0), (-1, 0), NEGRO), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
           ("ALIGN", (izq, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
           ("LINEBELOW", (0, 0), (-1, -1), .25, GRIS_CLARO), ("TOPPADDING", (0, 0), (-1, -1), 2),
           ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("LEFTPADDING", (0, 0), (-1, -1), 3),
           ("RIGHTPADDING", (0, 0), (-1, -1), 3)]

    def valores(col):
        v = (num or {}).get(col, df[col].tolist())
        return pd.to_numeric(pd.Series(list(v)), errors="coerce").to_numpy(float)
    for col, (bueno, malo, mayor) in (semaforo or {}).items():
        j = cols.index(col)
        for i, v in enumerate(valores(col), start=1):
            if np.isfinite(v):
                ok = v >= bueno if mayor else v <= bueno
                mal = v <= malo if mayor else v >= malo
                est.append(("BACKGROUND", (j, i), (j, i), VERDE if ok else ROJO if mal else AMBAR))
    for col, (bueno, malo) in (semaforo_abs or {}).items():
        j = cols.index(col)
        for i, v in enumerate(np.abs(valores(col)), start=1):
            if np.isfinite(v):
                est.append(("BACKGROUND", (j, i), (j, i), VERDE if v <= bueno else ROJO if v >= malo else AMBAR))
    for col, lista in (fondo or {}).items():
        j = cols.index(col)
        for i, c in enumerate(lista, start=1):
            if c is not None:
                est.append(("BACKGROUND", (j, i), (j, i), c))
    t.setStyle(TableStyle(est + list(extra or [])))
    return t


# ── figuras ─────────────────────────────────────────────────────────────────
def _fig_dir():
    d = config.OUT / "fig"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _guardar(fig, nombre, bbox=None):
    path = _fig_dir() / nombre
    fig.savefig(path, dpi=170, bbox_inches=bbox)
    plt.close(fig)
    w, h = ImageReader(str(path)).getSize()                  # con bbox="tight" no es el tamano del lienzo
    ancho = ANCHO * AJUSTE["figura"]
    return Image(str(path), ancho, ancho * h / w)


def _ejes(ax):
    for lado_ in ("top", "right"):
        ax.spines[lado_].set_visible(False)
    for lado_ in ("left", "bottom"):
        ax.spines[lado_].set_color("#888888")
        ax.spines[lado_].set_linewidth(.6)
    ax.tick_params(colors="#333333", width=.6, length=3)


def fig_parques(D):
    """Promedios crudos por categoria de altitud: IVB de la recta y carreras por 100 pitcheos contra altitud."""
    pk = D.parks
    pide(pk, "park_table.csv", "stadium", "altitud", "ivb_ff", "rv_por_100")
    p = pk.sort_values("altitud").reset_index(drop=True)
    casa = (p["stadium"].astype(str) == config.HOME_ALTITUDE).to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.55))
    x_lbl = "Altitud representativa de la categoría (m)"
    alt_ = p["altitud"].to_numpy(float)
    for a, col, tit, yl in ((ax[0], "ivb_ff", "IVB medio de la recta de 4 costuras", "pulgadas"),
                            (ax[1], "rv_por_100", "Carreras por 100 pitcheos (+ = a favor del bateo)", "carreras por 100")):
        a.scatter(p["altitud"][~casa], p[col][~casa], c="#111111", s=24, zorder=2)
        a.scatter(p["altitud"][casa], p[col][casa], c=C_PIERDE, s=64, zorder=3)
        a.set_title(tit, fontsize=10, loc="left")
        a.set_xlabel(x_lbl)
        a.set_ylabel(yl)
        _ejes(a)
        if col == "rv_por_100":
            a.axhline(0, color="#999999", lw=.6, zorder=1)
        a.margins(x=.14, y=.26)
        v = p[col].to_numpy(float)
        if casa.any():                                       # la etiqueta de casa va a la esquina derecha que queda libre
            i = int(np.flatnonzero(casa)[0])
            arriba = v[i] < np.nanmedian(v)                  # el punto de casa esta abajo -> etiqueta arriba
            a.annotate(f"{config.HOME_ALTITUDE} (casa)", (alt_[i], v[i]),
                       xytext=(.99, .95 if arriba else .06), textcoords="axes fraction", fontsize=9, color="#111111",
                       ha="right", va="top" if arriba else "bottom",
                       arrowprops=dict(arrowstyle="-", color="#888888", lw=.6, shrinkA=1, shrinkB=5))
        # pocas categorias: cada punto con su nombre
        ancho_x = float(np.nanmax(alt_) - np.nanmin(alt_)) or 1.0
        x_casa = alt_[casa][0] if casa.any() else np.nan
        for i in np.flatnonzero(~casa):
            if np.isfinite(x_casa) and abs(x_casa - alt_[i]) < .3 * ancho_x:
                # pegado al punto de casa: a la izquierda y abajo, lejos de la linea que une casa con su etiqueta
                a.annotate(str(p["stadium"].iloc[i]), (alt_[i], v[i]), xytext=(-7, -11), ha="right",
                           textcoords="offset points", fontsize=8.5, color="#333333")
                continue
            # del lado contrario a donde queda el punto siguiente, para no encimar etiquetas
            sube = i + 1 < len(v) and np.isfinite(v[i + 1]) and v[i + 1] > v[i]
            a.annotate(str(p["stadium"].iloc[i]), (alt_[i], v[i]), xytext=(7, -11 if sube else 5),
                       textcoords="offset points", fontsize=8.5, color="#333333")
    fig.tight_layout()
    return _guardar(fig, "parques.png")


def fig_tipo(D):
    """Cambio relativo por tipo con su intervalo mas ancho; color solo donde la regla de clase_tipo afirma algo."""
    pide(D.tipo, "altitude_by_pitch_type.csv", "pt", "delta")
    t = D.tipo[[fin(v) for v in D.tipo["delta"]]].sort_values("delta").reset_index(drop=True)
    if not len(t):
        raise NoEstimable("out/altitude_by_pitch_type.csv no trae cambios estimados")
    clases = [clase_tipo(r) for _, r in t.iterrows()]
    col = [C_GANA if c == "gana" else C_PIERDE if c == "pierde" else C_NEUTRO for c, _, _ in clases]
    d = t["delta"].to_numpy(float)
    lo = np.array([c[1] if fin(c[1]) else v for c, v in zip(clases, d)], dtype=float)
    hi = np.array([c[2] if fin(c[2]) else v for c, v in zip(clases, d)], dtype=float)
    fig, ax = plt.subplots(figsize=(9, .85 + .2 * len(t)))
    y = np.arange(len(t))
    ax.barh(y, d, color=col, height=.62, zorder=2)
    ax.errorbar(d, y, xerr=[np.maximum(d - lo, 0), np.maximum(hi - d, 0)], fmt="none", ecolor="#111111", elinewidth=.9,
                capsize=2.5, zorder=3)
    ax.axvline(0, color="#111111", lw=.8, zorder=1)
    ax.set_yticks(y)
    npit = t["pitchers"] if "pitchers" in t else [np.nan] * len(t)
    ax.set_yticklabels([nombre_pt(p) + (f" · {entero(n)} pitchers" if fin(n) else "") for p, n in zip(t["pt"], npit)])
    x1 = max(float(np.nanmax(hi)), 0)
    x0 = min(float(np.nanmin(lo)), 0)
    paso = (x1 - x0) or 1.0
    for yi, v, (c, a, b) in zip(y, d, clases):               # la cifra, su intervalo y la lectura, junto a cada barra
        ax.text(x1 + .03 * paso, yi, f"{vic(v, a, b, 1)}  {c}", va="center", ha="left", fontsize=8.5, color="#222222")
    ax.set_xlim(x0 - .04 * paso, x1 + .7 * paso)
    ax.set_xlabel("Cambio relativo de Stuff+: Harp Helú menos nivel del mar (puntos por pitcheo)", loc="left")
    ax.set_xticks([v for v in ax.get_xticks() if x0 - .04 * paso <= v <= x1 + .02 * paso])   # sin marcas bajo el texto
    _ejes(ax)
    fig.tight_layout()
    return _guardar(fig, "tipo.png")


def fig_mapa(D):
    """El mismo conjunto de pitcheos evaluado en los dos aires; cada panel con el quiebre que tendria en ESE aire."""
    sp = D.sp
    pide(sp, "scores_pitch.parquet", "pt", "ivb_mar", "hb_mar", "ivb_cdmx", "hb_cdmx", "stuff_mar_plus",
         "stuff_cdmx_plus")
    d = sp.dropna(subset=["ivb_mar", "hb_mar", "ivb_cdmx", "hb_cdmx", "stuff_mar_plus", "stuff_cdmx_plus"])
    d = d.sample(min(N_FIGURA, len(d)), random_state=config.RANDOM_STATE) if len(d) else d
    if len(d) < 200:
        raise NoEstimable("muy pocos pitcheos con quiebre transportado en out/scores_pitch.parquet")
    cmap = LinearSegmentedColormap.from_list("stuff", ["#2166AC", "#9DB9D6", "#DCDCDC", "#E0A08F", "#B2182B"])
    xs = np.r_[d["hb_mar"].to_numpy(float), d["hb_cdmx"].to_numpy(float)]
    ys = np.r_[d["ivb_mar"].to_numpy(float), d["ivb_cdmx"].to_numpy(float)]
    ext = (*np.percentile(xs, [.3, 99.7]), *np.percentile(ys, [.3, 99.7]))
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.55), sharex=True, sharey=True)
    hb = None
    for ax, x, y, c, tit in ((axs[0], "hb_mar", "ivb_mar", "stuff_mar_plus", f"Nivel del mar (densidad {nf(D.rho_m, 3)})"),
                             (axs[1], "hb_cdmx", "ivb_cdmx", "stuff_cdmx_plus", f"Harp Helú (densidad {nf(D.rho_c, 3)})")):
        hb = ax.hexbin(d[x], d[y], C=d[c], reduce_C_function=np.mean, gridsize=38, extent=ext, cmap=cmap,
                       vmin=85, vmax=115, mincnt=max(3, len(d) // 8000), linewidths=0)
        ax.axhline(0, color="#999999", lw=.5)
        ax.axvline(0, color="#999999", lw=.5)
        for pt, g in d.groupby("pt"):                        # etiqueta en la mediana de cada tipo
            if len(g) >= 150:
                ax.text(g[x].median(), g[y].median(), str(pt), fontsize=8.5, ha="center", va="center", color="#111111",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.75))
        ax.set_title(tit, fontsize=10)
        ax.set_xlabel("HB (pulg; + = lado del brazo)")
        _ejes(ax)
    axs[0].set_ylabel("IVB (pulg)")
    fig.tight_layout()
    cb = fig.colorbar(hb, ax=axs, shrink=.9, pad=.015)
    cb.set_label("Stuff+ medio de la celda")                 # corto: una etiqueta mas larga que el eje agranda la figura
    cb.outline.set_visible(False)
    return _guardar(fig, "mapa.png", bbox="tight"), len(d)


def fig_calibracion(D):
    """Curvas de confiabilidad fuera de muestra (10 grupos de igual tamano): observado contra predicho."""
    cal = D.cal
    pide(cal, "val_calibracion.csv", "objetivo", "modelo", "grupo", "pred_media", "obs_media")
    c = cal[cal["grupo"].astype(str) == "todos"]
    paneles = [("whiff", "Swing fallado (%)"), ("called_strike", "Strike cantado (%)"), ("gb", "Rodado (%)"),
               ("xrv", "xRV (carreras por 100 pitcheos)")]
    fig, axs = plt.subplots(1, 4, figsize=(9.2, 2.3))
    hay = False
    for ax, (obj, tit) in zip(axs, paneles):
        lim = []
        for modelo, color, mk, nom, mfc in (("stuff", "#111111", "o", "Stuff+", "#111111"),
                                            ("pitching", C_AZUL, "s", "Pitching+", "white")):
            g = c[(c["objetivo"].astype(str) == obj) & (c["modelo"].astype(str) == modelo)].sort_values("pred_media")
            if not len(g):
                continue
            x, y = 100 * g["pred_media"].to_numpy(float), 100 * g["obs_media"].to_numpy(float)
            ax.plot(x, y, marker=mk, ms=3.6, lw=1, color=color, mfc=mfc, mew=.9, label=nom, zorder=3)
            lim += [np.nanmin(np.r_[x, y]), np.nanmax(np.r_[x, y])]
            hay = True
        if lim:
            a, b = min(lim), max(lim)
            ax.plot([a, b], [a, b], color="#999999", lw=.7, zorder=1)
        ax.set_title(tit, fontsize=9.5)
        ax.set_xlabel("Predicho")
        _ejes(ax)
    if not hay:
        plt.close(fig)
        raise NoEstimable("out/val_calibracion.csv no trae el grupo 'todos'")
    axs[0].set_ylabel("Observado")
    axs[0].legend(frameon=False, fontsize=8.5, loc="upper left", handlelength=1.6)
    fig.tight_layout()
    return _guardar(fig, "calibracion.png")


# ── secciones ───────────────────────────────────────────────────────────────
def sec_titulo(D):
    quien = []
    if config.EQUIPO not in PLACEHOLDERS:
        quien.append(f"<b>{esc(str(config.EQUIPO))}</b>")
    # un texto suelto ("Ana, Luis") no es una lista: recorrerlo imprimia una letra por integrante
    integrantes = [config.INTEGRANTES] if isinstance(config.INTEGRANTES, str) else list(config.INTEGRANTES)
    nombres = [esc(str(x)) for x in integrantes if str(x).strip() and x not in PLACEHOLDERS]
    if nombres:
        quien.append(", ".join(nombres))
    if config.INSTITUCION not in PLACEHOLDERS:
        quien.append(esc(str(config.INSTITUCION)))
    if len(quien) < 3:
        aviso("la portada sale sin equipo, integrantes o institucion: llena EQUIPO, INTEGRANTES e INSTITUCION "
              "en src/config.py y vuelve a correr 'python run_all.py --solo report'")
    quien.append("ISAC Sports Analytics Hackathon 2026 · Diablos Rojos del México")
    n_pit = len(D.folds) if len(D.folds) else (pd.to_numeric(D.res["pitchers"], errors="coerce").max()
                                               if "pitchers" in D.res else np.nan)
    datos = ["Stuff+, Location+ y Pitching+ para el Estadio Alfredo Harp Helú (2,236 m)"]
    if D.n_pitcheos:
        datos.append(f"{D.n_pitcheos:,} pitcheos calificados")
    if fin(n_pit):
        datos.append(f"{entero(n_pit)} pitchers")
    if D.temporadas:
        datos.append("temporada " + str(D.temporadas[0]) if len(D.temporadas) == 1 else
                     f"temporadas {D.temporadas[0]}–{D.temporadas[-1]}")
    if fin(D.k):
        datos.append(f"{entero(D.k)} categorías de altitud (modo categoría)")
    out = [Paragraph("Stuff+ LMB calibrado por altitud", ParagraphStyle("t", parent=H1, fontSize=20, leading=24,
                                                                         spaceBefore=0, keepWithNext=0)),
           Paragraph(" · ".join(quien), ParagraphStyle("au", parent=P, fontSize=9, spaceAfter=2, alignment=0)),
           Paragraph(" · ".join(datos), ParagraphStyle("dl", parent=NOTA, alignment=0))]
    if D.rapido:
        out.append(Paragraph(f"<b>{MARCA_RAPIDA}</b>: las cifras de este documento solo sirven para comprobar que todo "
                             "corre; con tan pocos pitchers la mayoría de los resultados sale «no concluyente».",
                             ParagraphStyle("rap", parent=P, backColor=GRIS_CLARO,
                                            borderPadding=(3, 4, 3, 4), spaceBefore=3, spaceAfter=6, alignment=0)))
    return out


def frase_validacion_resumen(D):
    pide(D.res, "val_resumen.csv", "modelo")
    k = D.ph["fold"].nunique() if "fold" in D.ph else None
    pit, st, sc = fila(D.res, modelo="pitching"), fila(D.res, modelo="stuff"), fila(D.res, modelo="stuff_cuenta")
    partes = []
    if pit:
        partes.append(frase_vs_nulo("Pitching+", pit, "cuenta"))
    if sc:
        partes.append(frase_vs_nulo("Stuff+ con la cuenta añadida", sc, "cuenta", corta=True))
    if st:
        partes.append(frase_vs_nulo("Stuff+ solo, que no ve la cuenta,", st, "global", corta=True)
                      + " y " + frase_vs_nulo("", st, "cuenta", corta=True))
    if not partes:
        raise NoEstimable("val_resumen.csv sin modelos")
    folds = f" ({k} folds)" if k else ""
    return f"Fuera de muestra, con pitchers que ningún modelo vio{folds}: " + "; ".join(partes) + "."


def frase_neutralidad_resumen(D):
    r = fila(D.cf, bloque="neutralidad", pt="TODOS")
    if not r or not fin(r.get("valor")):
        return "La prueba de neutralidad del contrafactual no es estimable con estos datos."
    return ("Prueba del contrafactual: el Stuff+ neutral del mismo pitcheo, lanzado en altitud y al nivel del mar, "
            f"difiere {vic(r['valor'], r.get('ic_lo'), r.get('ic_hi'))} puntos por pitcheo ("
            + frase_aceptacion("neutralidad", r["valor"], r.get("ic_lo"), r.get("ic_hi"),
                               r.get("ic_dentro_de_tolerancia")) + ").")


def sec_resumen(D):
    p1 = [seguro(frase_aire, D), seguro(frase_movimiento, D), seguro(frase_costo, D.parque, True)]
    p2 = [seguro(frase_validacion_resumen, D), seguro(frase_neutralidad_resumen, D),
          seguro(frase_arsenal_cdmx, D, True), seguro(frase_tipos, D.tipo)]
    p3 = [seguro(frase_uso_operativo, D)]
    out = [Paragraph(" ".join(x for x in p if x), P) for p in (p1, p2, p3) if any(p)]
    if not out:
        raise NoEstimable("no hay resultados en out/")
    return out


def frase_sin_rastreo(lz):
    """Los batazos que el sistema de medicion no rastreo (sin tipo, angulo ni velocidad de salida) se descartan en
    features.py: se dice cuantos son y si acabaron distinto que los rastreados (out/limpieza.json)."""
    n, tot = lz.get("sin_rastreo"), lz.get("batazos")
    if not (fin(n, tot) and n and tot):
        return None
    ev, evr, nr = lz.get("sin_rastreo_por_evento") or {}, lz.get("rastreados_por_evento") or {}, lz.get("rastreados")
    txt = (f"También se descartan {entero(n)} batazos sin rastreo (sin tipo, ángulo ni velocidad de "
           f"salida; {100 * float(n) / float(tot):.1f}% de los batazos): sin su tipo no entran a las clases de "
           "resultado.")
    if not (fin(nr) and nr):
        return txt
    out_sr, out_r = 100 * ev.get("Out", 0) / float(n), 100 * evr.get("Out", 0) / float(nr)
    if out_sr > out_r + 5:
        return txt + (f" Acabaron en out {out_sr:.0f}% de ellos, contra {out_r:.0f}% de los rastreados: el sistema de rastreo pierde "
                      "sobre todo contacto débil, así que el modelo ve un poco menos de ese contacto que el real.")
    if out_sr < out_r - 5:
        return txt + (f" Acabaron en out {out_sr:.0f}% de ellos, contra {out_r:.0f}% de los rastreados: el sistema de rastreo pierde "
                      "más los batazos que acaban en hit.")
    return txt + f" Acabaron en out {out_sr:.0f}% de ellos, como los rastreados ({out_r:.0f}%)."


def es_markov(vc):
    """True si out/valores_carrera.json dice que el valor en carreras salio de la cadena de bases y outs. Sin el
    archivo (o con otro metodo) el documento no afirma como se estimo."""
    return "markov" in str((vc or {}).get("metodo") or "").lower()


def frase_valor_carrera(vc):
    """Como se estimo el valor en carreras, que el dataset no trae (features.estimar_valores_carrera), con las cifras
    de out/valores_carrera.json: medias entradas con que se calibra la cadena, su desvio maximo (ajuste.desvio_max_frecuentes, o residuo_max si falta:
    el mayor error relativo, con piso, entre lo que da la cadena y lo medido) y los pesos de los eventos. Lo que se
    ajusta (carreras por tipo de jugada, doble play y carreras por media entrada) y que solo entran las medias
    entradas completas de las entradas 1 a 9 es lo que hace features.estimar_valores_carrera."""
    if not es_markov(vc):
        raise NoEstimable("falta out/valores_carrera.json o no dice el metodo")
    aj = vc.get("ajuste") or {}
    n_me, res = vc.get("medias_entradas"), aj.get("desvio_max_frecuentes", aj.get("residuo_max"))
    w = vc.get("pesos") if isinstance(vc.get("pesos"), dict) else {}
    pesos = [f"{nom} {nf(w.get(e), 2, True)}" for e, nom in (("Single", "sencillo"), ("Double", "doble"),
             ("HomeRun", "jonrón"), ("Walk", "base por bolas"), ("Out", "out"), ("Strikeout", "ponche")) if fin(w.get(e))]
    return ("El dataset no trae valor en carreras: se estima con él mismo. Una cadena de Markov de bases y outs, "
            "calibrada con sus " + (f"{entero(n_me)} " if fin(n_me) else "") + "medias entradas completas (entradas 1 a 9) "
            "para reproducir las carreras medias de cada tipo de jugada, el doble play y las carreras por media entrada"
            + (f" (desvío máximo {nf(100 * float(res), 1)}%)" if fin(res) else "")
            + ", da la expectativa de carreras de cada estado (RE24) y el peso de cada evento"
            + (f" ({', '.join(pesos)})" if pesos else "") + ". El valor de cada cuenta sale de la cadena de cuentas; "
            "sacrificio y jugada de selección valen como out.")


def sec_datos(D):
    partes = ["Dataset oficial del hackathon: pitcheo por pitcheo de la LMB, con identificadores anónimos."]
    if D.n_features and D.n_pitcheos and D.n_features != D.n_pitcheos:
        partes.append(f"Tras la limpieza quedan {D.n_features:,} pitcheos, de los cuales {D.n_pitcheos:,} tienen valor "
                      "en carreras y se califican.")
    elif D.n_features and D.n_pitcheos:
        partes.append(f"Tras la limpieza quedan {D.n_pitcheos:,} pitcheos, todos con valor en carreras: son los que se "
                      "califican.")
    elif D.n_pitcheos:
        partes.append(f"Se califican {D.n_pitcheos:,} pitcheos.")
    partes.append("Se descartan bolas intencionales y automáticas, pitcheos sin cantar, toques, pitcheos sin tipo "
                  "reconocido y pitcheos sin velocidad, movimiento, ubicación o cuenta.")
    sr = frase_sin_rastreo(D.limpieza)
    if sr:
        partes.append(sr)
    partes.append("Todo lo horizontal (quiebre, lado de salida, ubicación) "
                  "se expresa hacia el lado del brazo del pitcher, para medir a zurdos y derechos en la misma escala.")
    partes.append(seguro(frase_valor_carrera, D.vc, defecto="El dataset no trae valor en carreras; su estimación "
                         f"(out/valores_carrera.json): {NO_ESTIMABLE.lower()}."))
    ch = D.chequeo
    if len(ch) and {"derivada", "acuerdo", "n"} <= set(ch.columns):
        nom = {"y_swing": "swing", "y_whiff": "swing fallado", "y_called_strike": "strike cantado", "y_chase": "chase"}
        ok = [(nom[str(r["derivada"])], r["acuerdo"]) for _, r in ch.iterrows()
              if str(r["derivada"]) in nom and fin(r["acuerdo"]) and fin(r["n"]) and r["n"] > 0]
        if ok:
            txt = ("Las etiquetas salen de PitchCall; contra las banderas oficiales del dataset coinciden en "
                   + juntar([f"{100 * a:.1f}% ({n})" for n, a in ok]) + " de los pitcheos")
            notas = sorted({str(x) for x in ch["nota"].dropna()} - {"", "nan"}) if "nota" in ch else []
            partes.append(txt + (f" ({esc('; '.join(notas))})." if notas else "."))
        z = fila(ch, derivada="in_zone_caja")
        if z.get("usada") and str(z["usada"]) != "nan":
            partes.append(f"Zona de strike: {esc(str(z['usada']))}.")
    return [Paragraph(" ".join(partes), P)]


def sec_variables(D):
    f = D.feats
    if not any(f.values()):
        raise NoEstimable("no se pudieron leer las variables de los modelos (models/*_fold0.joblib)")
    todas = []
    for nm in ("stuff", "location", "pitching"):
        todas += [v for v in f[nm] if v not in todas]
    familias = []
    for v in todas:
        fam = VARIABLES.get(v, ("Otras", v))[0]
        if fam not in familias:
            familias.append(fam)
    orden = [x for x in dict.fromkeys(fa for fa, _ in VARIABLES.values()) if x in familias] + \
        [x for x in familias if x == "Otras"]
    filas = []
    for fam in orden:
        vs = [v for v in todas if VARIABLES.get(v, ("Otras", v))[0] == fam]
        marcas = []
        for nm in ("stuff", "location", "pitching"):
            k = sum(v in f[nm] for v in vs)
            marcas.append("sí" if k == len(vs) else RAYA if k == 0 else f"{k} de {len(vs)}")
        filas.append([celda(fam), celda(", ".join(esc(VARIABLES.get(v, ("", v))[1]) for v in vs))] + marcas)
    t = pd.DataFrame(filas, columns=["Familia", "Variables (leídas del modelo ajustado)", "Stuff+", "Location+",
                                     "Pitching+"])
    out = [tabla(t, anchos=[x * inch for x in (1.55, 3.6, .6, .7, .65)], izq=2)]
    fuera = [txt for claves, txt in FUERA_DE_STUFF if f["stuff"] and not any(c in f["stuff"] for c in claves)]
    nota = (f"{len(f['stuff'])} variables en Stuff+, {len(f['location'])} en Location+ y {len(f['pitching'])} en "
            "Pitching+ (el tipo de pitcheo incluido).")
    if any(o != "modelo" for o in D.feats_origen.values()):
        nota += " No se pudo abrir algún modelo: esa lista es la del código."
    if fuera:
        nota += " Fuera de Stuff+ a propósito: " + "; ".join(fuera) + "."
    return out + [Paragraph(nota, NOTA)]


def sec_fisica_intro(D):
    fiable = bool(len(D.el)) and "ic_fiable" in D.el and any(verdadero(x) for x in D.el["ic_fiable"])
    donde = f"cada {D.nivel()} donde lanzó" + (f" {MIN_CELDA_FISICA} o más pitcheos" if MIN_CELDA_FISICA else "")
    txt = ("Diseño: cada pitcher contra sí mismo. Para cada pitcher × temporada × tipo de pitcheo se promedia el "
           f"movimiento en {donde} y se ajusta movimiento = nivel propio × densidad<super>β</super>. β es la "
           "elasticidad: el cambio porcentual del movimiento por cada 1% de densidad del aire; β = 1 es lo que daría "
           f"una fuerza proporcional a la densidad. IC 95% quitando {D.un_nivel()} a la vez (jackknife)")
    txt += "." if fiable else (f"; con {entero(D.k)} niveles de densidad esos intervalos son <b>orientativos</b>."
                               if fin(D.k) else "; con tan pocos niveles de densidad son <b>orientativos</b>.")
    return [Paragraph(txt, P)]


def sec_fisica_figura(D):
    return [fig_parques(D),
            Paragraph(f"Promedios crudos por {D.nivel()} (incluyen quién lanza en cada parque): no son la estimación, "
                      f"solo muestran la escala del efecto. En rojo, {D.casa()}.", NOTA)]


def sec_fisica_tabla(D):
    el = D.el
    pide(el, "altitude_elasticities.csv", "var", "pt", "beta", "ci_lo", "ci_hi", "estimable", "beta_usada", "fuente",
         "media_mar", "delta_cdmx_vs_mar", "pitchers")

    def medida(r):
        if not r or not verdadero(r.get("estimable")) or not fin(r.get("beta")):
            return "no estimable"
        return f"{nf(r['beta'])} {ic(r.get('ci_lo'), r.get('ci_hi'))}"

    limites = tuple(float(x) for x in getattr(config, "BETA_LIMITES", ()))
    MARCA = " °"

    def usada(r):
        """'1.32 fam.'; la propia de la fila, si physics.py la recorto a un limite de cordura, sale '2.00 tope'."""
        f = str(r.get("fuente", ""))
        if not fin(r.get("beta_usada")):
            return RAYA
        en_limite = any(abs(float(r["beta_usada"]) - x) < 1e-9 for x in limites)
        recortada = f.startswith("propia") and (str(r.get("motivo")) == "recortada" or (
            en_limite and fin(r.get("beta")) and abs(r["beta_usada"] - r["beta"]) > .005))
        return f"{nf(r['beta_usada'])} {'tope' if recortada else ABREV_FUENTE.get(f, f)}".strip()

    def delta(r):
        """(texto, intervalo fiable: True / False / None si no hay) del cambio en pulgadas con la beta usada. El
        intervalo sale de los extremos del intervalo de ESA beta (delta_lo / delta_hi de physics.py)."""
        v, lo, hi = r.get("delta_cdmx_vs_mar"), r.get("delta_lo"), r.get("delta_hi")
        if not fin(v, lo, hi):
            return nf(v, 1, True), None
        d = 1 if abs(v) >= .05 else 2                        # un quiebre medio de ~0 no sale como "+0.0 [-0.01, -0.01]"
        return vic(v, lo, hi, d), verdadero(r.get("ic_fiable_usada"))

    def origen(var, pt, r):
        """Fila de la que un tipo toma PRESTADA su beta ({} si usa la suya, que ya esta en su renglon)."""
        f = str(r.get("fuente", ""))
        if f == "mov_total":
            m = fila(el, var="mov_total", pt=pt)
            return (origen("mov_total", pt, m) or m) if str(m.get("fuente")) != "mov_total" else {}
        return {"familia": fila(el, var=var, pt=FAMILIA_DE.get(pt, "")), "liga": fila(el, var=var, pt="TODOS")}.get(f, {})
    filas, fiables, total, motivos, dudosas = [], [], 0, {}, {}
    pts = [p for p in ORDEN_PT + list(FAMILIAS) + ["TODOS"] if (el["pt"].astype(str) == p).any()]
    for pt in pts:
        a, b = fila(el, var="ivb", pt=pt), fila(el, var="hb_n", pt=pt)
        if not a and not b:
            continue
        if pt in ORDEN_PT:
            for var, nom, r in (("ivb", "IVB", a), ("hb_n", "HB", b)):
                if not fin(r.get("beta_usada")):
                    continue
                total += 1
                if not str(r.get("fuente", "")).startswith("propia"):
                    m = str(r.get("motivo", ""))
                    motivos[m] = motivos.get(m, 0) + 1
                o = origen(var, pt, r)
                if o and lado(o.get("ci_lo"), o.get("ci_hi")) == 0:
                    de = ("la del movimiento total de " if str(o.get("var")) == "mov_total" else "la de ") \
                        + f"«{nombre_pt(o.get('pt'))}»"
                    clave = (nom, de, vic(o.get("beta"), o["ci_lo"], o["ci_hi"], 2, False))
                    dudosas.setdefault(clave, []).append(nombre_pt(pt, articulo=True))
        (da, fa), (db, fb) = delta(a), delta(b)
        fiables.append((fa, fb))
        filas.append([nombre_pt(pt), entero(a.get("pitchers", b.get("pitchers"))), medida(a), usada(a),
                      nf(a.get("media_mar"), 1), da, medida(b), usada(b), nf(b.get("media_mar"), 1), db])
    if not filas:
        raise NoEstimable("out/altitude_elasticities.csv no trae filas de ivb / hb_n")
    # los intervalos orientativos se marcan uno por uno solo si conviven con otros fiables; si no, se dice una vez
    con_ic = [f for par_ in fiables for f in par_ if f is not None]
    mixto = any(con_ic) and not all(con_ic)
    if mixto:
        for f_, (fa, fb) in zip(filas, fiables):
            f_[5], f_[9] = f_[5] + (MARCA if fa is False else ""), f_[9] + (MARCA if fb is False else "")
    cols = ["Pitcheo", "Pitchers", "β del IVB [IC 95%]", "β usada", "IVB, mar", "Δ en el Harp Helú [IC]",
            "β del HB [IC 95%]", "β usada ", "HB, mar", "Δ en el Harp Helú [IC] "]
    n_tipos = sum(p in ORDEN_PT for p in pts)
    t = tabla(pd.DataFrame(filas, columns=cols),
              anchos=[x * inch for x in (1.08, .5, .9, .56, .42, .98, .9, .56, .42, .98)], tam=7.2,
              extra=[("LINEABOVE", (0, n_tipos + 1), (-1, n_tipos + 1), .6, GRIS)] if 0 < n_tipos < len(filas) else None)
    tope = f" [{nf(limites[0], 1)}, {nf(limites[1], 1)}]" if len(limites) == 2 else ""
    nota = ("β usada = la que mueve los pitcheos en el contrafactual; su origen: prop. = la propia de la fila (en un "
            "tipo, encogida hacia su familia), fam. = la de su familia de pitcheos, liga = la de todos los pitcheos, "
            "mov. = la del movimiento total del tipo (cuando el quiebre medio es casi 0 o cambia de signo), fís. = 1, "
            f"tope = la propia recortada al límite de cordura{tope}. «IVB, mar» y «HB, mar» = quiebre medio en "
            f"pulgadas llevado a densidad {nf(D.rho_m, 3)}; Δ = lo que cambia al llevarlo a {nf(D.rho_c, 3)} con la β "
            "usada, y su intervalo sale de los extremos del intervalo de esa β, recortados a los mismos límites"
            + (f" ({MARCA.strip()} = orientativo)" if mixto else ", orientativo" if con_ic and not any(con_ic) else "")
            + ". HB + = lado del brazo. Las filas de familia y de todos los pitcheos no usan la etiqueta del tipo.")
    if total:
        det = [f"{n} {MOTIVO.get(m, m)}" for m, n in sorted(motivos.items(), key=lambda x: -x[1]) if m and m != "nan"]
        nota += (f" En esta corrida {sum(motivos.values())} de las {total} elasticidades por tipo usan un valor "
                 "prestado" + (f": {juntar(det)}" if det else "") + ".")
    for (nom, de, cifra), tipos in dudosas.items():
        nota += (f" La β del {nom} que {'usan' if len(tipos) > 1 else 'usa'} {juntar(tipos)} es {de}, cuyo "
                 f"intervalo contiene 0 ({cifra}): lo que este documento dice de "
                 f"{'esos tipos' if len(tipos) > 1 else 'ese tipo'} descansa en un valor que estos datos no fijan.")
    return [t, Paragraph(nota, NOTA)]


def sec_controles(D):
    el, alt = D.el, D.alt
    pide(el, "altitude_elasticities.csv", "var", "pt", "beta", "ci_lo", "ci_hi")
    partes = []
    vl = fila(el, var="velo_loss", pt="TODOS")
    if fin(vl.get("beta"), vl.get("ci_lo"), vl.get("ci_hi")):
        s, mph = lado(vl["ci_lo"], vl["ci_hi"]), vl.get("delta_cdmx_vs_mar")
        cifra = f"β = {vic(vl['beta'], vl['ci_lo'], vl['ci_hi'], 2, False)}{orient(vl.get('ic_fiable'))}"
        puntual = f" (estimación puntual del cambio en el Harp Helú: {nf(mph, 1, True)} mph)" if fin(mph) else ""
        if s == 1:
            lectura = (f"la bola frena menos en aire delgado: en el Harp Helú pierde {nf(abs(mph), 1)} mph menos de la "
                       "mano al plato" if fin(mph) else "la bola frena menos en aire delgado")
        elif s == -1:
            lectura = "la bola frena más en aire delgado"
        elif (not verdadero(vl.get("ic_fiable")) and verdadero(vl.get("signo_estable"))
              and fin(vl.get("loo_min"), vl.get("loo_max"))):
            # con pocos niveles el jackknife es tan ancho que incluye 0 aun con un efecto real: se dice lo que SI se
            # sabe (la direccion no depende de un solo nivel) sin llamarlo significativo
            lectura = (f"el intervalo incluye 0, pero la dirección no depende de {D.un_nivel()}: quitando cualquiera, β "
                       f"queda entre {nf(vl['loo_min'])} y {nf(vl['loo_max'])}, siempre "
                       + ("con menos frenado" if vl["beta"] > 0 else "con más frenado") + f" en aire delgado{puntual}")
        else:
            lectura = "no se distingue de 0 con este intervalo" + puntual
        partes.append(f"<b>Arrastre</b> (velocidad que pierde la bola de la mano al plato): {cifra}; {lectura}.")
    filas_pl = [fila(el, var="spin_rate", pt="TODOS"), fila(el, var="velocity", pt="TODOS")]
    pl = [frase_placebo("spin", filas_pl[0], "rpm", 0), frase_placebo("velocidad de salida", filas_pl[1], "mph", 2)]
    fam = fila(el, var="ivb", pt="FAM_RECTAS")
    escala = (f" Como escala, el IVB de las rectas cambia {100 * fam['delta_cdmx_vs_mar'] / fam['media_mar']:+.0f}%."
              if fin(fam.get("delta_cdmx_vs_mar"), fam.get("media_mar")) and fam["media_mar"] else "")
    pocos = any(r and not verdadero(r.get("ic_fiable")) for r in filas_pl)
    partes.append("<b>Placebos</b> (lo que el aire no debería mover; una β distinta de 0 querría decir que parte de lo "
                  "medido es diferencia entre parques y no aire"
                  + (f"; con {entero(D.k)} niveles de densidad, un intervalo que incluye 0 no prueba que no la haya"
                     if pocos and fin(D.k) else "") + "): " + "; ".join(pl) + "." + escala)
    va = fila(D.adapt, tipo="Mecánica", variable="Velocidad") if "tipo" in D.adapt else {}
    if va and verdadero(va.get("significativo")) and fin(va.get("diferencia"), va.get("ic_lo"), va.get("ic_hi")):
        partes[-1] += (f" Con el remuestreo de pitchers (sección 3) la velocidad de la mano sí difiere entre bandas "
                       f"({vic(va['diferencia'], va['ic_lo'], va['ic_hi'], 2)} mph): una diferencia entre parques que no "
                       "viene del aire.")
    c = alt.get("carry") or {}
    if fin(c.get("pies_extra_cdmx_vs_mar")):
        lo, hi = par(c, "ic95")
        s = lado(lo, hi)
        cifra = (f"{vic(c['pies_extra_cdmx_vs_mar'], lo, hi, 0)} pies{orient(c.get('ic_fiable'))}; "
                 f"{entero(c.get('batazos'))} batazos")
        lectura = {1: "viaja más en el Harp Helú", -1: "viaja menos en el Harp Helú",
                   0: "no viaja distinto de forma medible", None: "sin intervalo"}[s]
        partes.append("<b>Batazos</b> (elevados de 15° a 45° y 80+ mph, a igual velocidad y ángulo de salida): el mismo "
                      f"batazo {lectura} que al nivel del mar ({cifra}).")
    if not partes:
        raise NoEstimable("sin controles en out/altitude_elasticities.csv")
    return [Paragraph(" ".join(partes), P)]


def frase_transporte(cf):
    """Por que el contrafactual usa la beta empirica (ADENDA 2.f), con las filas TODOS de val_contrafactual.csv:
    'transporte' (la beta usada) y 'transporte_beta_1' (la misma cuenta con el quiebre proporcional a la
    densidad). Solo se dice que la empirica reproduce el movimiento y que beta = 1 no cuando la primera queda
    dentro de su tolerancia y la segunda fuera de ella: las dos se juzgan con la misma regla (|error| < tolerancia,
    ADENDA 2.d). Si la empirica pasa solo por la estimacion puntual (su intervalo sale de la tolerancia) se dice.
    Es una comprobacion de AJUSTE: las celdas son las mismas con las que physics.py estimo beta."""
    tol = TOLERANCIA["transporte"]
    cifras, falla, hay_b1, b1_mal, puntual = [], [], False, [], []
    for var, nom in (("ivb", "IVB"), ("hb_n", "HB")):
        r, u = (fila(cf, bloque=b, pt="TODOS", variable=var) for b in ("transporte", "transporte_beta_1"))
        if not fin(r.get("valor")):
            continue
        c = f"{vic(r['valor'], r.get('ic_lo'), r.get('ic_hi'))} pulg en {nom}"
        if fin(u.get("valor")):
            hay_b1 = True
            c += f" (con β = 1, {vic(u['valor'], u.get('ic_lo'), u.get('ic_hi'))})"
            if abs(u["valor"]) >= tol:                       # la MISMA regla que a la beta empirica: |error| < tolerancia
                b1_mal.append(nom)
        cifras.append(c)
        if abs(r["valor"]) >= tol:
            falla.append(nom)
        elif fin(r.get("ic_lo"), r.get("ic_hi")) and frase_aceptacion(
                "transporte", r["valor"], r["ic_lo"], r["ic_hi"], r.get("ic_dentro_de_tolerancia"),
                corta=True) != "equivalente":
            puntual.append(nom)                              # pasa, pero su intervalo no cabe en la tolerancia
    if not cifras:
        return " El contrafactual usa la β empírica; su prueba de transporte no es estimable con estos datos."
    txt = (" El contrafactual usa la β empírica. Llevando al aire de la banda alta lo que cada pitcher lanzó al nivel "
           "del mar, el error medio en la dirección del quiebre es " + juntar(cifras) + ": ")
    if falla:
        txt += (f"en {juntar(falla)} queda fuera de la tolerancia de {tol:g} pulg, así que la β empírica no reproduce "
                "ahí el movimiento observado")
    elif b1_mal:
        txt += (f"la β empírica reproduce el movimiento observado dentro de la tolerancia de {tol:g} pulg y β = 1 no lo "
                f"hace en {juntar(b1_mal)}")
    elif hay_b1:
        txt += f"las dos quedan dentro de la tolerancia de {tol:g} pulg, así que esta prueba no distingue una de otra"
    else:
        txt += f"queda dentro de la tolerancia de {tol:g} pulg"
    if puntual and not falla:
        txt += (f" (en {juntar(puntual)} la β empírica pasa solo por la estimación puntual: su intervalo sale de la "
                "tolerancia)")
    return txt + (". Es una comprobación de ajuste, con las mismas celdas con que se estimó β, no una validación "
                  "independiente (sección 5).")


def sec_lectura(D):
    """Parrafo 'Lectura' (ADENDA 2.f): cuanto de la perdida medida explica beta = 1 y cuanto queda 'no explicado'.
    El tiempo de vuelo NO se cuenta como explicacion: lo que el vuelo mas corto le quita al quiebre se lo devuelve
    la mayor velocidad (encabezado de physics.py); la fila `sustentacion` mide la elasticidad sin tiempo de vuelo
    ni velocidad y se cita con su intervalo. No se propone ningun mecanismo."""
    el = D.el
    pide(el, "altitude_elasticities.csv", "var", "pt", "beta", "ci_lo", "ci_hi")
    r = fila(el, var="ivb", pt="FAM_RECTAS") or fila(el, var="ivb", pt="FF")
    b, lo, hi = r.get("beta"), r.get("ci_lo"), r.get("ci_hi")
    q = D.rho_c / D.rho_m if fin(D.rho_c, D.rho_m) and D.rho_m else np.nan
    if not r or not verdadero(r.get("estimable", True)) or not fin(b, q) or not 0 < q < 1:
        raise NoEstimable("sin elasticidad del IVB de las rectas")
    quien = "las rectas (FF+SI)" if str(r.get("pt")) == "FAM_RECTAS" else "la recta de 4 costuras"
    total, base = 1 - q ** b, 1 - q
    txt = (f"<b>Lectura.</b> Con {100 * (1 - q):.0f}% menos aire, una fuerza proporcional a la densidad (β = 1) quitaría "
           f"{100 * base:.0f}% del quiebre. En {quien} la β medida del IVB es {vic(b, lo, hi, 2, False)}"
           f"{orient(r.get('ic_fiable'))}; {frase_beta_vs_1(lo, hi)}.")
    if total > base:                                         # hay algo mas que la proporcionalidad
        txt += (f" Con esa β la pérdida es de {100 * total:.0f}%: β = 1 da cuenta de {100 * base / total:.0f}% de ella; "
                f"el {100 * (total - base) / total:.0f}% restante queda <b>no explicado</b>"
                + (" (estimación puntual: como el intervalo de β incluye 1, ese resto no se distingue de 0)."
                   if lado(lo, hi, 1.0) == 0 else "."))
    elif total > 0:
        txt += f" Con esa β la pérdida es de {100 * total:.0f}%, no mayor que la que da β = 1."
    su = fila(el, var="sustentacion", pt=str(r.get("pt")))
    if fin(su.get("beta"), su.get("ci_lo"), su.get("ci_hi")):
        txt += (" El tiempo de vuelo no entra en esa cuenta: una bola que llega antes también va más rápido, y la "
                "fuerza que la desvía crece con la velocidad al cuadrado. Medida sin tiempo de vuelo ni velocidad "
                "(aceleración vertical inducida entre velocidad media al cuadrado), la β es "
                f"{vic(su['beta'], su['ci_lo'], su['ci_hi'], 2, False)}{orient(su.get('ic_fiable'))} "
                f"({frase_beta_vs_1(su['ci_lo'], su['ci_hi']).split(':')[0]}).")
    otras = []
    for nom, f in (("la del HB", fila(el, var="hb_n", pt=str(r.get("pt")))),
                   ("la del movimiento total de todos los pitcheos", fila(el, var="mov_total", pt="TODOS"))):
        if verdadero(f.get("estimable")) and fin(f.get("beta")):
            otras.append(f"{nom} es {vic(f['beta'], f.get('ci_lo'), f.get('ci_hi'), 2, False)} "
                         f"({frase_beta_vs_1(f.get('ci_lo'), f.get('ci_hi')).split(':')[0]})")
    if otras:
        txt += " Como elasticidad del quiebre en pulgadas (la de la tabla), " + juntar(otras) + "."
    txt += (" No proponemos un mecanismo para una β distinta de 1: no se midió ninguno. Parte puede ser calibración de "
            f"la medición propia de cada parque, que con {'los ' + entero(D.k) if fin(D.k) else 'pocos'} niveles de "
            "densidad disponibles (y pocos parques altos) no se puede separar del efecto del aire.")
    lit = seguro(frase_literatura, D, defecto="")
    return [Paragraph(txt + frase_transporte(D.cf) + (lit or ""), P)]


def origen_densidad(D):
    """{etiqueta: de donde salio su densidad}, con la misma regla que features._ambiente:
      'medida'    la etiqueta esta en config.ALTITUDE_RHO (medias de los parques de la LMB con el reanalisis ERA5
                  en los juegos del calendario publico de la liga) y la densidad de park_table.csv es esa
      'estandar'  no esta: atmosfera estandar a la altitud representativa x config.CLIMA_FACTOR. NO es una medicion
    El documento no puede decir 'medida' de una etiqueta que no esta en config.ALTITUDE_RHO. {} sin tabla de
    parques."""
    if not len(D.parks) or not {"stadium", "densidad"} <= set(D.parks.columns):
        return {}
    tabla_rho = getattr(config, "ALTITUDE_RHO", {})
    out = {}
    for _, r in D.parks.iterrows():
        et = str(r["stadium"]).strip()
        ref = tabla_rho.get(et)
        out[et] = "medida" if fin(ref, r["densidad"]) and abs(float(r["densidad"]) - float(ref)) < 5e-4 else "estandar"
    return out


def sec_aire(D):
    pk = D.parks
    pide(pk, "park_table.csv", "stadium", "densidad", "pitcheos")
    cats = juntar([f"{esc(str(r['stadium']))}: {nf(r['densidad'], 3)} ({entero(r['pitcheos'])} pitcheos)"
                   for _, r in pk.sort_values("densidad", ascending=False).iterrows()])
    org = origen_densidad(D)
    med, est = ([esc(e) for e, o in org.items() if o == k] for k in ("medida", "estandar"))
    medida = ("la media, a mitad del juego, de los parques de la LMB de ese rango de altitud (calendario de la liga "
              "en MLB Stats API; temperatura, humedad y presión de ERA5, Hersbach et al., 2020: fuentes públicas, externas al dataset)")
    estandar = ("la de la atmósfera estándar a la altitud representativa de la categoría, corregida por el clima "
                "típico de los juegos de la LMB: no es una densidad medida")
    if med and est:
        fuente = f"para {juntar(med)}, {medida}; para {juntar(est)}, {estandar}"
    else:
        fuente = estandar if est else medida
    return [Paragraph("El dataset oficial solo trae la categoría de altitud de cada pitcheo. A cada categoría se le "
                      f"asigna una densidad relativa del aire (1 = 1.225 kg/m³): {fuente}. Son {cats}. Los "
                      "metros de cada parque son los del modelo de elevación Copernicus GLO-90. El "
                      "contrafactual evalúa cada pitcheo en dos aires de referencia: Harp Helú, "
                      f"{nf(D.rho_c, 3)}, y parques al nivel del mar, {nf(D.rho_m, 3)}. En este modo «casa» es toda la "
                      f"categoría {config.HOME_ALTITUDE}: junta todos sus parques, no solo el Harp Helú.", P)]


def nota_de(r):
    """Texto de la columna `nota` de una fila ('' si esta vacia: un vacio se lee de vuelta como NaN)."""
    n = r.get("nota")
    return "" if n is None or str(n).strip().lower() in ("", "nan", "none") else str(n).strip()


def sec_adaptacion(D):
    a = D.adapt
    pide(a, "adaptacion.csv", "tipo", "variable", "unidad", "diferencia", "ic_lo", "ic_hi", "pitchers")
    tipo = a["tipo"].astype(str)
    mec, mez, et = a[tipo.str.startswith("Mec")], a[tipo == "Mezcla"], a[tipo == "Etiqueta"]
    out, al_pie = [], ""
    if len(mec):
        dec = [0 if str(u) == "rpm" else 2 for u in mec["unidad"]]       # rpm sin decimales; mph y pulgadas con dos
        # physics.py anota las filas que son la misma cantidad que un placebo (velocidad y spin de la mano): un
        # cambio ahi no se puede leer como adaptacion del pitcher, y la tabla lo marca
        notas = [nota_de(r) for _, r in mec.iterrows()]
        t = pd.DataFrame({"Variable": [str(v) + (" *" if n else "") for v, n in zip(mec["variable"], notas)],
                          "Unidad": mec["unidad"].astype(str).tolist(),
                          "Banda alta menos banda mar": [nf(v, d, True) for v, d in zip(mec["diferencia"], dec)],
                          "IC 95%": [ic(lo, hi, d, True) for lo, hi, d in zip(mec["ic_lo"], mec["ic_hi"], dec)],
                          "Lectura": [frase_cambia(r) for _, r in mec.iterrows()],
                          "Pitchers": [entero(v) for v in mec["pitchers"]]})
        if any(notas):
            al_pie = "* " + "; ".join(dict.fromkeys(esc(n[0].upper() + n[1:]) for n in notas if n)) + " (sección 2). "
        out += [KeepTogether([
            Paragraph("Mismo pitcher, misma temporada y mismo tipo de pitcheo, lanzado en banda alta y en banda "
                      "mar. El contrafactual deja fijas estas variables: si cambiaran mucho, comparar el mismo "
                      "pitcheo en dos aires no sería válido.", P),
            tabla(t, anchos=[x * inch for x in (1.5, .7, 1.6, 1.4, 1.2, .7)])])]
    else:
        out.append(Paragraph("Mecánica: no estimable con estos datos (no hay suficientes pitchers en las dos bandas).", P))
    out.append(Paragraph(al_pie + frase_mezcla(mez) + " " + frase_etiqueta(et) + " Los IC son de remuestreo de "
                         "pitchers y no incluyen diferencias de calibración entre parques.", NOTA))
    return out


def sec_modelo(D):
    v = D.val
    backend = str(D.iters.get("backend") or fila(D.res, modelo="pitching").get("backend") or "")
    out = [Paragraph("Tres clasificadores multiclase (gradient boosting por histogramas"
                     + (f", {esc(backend)}" if backend and backend != "nan" else "") + ") estiman para cada pitcheo la "
                     "probabilidad de diez resultados: bola, strike cantado, swing fallado, foul, golpe al bateador, "
                     "rodado, línea, elevado suave, elevado duro (95+ mph) y pop. El valor esperado del pitcheo y su "
                     "escala son", P),
           Paragraph("<b>xRV = Σ P(resultado) × valor(resultado)</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                     "<b>Plus = 100 + 10 × z(−xRV)</b>", FORMULA),
           Paragraph(seguro(frase_perdida, D) or "", P),
           Paragraph("Por pitcheo: Stuff+<sub>i</sub> = 100 + 10·(s<sub>i</sub> − μ<sub>t,a</sub>)/σ<sub>t</sub>, con "
                     "s = −xRV, μ<sub>t,a</sub> la media de los pitcheos de la temporada t evaluados en el aire a y "
                     "σ<sub>t</sub> la desviación de la versión neutral en t (así 100 es el promedio de la liga en esa "
                     "temporada y en ese aire, y Stuff+ neutral pone a todos los parques en el mismo aire). Entre "
                     "pitchers (pitcher-temporada j): Plus<sub>j</sub> = 100 + 10·(m<sub>j</sub> − μ)/σ, con m<sub>j</sub> el promedio "
                     "de s en sus pitcheos y μ y σ pesadas por pitcheos, y se encoge según su confiabilidad: Plus<sub>enc,j</sub> = 100 + R<sub>j</sub>·"
                     "(Plus<sub>j</sub> − 100), R<sub>j</sub> = τ²/(τ² + v<sub>j</sub>), v<sub>j</sub> = varianza de su "
                     "promedio (el juego como unidad), τ² = varianza entre pitchers menos el ruido medio; IC 95% = "
                     "Plus<sub>enc,j</sub> ± 1.96·10·√(R<sub>j</sub>·v<sub>j</sub>)/σ. Lectura: por pitcheo, 110 = una "
                     "desviación mejor que el promedio, 90 = una peor, 120 = dos (la convención de FanGraphs); entre "
                     "pitchers la desviación es la de sus promedios, así que esas cifras no se comparan con las de FanGraphs.",
                     NOTA)]
    d = v.get("densidad") or {}
    neutral, pend, rl = v.get("neutral") or {}, d.get("pendiente") or {}, d.get("rho_liga")
    f, it = D.feats, D.iters

    def tam(nm):                                             # "(16 variables, 20 árboles)", con lo que se sepa
        partes = ([f"{len(f[nm])} variables"] if f.get(nm) else []) + \
            ([f"{entero(it.get(nm))} árboles"] if fin(it.get(nm)) else [])
        return f" ({', '.join(partes)})" if partes else ""
    txt = ("El valor de cada resultado es su run value medio en la LMB. Stuff+" + tam("stuff") + " no ve la cuenta: usa "
           "el valor en la mezcla de cuentas de la liga y, para los batazos, la densidad media de la liga (o la del aire "
           "destino, en el contrafactual). Location+" + tam("location") + " y Pitching+" + tam("pitching") + " usan el "
           "valor de la cuenta del pitcheo; Pitching+ valúa además los batazos a la densidad del parque donde se lanzó. "
           "Las variables de cada modelo están en la tabla de la sección 1; el número de árboles se elige con una "
           "validación agrupada por pitcher")
    # Stuff promedia las curvas de varios cortes de pitchers: los minimos de cada corte dicen cuanto baila ese numero
    st = (it.get("detalle") or {}).get("stuff") or {}
    cortes = [entero(x) for x in st.get("mejor_por_corte") or [] if fin(x)]
    # model.py no baja de su piso de arboles (iters.json: piso): si el minimo de la validacion quedo debajo, el numero
    # usado es el piso y no "el elegido", y hay que decirlo
    usado, mejor = it.get("stuff"), st.get("mejor")
    al_piso = (f"; el mínimo de la curva promedio cae en {entero(mejor)} y se usa el mínimo permitido, {entero(usado)}"
               if fin(usado, mejor, it.get("piso")) and float(mejor) < float(usado) == float(it["piso"]) else "")
    txt += (f" (en Stuff+, promediando {len(cortes)} cortes cuyos mínimos caen en {juntar(cortes)} árboles{al_piso})."
            if len(cortes) > 1 else ".")
    clases = [("ip_gb", "Rodado"), ("ip_ld", "Línea"), ("ip_fb_soft", "Elevado suave"), ("ip_fb_hard", "Elevado duro"),
              ("ip_pu", "Pop")]
    if fin(rl) and all(fin(neutral.get(c), pend.get(c)) for c, _ in clases):
        ics = d.get("pendiente_ic") or {}
        filas = []
        for c, nom in clases:
            mar, cdmx = (neutral[c] + pend[c] * (D.rho_m - rl), neutral[c] + pend[c] * (D.rho_c - rl))
            lo, hi = par(ics, c)
            filas.append([nom, nf(mar, 3, True), nf(cdmx, 3, True), nf(cdmx - mar, 3, True),
                          f"{nf(pend[c], 2, True)} {ic(lo, hi, 2, True)}"])
        t = pd.DataFrame(filas, columns=["Batazo", f"Valor al nivel del mar (densidad {nf(D.rho_m, 3)})",
                                         f"Valor en el Harp Helú (densidad {nf(D.rho_c, 3)})",
                                         "Harp Helú menos mar", "Pendiente [IC 95%]"])
        fiable = verdadero(d.get("ic_fiable"))
        # cuantas pendientes dejan fuera el 0 y, si existe, el efecto conjunto sobre un pitcheo promedio: nada de
        # "el valor cambia con el aire" como texto fijo
        fuera = sum(lado(*par(ics, c)) in (1, -1) for c, _ in clases)
        junto, (jlo, jhi) = D.parque.get("por_valor_batazos_rv100"), par(D.parque, "por_valor_batazos_ic95_recta")
        txt += (" Para cada tipo de batazo se ajusta una recta de su valor contra la densidad del aire con "
                f"{entero(d.get('n_puntos'))} {D.nivel(2)}" + ("" if fiable else " (intervalos orientativos)")
                + f"; {fuera} de las {len(clases)} pendientes {'tiene' if fuera == 1 else 'tienen'} un intervalo que "
                "excluye 0"
                + (f", y juntas mueven el valor de un pitcheo promedio {vic(junto, jlo, jhi)} carreras por 100 del nivel "
                   "del mar al Harp Helú (sección 6)" if fin(junto, jlo, jhi) else "") + ":")
        out += [Paragraph(txt, P), tabla(t, anchos=[x * inch for x in (1.3, 1.5, 1.5, 1.2, 1.6)]),
                Paragraph("Carreras por batazo en la mezcla de cuentas de la liga (positivo = a favor del bateador). "
                          "Pendiente: carreras por unidad de densidad relativa; negativa = el batazo vale más en aire "
                          "delgado.", NOTA)]
    else:
        out.append(Paragraph(txt + f" Valor de los batazos según la densidad: {NO_ESTIMABLE.lower()}.", P))
    min_p = entero((D.ins.get("parametros") or {}).get("min_pitcheos_pitcher", config.MIN_PITCHES_PITCHER))
    # donde se entrega cada objetivo sale del encabezado del archivo, no de un texto fijo
    pc = set(D.pred_cols)
    if sum(c.startswith("p_") for c in pc - set(P_DERIVADAS)) >= 10:
        entrega = (" El archivo predicciones_por_pitcheo.csv.gz trae por pitcheo los Plus, el xRV y esas diez "
                   "probabilidades (las de Pitching+)"
                   + (", con P(swing), P(chase) y P(línea o elevado duro) ya sumadas" if set(P_DERIVADAS) <= pc else
                      "; P(swing), P(chase) y P(línea o elevado duro) se obtienen sumándolas (P(chase) pide además la "
                      "bandera de zona del dataset)")
                   + "; scores_pitcher.csv y scores_arsenal.csv traen los Plus y las tasas observadas, no "
                   "probabilidades.")
    else:
        entrega = " scores_pitcher.csv y scores_arsenal.csv traen los Plus y las tasas observadas, no probabilidades."
    out.append(Paragraph(
        "A cada pitcher lo califica el modelo de la validación cruzada que nunca lo vio. Hay dos escalas que no se "
        "comparan: <b>puntos por pitcheo</b> (z entre todos los pitcheos de la temporada; 1 punto = 1/10 de la "
        f"desviación entre pitcheos) y <b>puntos entre pitchers</b> (pitcher-temporadas con {min_p}+ pitcheos; valor "
        "encogido hacia 100 según su confiabilidad, con IC 95%). Objetivos del reto, todos de las diez probabilidades: "
        "P(swing fallado), P(strike cantado), P(rodado), P(swing) = swing fallado + foul + batazo, P(chase) = P(swing) "
        "de un pitcheo fuera de la zona, P(línea o elevado duro), un tipo de batazo que sustituye al contacto duro, y "
        f"el xRV. P(out dentro del turno), el Out+ del reto, sale de las mismas diez probabilidades (sección 4).{entrega} "
        "<b>No se modela</b> como objetivo propio el contacto débil: los batazos sin rastreo, que se descartan, acaban "
        "en out más seguido que los rastreados (sección 1), así que son sobre todo débiles y esa etiqueta saldría sesgada; "
        "del barril o el contacto duro, lo más cercano es P(línea o elevado duro). Tampoco entran la secuencia de "
        "pitcheos ni los corredores (el dataset no trae orden ni bases) ni la entrada, que el diccionario del dataset "
        "marca solo como contexto. La ubicación entra como coordenadas continuas en el plato, distancia al centro y "
        "dentro o fuera de la zona: más detallada que los 9 cuadrantes más «fuera de zona».", P))
    return out


COLS_VAL = ["Modelo", "Log-loss", "Mejora contra el nulo global [IC 95%]", "Mejora contra el nulo por cuenta [IC 95%]",
            "AUC swing fallado, dado swing", "AUC strike cantado, sin swing", "AUC rodado, dado batazo", "Corr. xRV"]
ANCHOS_VAL = (1.36, .56, 1.3, 1.3, .7, .7, .72, .46)


def _filas_validacion(res, con_nulos=True):
    """(filas, fondo vs global, fondo vs cuenta) de una tabla con las columnas de val_resumen.csv."""
    filas, f_g, f_c = [], [], []
    for clave, nom in MODELOS_VAL:
        r = fila(res, modelo=clave)
        if not r or (not con_nulos and clave.startswith("nulo")):
            continue
        celdas = []
        for ref, fondo in (("global", f_g), ("cuenta", f_c)):
            v, lo, hi = (r.get(f"mejora_vs_{ref}_pct"), r.get(f"mejora_vs_{ref}_ci_lo"), r.get(f"mejora_vs_{ref}_ci_hi"))
            if clave == f"nulo_{ref}":
                celdas.append("referencia")
                fondo.append(None)
            elif fin(v):
                d = 2 if abs(v) < 10 else 1
                celdas.append(f"{nf(v, d, True)}% {ic(lo, hi, d, True)}")
                fondo.append(color_lado(lo, hi))
            else:
                celdas.append(RAYA)
                fondo.append(None)
        filas.append([nom, nf(r.get("logloss"), 4), celdas[0], celdas[1], nf(r.get("auc_whiff_swing"), 3),
                      nf(r.get("auc_cs_take"), 3), nf(r.get("auc_gb_bip"), 3), nf(r.get("corr_xrv"), 3)])
    return filas, f_g, f_c


def sec_validacion(D):
    """Pitchers no vistos y temporada no vista en UNA tabla (mismas columnas), con una fila de titulo por prueba."""
    res, sh = D.res, D.sh
    pide(res, "val_resumen.csv", "modelo", "logloss", "mejora_vs_global_pct", "mejora_vs_cuenta_pct")
    pit = fila(res, modelo="pitching") or fila(res, modelo="stuff")
    k = D.ph["fold"].nunique() if "fold" in D.ph else None
    tope = getattr(config, "MAX_FILAS_FOLD", None)
    intro = ("Todo se mide fuera de muestra. Nulo global = frecuencia de cada resultado en la liga; nulo por cuenta = "
             "frecuencia por cuenta, el que pide el reto. Mejora = reducción de log-loss contra ese nulo, IC 95% "
             "remuestreando pitchers (verde: el intervalo queda sobre 0; rojo: bajo 0). Los scores que se entregan "
             "salen de los mismos modelos de la validación cruzada"
             + (f", cada uno entrenado con hasta {tope:,} pitcheos de los demás pitchers." if tope else "."))
    filas, f_g, f_c = _filas_validacion(res)
    if not filas:
        raise NoEstimable("out/val_resumen.csv no trae modelos")
    titulo = ParagraphStyle("vt", parent=CELDA, fontSize=7.4, leading=9)
    grupos = [("<b>Pitchers no vistos</b>: validación cruzada agrupada por pitcher"
               + (f", {k} folds" if k else "")
               + (f" ({entero(pit['n'])} pitcheos, {entero(pit['pitchers'])} pitchers)"
                  if fin(pit.get("n"), pit.get("pitchers")) else ""), filas, f_g, f_c)]
    sin_temporada = ""
    f2, g2, c2 = ([], [], [])
    if len(sh) and {"modelo", "logloss", "mejora_vs_global_pct", "mejora_vs_cuenta_pct"} <= set(sh.columns):
        f2, g2, c2 = _filas_validacion(sh, con_nulos=False)
    if f2:
        r = sh.iloc[0]
        grupos.append(("<b>Temporada no vista</b>: entrenado con las temporadas anteriores"
                       + (f" ({entero(r.get('n_train'))} pitcheos)" if fin(r.get("n_train")) else "") + ", probado en "
                       + (str(int(r["test_season"])) if fin(r.get("test_season")) else "la última")
                       + (f" ({entero(r.get('n'))} pitcheos, {entero(r.get('pitchers'))} pitchers)"
                          if fin(r.get("n"), r.get("pitchers")) else ""), f2, g2, c2))
    else:
        sin_temporada = (" Temporada no vista: no estimable con estos datos (hace falta más de una temporada y el "
                         "archivo val_season_holdout.csv).")
    cuerpo, fg, fc, extra, titulos = [], [], [], [], []
    for txt, fs, g, c in grupos:
        i = len(cuerpo) + 1                                  # +1: la fila 0 es el encabezado
        titulos.append(i)
        cuerpo.append([Paragraph(txt, titulo)] + [""] * (len(COLS_VAL) - 1))
        extra += [("SPAN", (0, i), (-1, i)), ("BACKGROUND", (0, i), (-1, i), GRIS_CLARO)]
        fg.append(None)
        fc.append(None)
        cuerpo += fs
        fg += g
        fc += c
    t = tabla(pd.DataFrame(cuerpo, columns=COLS_VAL), anchos=[x * inch for x in ANCHOS_VAL],
              fondo={COLS_VAL[2]: fg, COLS_VAL[3]: fc}, tam=7.4, extra=extra, pegadas=titulos)
    st, sc, lo, pi = (fila(res, modelo=m) for m in ("stuff", "stuff_cuenta", "location", "pitching"))
    lect = []
    if st:
        sin_cuenta = D.feats.get("stuff") and not {"balls", "strikes"} & set(D.feats["stuff"])
        debajo = lado(st.get("mejora_vs_cuenta_ci_lo"), st.get("mejora_vs_cuenta_ci_hi")) == -1
        lect.append("<b>Stuff+</b> " + frase_vs_nulo("", st, "global", corta=True) + " y "
                    + frase_vs_nulo("", st, "cuenta", corta=True)
                    + (": no ve la cuenta, y ese nulo sí" if sin_cuenta and debajo else "") + "." + frase_brier(st))
    if sc:
        lect.append("La comparación que pide el reto para un modelo que no ve la cuenta es la fila «Stuff+ con cuenta» "
                    "(las probabilidades de Stuff+ reponderadas por la frecuencia de cada resultado en la cuenta): "
                    + frase_vs_nulo("", sc, "cuenta", corta=True) + ".")
    if lo or pi:
        lect.append(juntar([frase_vs_nulo(f"<b>{n}</b>", r, "cuenta", corta=True)
                            for n, r in (("Location+", lo), ("Pitching+", pi)) if r]) + "."
                    + (f" Location+ sola alcanza el {100 * lo['mejora_vs_cuenta_pct'] / pi['mejora_vs_cuenta_pct']:.0f}% de "
                       "la mejora de Pitching+: casi todo lo que se gana sobre el nulo viene de la ubicación y la cuenta."
                       if lo and pi and fin(lo.get("mejora_vs_cuenta_pct"), pi.get("mejora_vs_cuenta_pct"))
                       and pi["mejora_vs_cuenta_pct"] > 0 and lo["mejora_vs_cuenta_pct"] >= .9 * pi["mejora_vs_cuenta_pct"]
                       and lado(pi.get("mejora_vs_cuenta_ci_lo"), pi.get("mejora_vs_cuenta_ci_hi")) == 1 else ""))
    if f2:
        pi2, sc2 = fila(sh, modelo="pitching"), fila(sh, modelo="stuff_cuenta")
        lect.append("<b>Temporada no vista</b>: " + juntar([frase_vs_nulo(n, x, "cuenta", corta=True) for n, x in
                                                            (("Pitching+", pi2), ("Stuff+ con cuenta", sc2)) if x]) + ".")
    rm = [frase_rmse(n, x, c) for n, x, c in (("Pitching+", pi, "cuenta"), ("Stuff+", st, "global")) if x]
    rm = [x for x in rm if x]
    if rm:
        sh_, ss_ = D.stab.get("split_half_run_value_observado"), D.stab.get("split_half_stuff")
        lect.append("<b>RMSE del xRV</b> contra el run value de cada pitcheo, en carreras por pitcheo (para Stuff+, el "
                    "run value llevado a la cuenta promedio), cada modelo contra el nulo de su misma valuación (IC pareado "
                    "por pitchers): " + "; ".join(rm) + ". El run value de un pitcheo es muy ruidoso, así que el RMSE baja "
                    "poco; la discriminación se ve en el log-loss y en el AUC.")
    out = [Paragraph(intro, P), t, Paragraph(" ".join(lect) + sin_temporada, P)]
    ot = seguro(frase_out_turno, D)
    if ot:
        out.append(Paragraph(ot, P))
    return out


def frase_perdida(D):
    """La funcion de perdida y como se elige el tamano del modelo, leidos del propio model.py."""
    try:
        import model as _model
        prm = _model._modelo(1).get_params()
    except (Exception, SystemExit):        # noqa: BLE001
        prm = {}
    hp = juntar([x for x in (f"tasa {prm['learning_rate']}" if "learning_rate" in prm else "",
                             f"{prm['max_leaf_nodes']} hojas" if prm.get("max_leaf_nodes") else "",
                             f"{prm['min_samples_leaf']} pitcheos mínimos por hoja" if prm.get("min_samples_leaf") else "",
                             f"L2 = {prm['l2_regularization']}" if "l2_regularization" in prm else "")])
    sh_, ss_ = D.stab.get("split_half_run_value_observado"), D.stab.get("split_half_stuff")
    return ("<b>Función de pérdida.</b> Cada clasificador minimiza la entropía cruzada multiclase (log-loss), "
            "L = −(1/N) Σ<sub>i</sub> log q<sub>i</sub>, con q<sub>i</sub> la probabilidad que el modelo da al resultado "
            "observado del pitcheo i" + (f" ({hp})" if hp else "")
            + "; el número de árboles es el que minimiza esa misma pérdida en pitchers que el modelo no vio. Se modela "
            "la probabilidad de cada resultado, y no el run value directo, porque lo que pasa después del contacto es muy "
            "ruidoso (sección 5).")


def sec_val_parque(D):
    pk = D.pk
    pide(pk, "val_park_holdout.csv", "escenario", "submodelo", "valuacion", "sesgo_por_100", "sesgo_ci_lo", "sesgo_ci_hi")
    nom_esc = {"sin_casa": f"Categoría {config.HOME_ALTITUDE} (casa) no vista",
               "sin_banda_alta": "Ningún parque de banda alta visto"}
    filas, numeros, lect = [], {v: [] for v in ("valor_neutral", "valor_nivel_mar", "valor_por_densidad")}, []
    for esc_ in list(dict.fromkeys(pk["escenario"].astype(str))):
        for sub, nsub in (("stuff", "Stuff+"), ("pitching", "Pitching+")):
            g = {v: fila(pk, escenario=esc_, submodelo=sub, valuacion=v) for v in numeros}
            base = next((x for x in g.values() if x), None)
            if base is None:
                continue
            fila_ = [celda(nom_esc.get(esc_, esc_), ParagraphStyle("pv", parent=CELDA, fontSize=7.4, leading=8.8)), nsub,
                     f"{entero(base.get('n'))} ({entero(base.get('n_juegos'))})", nf(base.get("rv_medio_real_x100"), 2, True)]
            for v in numeros:
                x = g[v]
                fila_.append(vic(x.get("sesgo_por_100"), x.get("sesgo_ci_lo"), x.get("sesgo_ci_hi")) if x else RAYA)
                numeros[v].append(x.get("sesgo_por_100", np.nan) if x else np.nan)
            filas.append(fila_)
            d, m = g["valor_por_densidad"], g["valor_nivel_mar"]
            if sub == "pitching" and d:
                txt = (f"{nom_esc.get(esc_, esc_)}: con la valuación entregada, Pitching+ "
                       + frase_sesgo(d.get("sesgo_por_100"), d.get("sesgo_ci_lo"), d.get("sesgo_ci_hi"))
                       + ("; con valores de nivel del mar "
                          + frase_sesgo(m.get("sesgo_por_100"), m.get("sesgo_ci_lo"), m.get("sesgo_ci_hi")) if m else ""))
                a, b, r = d.get("rho_train_min"), d.get("rho_train_max"), d.get("rho_test")
                if fin(a, b, r, d.get("n_puntos")):
                    txt += (f". La recta se ajustó con {entero(d['n_puntos'])} {D.nivel(2)} de densidad {nf(a, 3)} a "
                            f"{nf(b, 3)} y el parque de prueba tiene {nf(r, 3)}: "
                            + ("queda dentro de ese rango" if a <= r <= b else "queda fuera de ese rango, la recta se "
                                                                               "extrapola"))
                lect.append(txt)
    if not filas:
        raise NoEstimable("out/val_park_holdout.csv no trae escenarios")
    # sin game_id completo model.py remuestrea pitchers y lo dice en agrupado_por: n_juegos son entonces pitchers
    unidad = ("pitchers" if "agrupado_por" in pk and pk["agrupado_por"].astype(str).str.strip().eq("pitcher").any()
              else "juegos")
    cols = ["Prueba", "Modelo", f"Pitcheos de prueba ({unidad})", "RV real por 100", "Sesgo con valor neutral",
            "Sesgo con valor de nivel del mar", "Sesgo con valor por densidad (el entregado)"]
    # que incertidumbre trae el intervalo lo dice el archivo: model.py escribe sesgo_ee_valores desde que el intervalo
    # incluye el ruido de los valores de entrenamiento; un archivo sin esa columna solo remuestreo los juegos de prueba
    con_valores = "sesgo_ee_valores" in pk
    intro = ("<b>Parque no visto.</b> Se entrena sin el parque de prueba y los valores de los batazos, incluida su "
             "recta contra la densidad, se estiman solo con los parques de entrenamiento. Sesgo = carreras predichas "
             "menos reales por 100 pitcheos, "
             + (f"con un IC 95% que junta el remuestreo de los {unidad} de prueba y el de los {unidad} de entrenamiento "
                "con los que se estiman los valores." if con_valores else f"IC 95% remuestreando los {unidad} de prueba."))
    alcance = ("El intervalo incluye el ruido de muestreo de los valores y de la recta; no incluye el de reajustar el "
               "clasificador" if con_valores else "El intervalo no incluye la incertidumbre de la recta de entrenamiento")
    um = {c: (0.25, 0.75) for c in cols[4:]}
    # el "real" de cada fila es el objetivo de ESE modelo: por eso difiere entre Stuff+ y Pitching+ con los mismos pitcheos
    reales = {m: {round(float(x), 2) for x in pd.to_numeric(g["rv_medio_real_x100"], errors="coerce").dropna()}
              for m, g in pk.groupby(pk["submodelo"].astype(str))} if "rv_medio_real_x100" in pk else {}
    dos_reales = (" El RV real de las filas de Stuff+ es el run value a cuenta promedio, con el que se compara un modelo "
                  "que no ve la cuenta; el de las de Pitching+, el run value realizado."
                  if reales.get("stuff") and reales.get("pitching") and reales["stuff"] != reales["pitching"] else "")
    return [KeepTogether([
        Paragraph(intro, P),
        tabla(pd.DataFrame(filas, columns=cols), anchos=[x * inch for x in (1.6, .62, .88, .6, 1.1, 1.1, 1.2)],
              semaforo_abs=um, num={c: numeros[v] for c, v in zip(cols[4:], numeros)}, tam=7.4, izq=2)]),
            Paragraph(". ".join(lect) + ("." if lect else "") + dos_reales + " Las tres valuaciones usan las mismas "
                      f"probabilidades: solo cambia cuánto vale cada batazo. {alcance}, y los pitchers sí aparecen en "
                      "entrenamiento: lo no visto es el parque. Color por |sesgo|: verde hasta 0.25, rojo desde 0.75 "
                      "carreras por 100.", NOTA)]


def sec_calibracion(D):
    cr = D.calr
    pide(cr, "val_calibracion_resumen.csv", "objetivo", "modelo", "grupo", "ece", "pendiente")
    cab = Paragraph("<b>Calibración</b> (fuera de muestra, 10 grupos de igual tamaño): observado contra predicho.", P)
    try:
        out = [KeepTogether([cab, fig_calibracion(D)])]      # el titulo no se queda solo al pie de la pagina
    except NoEstimable as e:
        aviso(f"figura de calibracion: {e}")
        out = [cab]
    # (modelo, grupo, rotulo): las tres primeras van a la tabla; Stuff+ en banda alta solo a la lectura (no cabe otra
    # pareja de columnas), pero se dice: es el producto, y ahi estan sus pendientes mas flojas
    series = [("stuff", "todos", "Stuff+"), ("pitching", "todos", "Pitching+"),
              ("pitching", "alta", "Pitching+ en banda alta"), ("stuff", "alta", "Stuff+ en banda alta")]
    color = {"calibrada": VERDE, "exagera": ROJO, "comprime": ROJO, "no concluyente": AMBAR}
    filas, fondo, clases = [], [[] for _ in series[:3]], {s: [] for s in series}
    for obj, nom in OBJETIVOS:
        rs = [fila(cr, objetivo=obj, modelo=m, grupo=g) for m, g, _ in series]
        if not rs[0] and not rs[1]:
            continue
        base = (rs[0] or rs[1]).get("tasa_base")
        celdas = []
        for i, (s, r) in enumerate(zip(series, rs)):
            k = calibracion_de(r) if r else "sin datos"
            if fin(r.get("pendiente")):
                clases[s].append((nom, k, f"{nf(r['pendiente'])} {ic(r.get('pendiente_ci_lo'), r.get('pendiente_ci_hi'))}"))
            if i < 3:
                celdas += [(nf(100 * r["ece"], 2) + (f" ({nf(100 * r['ece_ruido'], 2)})" if fin(r.get("ece_ruido")) else "")
                            if fin(r.get("ece")) else RAYA),
                           (nf(r["pendiente"]) + (f" {ic(r['pendiente_ci_lo'], r['pendiente_ci_hi'])}"
                                                  if fin(r.get("pendiente_ci_lo"), r.get("pendiente_ci_hi")) else "")
                            if fin(r.get("pendiente")) else RAYA)]
                fondo[i].append(color.get(k))
        filas.append([nom, RAYA if obj == "xrv" else (f"{100 * base:.1f}%" if fin(base) else RAYA)] + celdas)
    if not filas:
        raise NoEstimable("out/val_calibracion_resumen.csv no trae el grupo 'todos'")
    cols = ["Objetivo", "Tasa base"] + [f"{rot}: {que}" for _, _, rot in series[:3]
                                        for que in ("ECE (piso)", "pendiente [IC 95%]")]
    out.append(tabla(pd.DataFrame(filas, columns=cols),
                     anchos=[x * inch for x in (1.4, .48, .72, 1.02, .72, 1.02, .72, 1.02)],
                     fondo={cols[3]: fondo[0], cols[5]: fondo[1], cols[7]: fondo[2]}, tam=7.2))
    lect = []
    for s in (series[1], series[0], series[3]):
        c = clases[s]
        if not c:
            continue
        n_fila = fila(cr, modelo=s[0], grupo=s[1]).get("n")
        de = {k: [f"{n} ({p_})" for n, kk, p_ in c if kk == k] for k in ("exagera", "comprime")}
        cuenta = {k: sum(kk == k for _, kk, _ in c) for k in ("calibrada", "no concluyente")}
        partes = ([f"calibrados {cuenta['calibrada']} de {len(c)} objetivos"]
                  + ([f"exagera las diferencias en {juntar(de['exagera'])}"] if de["exagera"] else [])
                  + ([f"las comprime en {juntar(de['comprime'])}"] if de["comprime"] else [])
                  + ([f"no concluyentes {cuenta['no concluyente']}"] if cuenta["no concluyente"] else []))
        lect.append(s[2] + (f" ({entero(n_fila)} pitcheos)" if s[1] != "todos" and fin(n_fila) else "") + ": "
                    + "; ".join(partes) + ".")
    sx = fila(cr, objetivo="xrv", modelo="stuff", grupo="todos")
    if calibracion_de(sx) == "exagera":
        lect.append("Una recalibración lineal del xRV de Stuff no cambiaría el orden entre pitcheos ni la escala Plus.")
    out.append(Paragraph("ECE = error medio |observado − predicho| × 100: puntos porcentuales (en xRV, carreras por 100 "
                         "pitcheos); entre paréntesis su piso, el ECE esperado por azar con un modelo bien calibrado, "
                         "que sube con menos pitcheos (banda alta). Pendiente de observado contra predicho con su IC "
                         f"95% (remuestreo de pitchers); regla: calibrado (verde) si todo el intervalo está entre "
                         f"{1 - CAL_OK:.2f} y {1 + CAL_OK:.2f}; si no, el modelo exagera las diferencias cuando el "
                         "intervalo queda bajo 1 y las comprime cuando queda sobre 1 (rojo); no concluyente si contiene "
                         "1 (ámbar). El xRV de Stuff+ se valúa a densidad media de la liga: en banda alta su nivel no "
                         "tiene por qué coincidir con lo observado (eso se juzga con Pitching+), pero su pendiente sí "
                         "se compara. " + " ".join(lect), NOTA))
    return out


def sec_contrafactual(D):
    cf = D.cf
    pide(cf, "val_contrafactual.csv", "bloque", "pt", "variable", "valor", "ic_lo", "ic_hi")
    intro = ("El contrafactual evalúa un pitcheo en otro aire: (1) su quiebre, y el de la recta de referencia de su "
             "pitcher, se llevan a la densidad destino con la β usada de la sección 2; (2) el modelo Stuff se evalúa "
             "con esas variables y los batazos se valúan a esa densidad. Velocidad, spin y punto de salida quedan "
             "fijos. Tres pruebas con los pitcher-temporada-pitcheo que tienen "
             + (f"{MIN_CELDA_CF}+ pitcheos" if MIN_CELDA_CF else "suficientes pitcheos")
             + " en banda alta y en banda mar (IC 95% remuestreando pitchers):")
    nom_var = {"stuff_plus": "Stuff+ neutral: banda alta contra banda mar",
               "stuff_plus_pendiente": "Stuff+ neutral: pendiente con todas las bandas",
               "ivb": "IVB en la dirección del quiebre (pulg)", "hb_n": "HB en la dirección del quiebre (pulg)",
               "swstr_pct": "Swing fallado %", "gb_bip_pct": "Rodados %", "rv100": "Carreras por 100"}
    nom_mod = {"pitching": "Pitching+", "stuff_transportado": "Stuff transportado"}
    todos = cf[cf["pt"].astype(str) == "TODOS"]
    chica = ParagraphStyle("cfm", parent=CELDA, fontSize=7.2, leading=8.6)
    centro = ParagraphStyle("cfl", parent=chica, alignment=1, fontSize=6.9)
    filas, fondo, lect = [], [], {}
    for bloque, titulo in (("neutralidad", "Neutralidad"), ("transporte", "Transporte"),
                           ("transporte_beta_1", "Con β = 1"), ("resultados", "Resultados")):
        for _, r in todos[todos["bloque"].astype(str) == bloque].iterrows():
            v, lo, hi = r["valor"], r["ic_lo"], r["ic_hi"]
            medida = nom_var.get(str(r["variable"]), str(r["variable"]))
            if bloque == "resultados":
                medida += " · " + nom_mod.get(str(r.get("modelo")), str(r.get("modelo")))
            if not fin(v):
                if bloque != "transporte_beta_1":            # la referencia solo se imprime si existe
                    filas.append([titulo, celda(medida, chica), RAYA, RAYA, RAYA, RAYA, RAYA, "sin datos"])
                    fondo.append(None)
                continue
            mar, alta, pred = r.get("nivel_mar"), r.get("nivel_alta"), r.get("predicho")
            if bloque in TOLERANCIA:                         # verde solo si TODO el intervalo cabe en la tolerancia
                esp, obs, d = (mar, alta, 1) if bloque == "neutralidad" else (pred, r.get("observado"), 2)
                ver = frase_aceptacion(bloque, v, lo, hi, r.get("ic_dentro_de_tolerancia"), corta=True)
                color = ROJO if abs(v) >= TOLERANCIA[bloque] else VERDE if ver == "equivalente" else AMBAR
            elif bloque == "transporte_beta_1":              # informativa: no es una prueba, no lleva color
                esp, obs, d, ver, color = pred, r.get("observado"), 2, "referencia", None
            else:
                esp, obs, d = (mar + pred if fin(mar, pred) else np.nan), alta, 2
                _, ver, color = lectura_resultado(r)
            filas.append([titulo, celda(medida, chica), f"{entero(r.get('celdas'))} ({entero(r.get('pitchers'))})",
                          nf(mar, d), nf(esp, d), nf(obs, d), vic(v, lo, hi), celda(ver, centro)])
            fondo.append(color)
            lect.setdefault(bloque, []).append((medida, v, lo, hi, r))
    if not filas:
        raise NoEstimable("out/val_contrafactual.csv no trae filas pt = TODOS")
    cols = ["Prueba", "Medida", "Celdas (pitchers)", "Nivel del mar", "En altitud: esperado", "En altitud: observado",
            "Diferencia [IC 95%]", "Lectura"]
    # un punto por pitcheo, en puntos de la escala entre pitchers (la de los rankings)
    k = pd.to_numeric(todos["factor_escala_pitcher"], errors="coerce").dropna() if "factor_escala_pitcher" in todos \
        else pd.Series(dtype=float)
    escala = f", unos {nf(TOLERANCIA['neutralidad'] * k.iloc[0], 1)} puntos entre pitchers" if len(k) else ""
    res = lect.get("resultados", [])
    dentro = list(dict.fromkeys(minuscula(m.split(" · ")[0]) for m, *_, r in res if "dentro de muestra" in nota_de(r)))
    out = [Paragraph(intro, P),
           tabla(pd.DataFrame(filas, columns=cols), anchos=[x * inch for x in (.84, 1.92, .62, .5, .6, .62, 1.1, .9)],
                 fondo={"Lectura": fondo}, izq=2, tam=7.2),
           Paragraph("Neutralidad: Stuff+ neutral del mismo pitcheo, altitud menos nivel del mar, y la pendiente con "
                     "todas las bandas llevada a esa misma diferencia; debería ser 0 (tolerancia "
                     f"{TOLERANCIA['neutralidad']:g} punto por pitcheo{escala}). Transporte: lo lanzado al nivel del mar "
                     "llevado al aire de la banda alta, menos lo observado ahí, en la dirección del quiebre de cada tipo "
                     f"(negativo = el transporte encoge de más; tolerancia {TOLERANCIA['transporte']:g} pulg)"
                     + ("; «Con β = 1» es la misma cuenta con el quiebre proporcional a la densidad, de referencia"
                        if "transporte_beta_1" in lect else "")
                     + ". Resultados: cambio entre bandas predicho menos observado (swings fallados por pitcheo, rodados "
                     "por batazo, carreras por 100 pitcheos), con Pitching+ sobre el pitcheo tal como se midió y con el "
                     "modelo Stuff sobre el pitcheo transportado, que es el contrafactual que se entrega"
                     + (f"; en {juntar(dentro)} la comprobación es dentro de muestra, porque el valor de los batazos "
                        "según la densidad se ajustó con estos mismos datos" if dentro else "")
                     + ". Lectura: «equivalente» (verde) = todo el intervalo dentro de la tolerancia; «pasa; IC hasta x» "
                     "(ámbar) = pasa por la estimación puntual, pero el intervalo no descarta diferencias de hasta x. En "
                     "resultados, «no se distinguen» (verde) = el intervalo contiene 0 y la prueba podía fallar; «sin "
                     "potencia» = tampoco habría rechazado la predicción de que el aire no mueve el pitcheo: no tiene "
                     "veredicto.", NOTA)]
    frases = []
    if res:
        con = [(m, v, lo, hi, p) for m, v, lo, hi, r in res for p in [lectura_resultado(r)[0]] if p is not None]
        mal = [f"{minuscula(m)} ({vic(v, lo, hi)}: {frase_resultado(lo, hi)})" for m, v, lo, hi, p in con if not p]
        sin = len(res) - len(con)
        if con:
            con_ic = [x for x in con if x[4] and fin(x[2], x[3])]      # solo las que 'no se distinguen'
            ancho = max(con_ic, key=lambda x: x[3] - x[2]) if con_ic else None
            txt = (f"De {len(res)} comparaciones de resultados, {len(con)} "
                   + ("podía fallar" if len(con) == 1 else "podían fallar")
                   + f": en {len(con) - len(mal)} de {len(con)} lo predicho y lo observado no se distinguen"
                   + ("; se distinguen en " + juntar(mal) if mal else "") + "."
                   + (" No distinguir no es lo mismo que coincidir: el intervalo más ancho de las que no se distinguen, "
                      f"el de {minuscula(ancho[0])}, va de {nf(ancho[2], 1, True)} a {nf(ancho[3], 1, True)}."
                      if ancho else ""))
        else:
            txt = f"Ninguna de las {len(res)} comparaciones de resultados tiene veredicto."
        # archivo anterior a con_potencia: trae filas que "pasan" sin decir si podian fallar (score.py ya no las escribe)
        viejo = any(verdadero(r.get("pasa")) and not verdadero(r.get("con_potencia")) for *_, r in res)
        if sin and viejo:
            txt += (" De las otras, el archivo no dice si podían fallar (es de una versión anterior de score.py)."
                    if con else
                    " El archivo no dice si estas pruebas podían fallar (es de una versión anterior de score.py).")
        elif sin and con:
            txt += (f" Las otras {sin} no tienen veredicto: con estas celdas habría pasado igual la predicción de que "
                    "el aire no cambia nada." if sin > 1 else
                    " La otra no tiene veredicto: con estas celdas habría pasado igual la predicción de que el aire no "
                    "cambia nada.")
        elif sin:
            txt += " Con estas celdas habría pasado igual la predicción de que el aire no cambia nada."
        frases.append(txt)
    # filas por tipo o por familia: las de etiqueta son menos robustas (la etiqueta migra con la altitud)
    otras = cf[(cf["pt"].astype(str) != "TODOS") & cf["bloque"].astype(str).isin(["neutralidad", "transporte", "resultados"])]
    otras = otras[[fin(v) for v in otras["valor"]]]
    if len(otras):
        def pasa(r):
            """True, False o None: una fila de resultados sin potencia no cuenta como que pasa."""
            b = str(r["bloque"])
            return bool(abs(r["valor"]) < TOLERANCIA[b]) if b in TOLERANCIA else lectura_resultado(r)[0]
        ver = [pasa(r) for _, r in otras.iterrows()]
        mal = otras[[p is False for p in ver]]
        n_sin = sum(p is None for p in ver)
        fam = otras["pt"].astype(str).str.startswith(("FAM_", "REPERTORIO"))
        txt = (f"Las mismas pruebas por tipo de pitcheo y por familia: {len(otras)} comparaciones ({int(fam.sum())} de "
               f"familia o repertorio, que no usan la etiqueta del tipo); pasan {sum(p is True for p in ver)}"
               + (f" y {n_sin} no {'tiene' if n_sin == 1 else 'tienen'} veredicto" if n_sin else ""))
        if len(mal):
            det = [f"{nombre_pt(r['pt'], True)}, {minuscula(nom_var.get(str(r['variable']), str(r['variable'])).split(' (')[0])}"
                   + (f" con {nom_mod.get(str(r.get('modelo')), '')}" if str(r["bloque"]) == "resultados" else "")
                   + f" ({vic(r['valor'], r['ic_lo'], r['ic_hi'])})" for _, r in mal.head(6).iterrows()]
            txt += "; no pasan " + juntar(det) + (f" y {len(mal) - 6} más" if len(mal) > 6 else "")
            n_fam = int(fam[mal.index].sum())
            txt += ((" (es una fila de familia)" if len(mal) == 1 else
                     f" ({n_fam} de las que no pasan {'es' if n_fam == 1 else 'son'} de familia)") if n_fam else
                    (" (es una fila por etiqueta de tipo, que puede migrar con la altitud; ninguna de familia falla)"
                     if len(mal) == 1 else
                     " (todas las que no pasan son por etiqueta de tipo, que puede migrar con la altitud; ninguna de "
                     "familia falla)") if int(fam.sum()) else "")
        frases.append(txt + ".")
    if frases:
        out.append(Paragraph(" ".join(frases), P))
    return out


def sec_estabilidad(D):
    s = D.stab
    if not s:
        raise NoEstimable("falta out/val_stability.json")

    def c(clave):
        return f"{nf(s.get(clave))} {ic(*par(s, clave + '_ci'))}" if fin(s.get(clave)) else RAYA
    por = "juegos" if str(s.get("split_half_por")) == "juego" else "pitcheos"
    filas = [
        [celda(f"Confiabilidad por mitades ({por} pares contra nones; {entero(s.get('split_half_pitcher_temporadas'))} "
               "pitcher-temporadas)"), c("split_half_stuff"), c("split_half_location"), c("split_half_pitching"),
         c("split_half_run_value_observado")],
        [celda("Estabilidad de un año al siguiente (r)"), c("estabilidad_stuff_plus_t_t1"), RAYA,
         c("estabilidad_pitching_plus_t_t1"), c("estabilidad_rv_observado_t_t1")],
        [celda("Predice las carreras por 100 del año siguiente (r; positivo = predice)"),
         c("predice_rv_t1__con_stuff_plus_t"), RAYA, c("predice_rv_t1__con_pitching_plus_t"),
         c("predice_rv_t1__con_rv_observado_t")],
        [celda("Diferencia contra las propias carreras al predecir"), c("dif_stuff_vs_rv"), RAYA,
         c("dif_pitching_vs_rv"), "referencia"]]
    t = pd.DataFrame(filas, columns=["Medida (nivel pitcher-temporada)", "Stuff+", "Location+", "Pitching+",
                                     "Carreras observadas"])
    cab = (f"<b>Estabilidad y poder predictivo</b>: {entero(s.get('pares'))} pares pitcher-temporada "
           f"({entero(s.get('pitchers'))} pitchers); r con IC 95% remuestreando pitchers."
           if fin(s.get("pares")) else "<b>Estabilidad y poder predictivo</b>.")
    return [KeepTogether([Paragraph(cab, P), tabla(t, anchos=[x * inch for x in (2.7, 1.1, 1.1, 1.1, 1.1)])]),
            Paragraph(frase_predictiva("Stuff+", s, "stuff") + " " + frase_predictiva("Pitching+", s, "pitching")
                      + " " + (seguro(frase_discrimina, D, defecto="") or ""), P)]


def sec_h4(D):
    ab = D.h4
    pide(ab, "val_h4_ablation.csv", "conjunto", "n_vars", "auc")
    corto = lambda k: CONJUNTO_H4.get(str(k), str(k)).split(" (")[0]      # noqa: E731
    chica = ParagraphStyle("h4", parent=CELDA, fontSize=7.0 * AJUSTE["tabla"], leading=8.4 * AJUSTE["tabla"])
    filas, fondo, fondo_spin, otra_ref, previa = [], [], [], {}, None
    for _, r in ab.iterrows():
        celdas = []
        for sufijo, colores in (("", fondo), ("_vs_spin", fondo_spin)):
            v, lo, hi = (r.get(f"delta_auc{sufijo}{c}") for c in ("", "_lo", "_hi"))
            celdas.append(vic(v, lo, hi, 3) if fin(v) else RAYA)
            colores.append(color_lado(lo, hi) if fin(v) else None)
        filas.append([celda(CONJUNTO_H4.get(str(r["conjunto"]), str(r["conjunto"])), chica),
                      entero(r["n_vars"]), nf(r["auc"], 3)] + celdas)
        ref = str(r.get("referencia"))
        if previa is not None and ref not in ("nan", "", "None", previa):     # no se compara con la fila anterior
            otra_ref.setdefault(ref, []).append(f"«{corto(r['conjunto'])}»")
        previa = str(r["conjunto"])
    c1 = ["Variables del modelo", "N", "AUC", "Δ AUC contra su referencia [IC 95%]", "Δ AUC contra tipo + spin [IC 95%]"]
    izq = tabla(pd.DataFrame(filas, columns=c1), anchos=[x * inch for x in (1.35, .24, .4, 1.2, 1.2)],
                fondo={c1[3]: fondo, c1[4]: fondo_spin}, tam=7.0)
    n = entero(ab["n"].iloc[0]) if "n" in ab else RAYA
    swings = "" if n == RAYA else f"{n} swings, "
    partes = [izq]
    imp = D.imp
    if len(imp) and {"variable", "delta_logloss"} <= set(imp.columns):
        g = imp.sort_values("delta_logloss", ascending=False).head(8)
        t2 = pd.DataFrame({"Variable (Stuff+)": [VARIABLES.get(str(v), ("", str(v)))[1] for v in g["variable"]],
                           "Δ log-loss × 1,000": [nf(1e3 * v, 2, True) for v in g["delta_logloss"]],
                           "± sd": [nf(1e3 * v, 2) for v in g["delta_logloss_sd"]] if "delta_logloss_sd" in g
                           else [RAYA] * len(g)})
        partes.append(tabla(t2, anchos=[x * inch for x in (1.4, .78, .4)], tam=7.0))
    else:
        partes.append(Paragraph(f"Importancia por permutación: {NO_ESTIMABLE.lower()}.", NOTA))
    lado_a_lado = Table([partes], colWidths=[4.47 * inch, ANCHO - 4.47 * inch],
                        style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                          ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                                          ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    refs = "; ".join(f"{juntar(quienes)}, con «{corto(ref)}»" for ref, quienes in otra_ref.items())
    return [KeepTogether([
        Paragraph("<b>¿Qué variables importan? (H4)</b> Izquierda: ablación de un clasificador de swing fallado "
                  f"dado swing ({swings}validación agrupada por pitcher) con conjuntos de variables anidados. Cada "
                  "fila se compara con la anterior" + (f" ({refs})" if refs else "") + "; esos pasos dependen del orden "
                  "en que entran las variables, y por eso cada conjunto se compara también con la misma base, tipo + "
                  "spin. Derecha: las 8 variables de "
                  "Stuff+ con mayor importancia por permutación en pitchers no vistos (cuánto sube el log-loss al "
                  "revolver la variable). El veredicto de H4 está en la sección 7.", P),
        lado_a_lado]), Spacer(1, 4)]


def sec_parques_observado(D):
    ps = D.park_scores
    pide(ps, "park_scores.csv", "grupo", "n", "stuff_plus", "pitching_plus")
    nom = {"casa": f"Casa: categoría {config.HOME_ALTITUDE}",
           "alta_sin_casa": "Otros parques de banda alta", "media": "Banda media", "mar": "Banda mar"}
    filas = []
    for _, r in ps.iterrows():
        if not fin(r["n"]) or r["n"] <= 0:
            continue
        filas.append([nom.get(str(r["grupo"]), str(r["grupo"])), f"{entero(r['n'])} ({entero(r.get('pitchers'))})",
                      nf(r["stuff_plus"], 1), nf(r.get("location_plus"), 1), nf(r["pitching_plus"], 1),
                      nf(r.get("pitching_neutral_plus"), 1), nf(r.get("xrv_x100"), 2, True), nf(r.get("rv_x100"), 2, True)]
                     + [f"{100 * r[c]:.1f}" if c in r and fin(r[c]) else RAYA
                        for c in ("swstr_pct", "gb_pct", "hard_pct", "hr_fb")])
    if not filas:
        raise NoEstimable("out/park_scores.csv sin pitcheos")
    cols = ["Dónde se lanzó", "Pitcheos (pitchers)", "Stuff+ neutral", "Location+", "Pitching+ en su parque",
            "Pitching+ neutral", "xRV por 100", "RV real por 100", "Swing fallado %", "Rodados %", "Contacto duro %",
            "Jonrón por elevado %"]
    return [tabla(pd.DataFrame(filas, columns=cols),
                  anchos=[x * inch for x in (1.2, .8, .5, .55, .6, .55, .46, .5, .5, .5, .5, .44)], tam=7.0),
            Paragraph("Scores y resultados según dónde se lanzó el pitcheo (promedios crudos en puntos por pitcheo: "
                      "incluyen quién lanza y quién batea en cada parque). Los «neutrales» evalúan todo en el mismo aire; "
                      "«Pitching+ en su parque» valúa los batazos a la densidad del parque. Swing fallado por pitcheo, "
                      "rodados por batazo, contacto duro = 95+ mph entre los batazos con velocidad medida; jonrón por elevado = "
                      "todos los jonrones entre los elevados.", NOTA)]


def sec_costo(D):
    if not D.parque:
        raise NoEstimable("falta out/altitude_park_effect.json")
    return [Paragraph("La altitud hace dos cosas distintas. <b>Costo absoluto de parque</b>, común a todos los pitchers. "
                      + frase_costo(D.parque), P),
            Paragraph("<b>Efecto relativo</b>, lo que distingue a un pitcheo de otro. En cada aire Stuff+ se centra en "
                      "el pitcheo promedio de la liga evaluado ahí (100), así que el cambio relativo = Stuff+ en el Harp "
                      "Helú menos Stuff+ al nivel del mar promedia 0 en la liga, en puntos por pitcheo.", P)]


def sec_tipo(D):
    tipo = D.tipo
    pide(tipo, "altitude_by_pitch_type.csv", "pt", "delta", "delta_lo", "delta_hi")
    mp = (D.ins.get("parametros") or {}).get("min_pitchers", config.MIN_PITCHERS_BETA)
    nota = ("Cambio relativo por tipo de pitcheo, con su IC 95% más ancho de dos: el de remuestreo de pitchers y el que "
            "hay entre los modelos de la validación cruzada. Lectura: «gana» o «pierde» (barra de color) solo si los dos "
            f"intervalos excluyen 0 y hay {entero(mp)}+ pitchers; «sin diferencia medible» si el más ancho contiene 0; "
            "«evidencia insuficiente» en otro caso. Ningún intervalo incluye la incertidumbre de las β.")
    # lo que la figura no dice: el nivel en cada aire y cuanto del cambio viene solo del movimiento
    if {"stuff_mar", "stuff_cdmx"} <= set(tipo.columns):
        det = [f"{nombre_pt(r['pt'], True)} {nf(r['stuff_mar'], 1)} → {nf(r['stuff_cdmx'], 1)}"
               + (f" ({nf(r['delta_mov'], 1, True)})" if fin(r.get("delta_mov")) else "")
               for _, r in tipo.sort_values("delta", ascending=False).iterrows() if fin(r["stuff_mar"], r["stuff_cdmx"])]
        if det:
            nota += (" Stuff+ medio al nivel del mar → en el Harp Helú (entre paréntesis, la parte del cambio que viene "
                     "solo del movimiento, con el valor de los batazos fijo): " + "; ".join(det) + ".")
    return [KeepTogether([fig_tipo(D), Paragraph(nota, NOTA)])]


def sec_mapa(D):
    img, n = fig_mapa(D)
    return [KeepTogether([img, Paragraph(
        f"Stuff+ medio por celda (100 = pitcheo promedio en ese aire) de una muestra de {n:,} pitcheos, la misma en los "
        "dos paneles: cada pitcheo se evalúa en ese aire y se dibuja con el quiebre que tendría ahí. Las siglas marcan la "
        "mediana de cada tipo (FF recta de 4 costuras, SI sinker, FC cutter, SL slider, CU curva, CH cambio, FS split).",
        NOTA)])]


def sec_formas(D):
    fo = D.formas
    pide(fo, "insights_formas.csv", "pt", "forma", "tercil", "valor_medio", "stuff_cdmx", "delta_altitud")
    filas, f_n, f_d = [], [], []
    for pt in [p for p in ORDEN_PT if (fo["pt"].astype(str) == p).any()]:
        g = fo[fo["pt"].astype(str) == pt]
        for forma in list(dict.fromkeys(g["forma"].astype(str))):
            h = {str(r["tercil"]): r for _, r in g[g["forma"].astype(str) == forma].iterrows()}
            ab = h.get("alto-bajo")
            uni = str(next(iter(h.values())).get("unidad", ""))
            d = 0 if uni == "rpm" else 1
            fila_ = [nombre_pt(pt), f"{FORMA_CORTA.get(forma, forma)}, {uni}"]
            for ter in ("bajo", "medio", "alto"):
                r = h.get(ter)
                fila_.append(f"{nf(r['valor_medio'], d)}: {nf(r['stuff_cdmx'], 1)}" if r is not None else RAYA)
            if ab is not None and fin(ab["stuff_cdmx"]):
                n_lo, n_hi = envolvente((ab.get("stuff_cdmx_lo"), ab.get("stuff_cdmx_hi")),
                                        (ab.get("stuff_cdmx_lo_folds"), ab.get("stuff_cdmx_hi_folds")))
                d_lo, d_hi = envolvente((ab.get("delta_lo"), ab.get("delta_hi")),
                                        (ab.get("delta_lo_folds"), ab.get("delta_hi_folds")))
                ns, ds = verdadero(ab.get("stuff_cdmx_sig")), verdadero(ab.get("delta_sig"))
                fila_ += [vic(ab["stuff_cdmx"], n_lo, n_hi, 1) + (" *" if ns else ""),
                          vic(ab["delta_altitud"], d_lo, d_hi, 1) + (" *" if ds else "")]
                f_n.append((VERDE if ab["stuff_cdmx"] > 0 else ROJO) if ns else None)
                f_d.append((VERDE if ab["delta_altitud"] > 0 else ROJO) if ds else None)
            else:
                fila_ += [RAYA, RAYA]
                f_n.append(None)
                f_d.append(None)
            filas.append(fila_)
    if not filas:
        raise NoEstimable("out/insights_formas.csv no trae formas")
    cols = ["Pitcheo", "Característica y unidad", "Tercil bajo", "Tercil medio", "Tercil alto",
            "Alto menos bajo: Stuff+ en el Harp Helú [IC 95%]", "Alto menos bajo: cambio por la altitud [IC 95%]"]
    n_sig = sum(c is not None for c in f_n)
    lect = (f"{n_sig} de {len(f_n)} diferencias de nivel en el Harp Helú pasan los dos intervalos (se enumeran en la "
            "recomendación «Forma dentro del pitcheo»)." if n_sig else
            "Ninguna diferencia de nivel en el Harp Helú entre tercil alto y bajo pasa los dos intervalos.")
    mf = (D.ins.get("parametros") or {}).get("min_pitcheos_forma")
    return [Paragraph("Dentro de cada tipo de pitcheo se parten los pitcheos en terciles de cada característica; el "
                      "quiebre es el llevado a la densidad media de la liga, para no confundir «mucho ride» con «lanzado "
                      "al nivel del mar». «Alto» siempre es más de lo que dice el nombre. Cada celda de tercil: valor "
                      "medio de la característica y su Stuff+ en el Harp Helú (puntos por pitcheo; 100 = pitcheo "
                      "promedio en ese aire).", P),
            tabla(pd.DataFrame(filas, columns=cols), anchos=[x * inch for x in (1.05, 1.72, .7, .7, .7, 1.13, 1.1)],
                  fondo={cols[5]: f_n, cols[6]: f_d}, izq=2, tam=7.2),
            Paragraph("IC 95%: el más ancho de pitchers y de modelos. * y color: la diferencia pasa los dos intervalos y "
                      "tiene pitchers suficientes (verde, a favor del tercil alto; rojo, en contra). «Cambio por la "
                      "altitud»: cuánto cambia esa diferencia al pasar del nivel del mar al Harp Helú. Hundimiento y "
                      "caída = IVB con el signo contrario; barrido y corte = HB hacia el lado del guante."
                      + (f" Solo tipos con {entero(mf)}+ pitcheos." if fin(mf) else "") + " " + lect, NOTA)]


def sec_hipotesis(D):
    chica = ParagraphStyle("regla", parent=CELDA, fontSize=6.9, leading=8.4, textColor=GRIS)

    def h(titulo, regla, fn, *a):
        r = seguro(fn, *a, defecto=None, nombre=titulo[:2])
        ver, txt = r if r else ("sin datos", NO_ESTIMABLE + ".")
        return [[celda(f"<b>{titulo}</b>"), Paragraph(regla, chica)],
                celda(f"<b>{VEREDICTO.get(ver, ver.capitalize())}.</b> {txt}")]
    h2, h3 = D.ins.get("H2") or {}, D.ins.get("H3") or {}

    def de_insights(d, fn):
        if not d:
            raise NoEstimable("falta out/insights.json")
        return fn(d)
    filas = [
        h("H1 · La menor densidad del aire reduce el movimiento del pitcheo.",
          "Regla: apoyada si la elasticidad del movimiento total de todos los pitcheos es positiva y su intervalo excluye "
          "0; contraria si es negativa y lo excluye.", veredicto_h1, D),
        h("H2 · " + esc(str(h2.get("hipotesis") or "Los pitcheos de diferencial de velocidad ganan valor relativo.")),
          regla_h2(h2), de_insights, h2, texto_h2),
        h("H3 · " + esc(str(h3.get("hipotesis") or "Los pitchers de rodados valen más en altitud.")),
          "Regla: decide solo la prueba dentro del mismo pitcher (su run value en banda alta menos en banda mar, contra "
          "su % de rodados): apoyada si la pendiente es negativa y su intervalo excluye 0; contraria si es positiva y "
          "lo excluye; sin datos si no hay pitchers suficientes en las dos bandas.", de_insights, h3, texto_h3),
        h("H4 · El spin por sí solo no basta.",
          "Regla: apoyada si el modelo completo sube el AUC de swing fallado frente al de tipo de pitcheo + spin con un "
          "intervalo que excluye 0; contraria si no lo sube y el spin sí suma al tipo.", veredicto_h4, D.h4)]
    return [tabla(pd.DataFrame(filas, columns=["Hipótesis y regla", "Veredicto y cifras"]),
                  anchos=[2.35 * inch, ANCHO - 2.35 * inch], izq=2, min_filas=2,
                  extra=[("VALIGN", (0, 1), (-1, -1), "TOP")])]


def sec_perfil(D):
    pf = D.perfil
    pide(pf, "insights_perfil.csv", "pt", "rasgo", "n_top", "top_mediana", "top_p25", "top_p75", "liga_mediana")
    if "control" in pf:
        ctrl = pf[[verdadero(x) for x in pf["control"]]]
        pf = pf[[not verdadero(x) for x in pf["control"]]]
    else:
        ctrl = pf.iloc[0:0]
    rasgos = [r for r in RASGO_CORTO if (pf["rasgo"].astype(str) == r).any()]
    if not rasgos:
        raise NoEstimable("out/insights_perfil.csv no trae rasgos")
    filas, distingue = [], []
    doble = ParagraphStyle("dob", parent=CELDA, fontSize=6.6, leading=8.1, alignment=1)

    def sin_miles(v, d):                                     # 2431, no 2,431: la celda es angosta
        return nf(v, d).replace(",", "")
    for pt in [p for p in ORDEN_PT if (pf["pt"].astype(str) == p).any()]:
        g = {str(r["rasgo"]): r for _, r in pf[pf["pt"].astype(str) == pt].iterrows()}
        r0 = next(iter(g.values()))
        hay_top = fin(r0.get("n_top")) and r0["n_top"] > 0
        cab = (f"<b>{nombre_pt(pt)}</b><br/>" + (f"top {entero(r0['n_top'])} de {entero(r0.get('arsenales'))}"
                                                 if hay_top else f"sin top ({entero(r0.get('arsenales'))})"))
        fila_ = [Paragraph(cab, ParagraphStyle("pc", parent=doble, alignment=0))]
        for ra in rasgos:
            r = g.get(ra)
            d = RASGO_DEC.get(ra, 1)
            if r is None:
                fila_.append(RAYA)
                continue
            liga = f"liga {sin_miles(r.get('liga_mediana'), d)}"
            if hay_top and fin(r.get("top_mediana")):
                sig = verdadero(r.get("dif_sig"))
                top = (f"{sin_miles(r['top_mediana'], d)} "
                       f"({rango(r.get('top_p25'), r.get('top_p75'), d).replace(',', '')})")
                fila_.append(Paragraph((f"<b>{top} *</b>" if sig else top) + f"<br/>{liga}", doble))
                if sig and fin(r.get("dif")):
                    lo, hi = envolvente((r.get("dif_lo"), r.get("dif_hi")), (r.get("dif_lo_folds"), r.get("dif_hi_folds")))
                    # tamano en unidades de la dispersion de la liga, para ordenar rasgos de unidades distintas
                    iqr = r["liga_p75"] - r["liga_p25"] if fin(r.get("liga_p25"), r.get("liga_p75")) else np.nan
                    distingue.append((abs(r["dif"]) / iqr if fin(iqr) and iqr > 0 else 0.0,
                                      f"{nombre_pt(pt, True)}, {minuscula(RASGO_CORTO[ra].split(' (')[0])}: "
                                      f"{'más' if r['dif'] > 0 else 'menos'} en el top ({vic(r['dif'], lo, hi, d)})"))
            else:
                fila_.append(Paragraph(liga, doble))
        filas.append(fila_)
    cols = ["Pitcheo (arsenales)"] + [RASGO_CORTO[r] for r in rasgos]
    w0 = 1.0
    peso = [{"spin": 1.2, "hb_cdmx": 1.08}.get(r, 1.0) for r in rasgos]     # las celdas con mas caracteres
    anchos = [w0 * inch] + [(ANCHO - w0 * inch) * p / sum(peso) for p in peso]
    par_ = (D.ins.get("parametros") or {})
    MAX_RASGOS = 8                                           # los mas grandes; el resto queda marcado en la tabla
    mayores = [t for _, t in sorted(distingue, key=lambda x: -x[0])[:MAX_RASGOS]]
    lect = ("* = el 25% superior se distingue del resto con los dos intervalos: " + "; ".join(mayores)
            + (f"; y {len(distingue) - MAX_RASGOS} más, marcados en la tabla" if len(distingue) > MAX_RASGOS else "") + "."
            if distingue else "Ningún rasgo separa al 25% superior del resto con los dos intervalos.")
    # el top se elige con el score del propio modelo y los rasgos son sus variables: se dice siempre, haya o no marcas
    lect += (" El top se define con el Stuff+ del modelo y estos rasgos son variables de ese modelo: la tabla describe "
             "lo que el modelo premia, no prueba que sean la causa.")
    resp = seguro(frase_arsenal_cdmx, D)
    out = ([Paragraph(resp, P)] if resp else []) + [Paragraph("Arsenales (pitcher-temporada-tipo de pitcheo)"
                     + (f" con {entero(par_.get('min_pitcheos_arsenal_perfil'))}+ pitcheos" if fin(
                         par_.get("min_pitcheos_arsenal_perfil")) else "")
                     + ": el 25% con mayor Stuff+ en el Harp Helú (el «top») contra toda la liga. En cada celda, arriba "
                     "la mediana del top y entre paréntesis sus percentiles 25 y 75; abajo, la mediana de la liga. IVB y "
                     "HB son los del pitcheo llevado al aire del Harp Helú, lo que se mediría ahí; HB y "
                     "lado de salida, + = lado del brazo (en un zurdo el dataset los trae con el signo contrario).", P),
           tabla(pd.DataFrame(filas, columns=cols), anchos=anchos, tam=6.8, min_filas=2,
                 extra=[("LEFTPADDING", (1, 0), (-1, -1), 1.5), ("RIGHTPADDING", (1, 0), (-1, -1), 1.5)])]
    nota = lect
    if fin(par_.get("min_arsenales_perfil")):
        nota += (f" «Sin top (n)»: el tipo tiene n arsenales, menos de los {entero(par_['min_arsenales_perfil'])} que "
                 "hacen falta para separar un 25% superior; solo se da la liga.")
    faltan = [nombre_pt(p, True) for p in ORDEN_PT if not (pf["pt"].astype(str) == p).any()]
    if faltan:
        nota += f" Sin arsenales suficientes: {juntar(faltan)}."
    sesgo = [nombre_pt(r["pt"], True) for _, r in ctrl.iterrows() if verdadero(r.get("dif_sig"))]
    if sesgo:
        nota += (f" Control: en {juntar(sesgo)} el top lanzó en un aire distinto al del resto, así que su perfil todavía "
                 "depende de dónde lanzó cada quien.")
    out.append(Paragraph(nota, NOTA))
    mix = D.mix
    if len(mix) and {"pt", "uso_top", "uso_resto", "dif_sig"} <= set(mix.columns):
        sig = mix[[verdadero(x) for x in mix["dif_sig"]]]
        if len(sig):
            det = {True: [], False: []}
            for _, r in sig.iterrows():
                det[bool(r["uso_top"] > r["uso_resto"])].append(
                    f"{nombre_pt(r['pt'], True)} ({100 * r['uso_top']:.0f}% contra {100 * r['uso_resto']:.0f}%)")
            partes = [f"{q} {juntar(det[k])}" for k, q in ((True, "más"), (False, "menos")) if det[k]]
            txt = ("Mezcla: los pitchers del 25% superior usan " + ", y ".join(partes)
                   + (" (cada diferencia pasa los dos intervalos)." if len(sig) > 1 else " (pasa los dos intervalos)."))
        else:
            txt = ("Mezcla: el uso de cada tipo de pitcheo no difiere de forma medible entre el 25% superior de pitchers "
                   "en el Harp Helú y el resto.")
        out.append(Paragraph(txt, NOTA))
    ficha = seguro(frase_ficha_perfil, D, pf)
    if ficha:
        out.append(Paragraph(ficha, P))
    return out


def frase_ficha_perfil(D, pf):
    """El perfil ideal en una ficha: los rasgos en que el 25% superior se distingue (dif_sig), por tipo, y por qué
    pesan distinto en altitud (betas medidas y el placebo de velocidad)."""
    partes = []
    for pt in ORDEN_PT:
        sig = np.array([verdadero(x) for x in pf["dif_sig"]], dtype=bool) if "dif_sig" in pf else np.zeros(len(pf), bool)
        g = pf[(pf["pt"].astype(str) == pt).to_numpy() & sig]
        r_ = []
        for _, r in g.iterrows():
            ra = str(r["rasgo"])
            if ra in RASGO_CORTO and fin(r.get("top_mediana"), r.get("liga_mediana")):
                d = RASGO_DEC.get(ra, 1)
                nom, _, uni = RASGO_CORTO[ra].partition(" (")
                uni = (" " + uni.rstrip(")").replace("pulg", "pulgadas")) if uni else ""
                r_.append(f"{minuscula(nom)} {nf(r['top_mediana'], d).replace(',', '')}{uni} "
                          f"(liga {nf(r['liga_mediana'], d).replace(',', '')})")
        if r_:
            partes.append(f"{nombre_pt(pt, True)}: " + ", ".join(r_[:3]))
    if not partes:
        return None
    el = D.el
    ivb, hb, ve = fila(el, var="ivb", pt="FAM_RECTAS"), fila(el, var="hb_n", pt="FAM_RECTAS"), fila(el, var="velocity", pt="TODOS")
    fis = ""
    if fin(ivb.get("beta"), hb.get("beta"), ve.get("beta")):
        fis = (f" Por qué pesa distinto en altitud: el ride y la corrida vienen de la fuerza de Magnus, que crece con la "
               f"densidad del aire (β medida del IVB {nf(ivb['beta'])} y del HB {nf(hb['beta'])} en las rectas), mientras "
               f"que la velocidad de salida de la mano no depende del aire (β del placebo de velocidad {nf(ve['beta'], 3)}); "
               "cuánto se encoge o crece en el Harp Helú la ventaja de cada forma lo da la sección 7 (columna «cambio por "
               "la altitud»).")
    return ("<b>Perfil ideal en una ficha</b> (lo que el modelo premia en el Harp Helú; mediana del 25% superior y, entre "
            "paréntesis, de la liga): " + "; ".join(partes) + "." + fis + " El dataset no trae estatura ni peso: lo físico "
            "medible es la extensión y el punto de salida.")


def sec_top(D):
    top = D.top
    pide(top, "insights_top_cdmx.csv", "pitcher_name", "season", "pitcheos", "stuff_cdmx_plus_enc", "stuff_cdmx_lo",
         "stuff_cdmx_hi")
    t = top.head(10)
    filas, fondo = [], []
    for _, r in t.iterrows():
        filas.append([esc(str(r["pitcher_name"])), str(int(r["season"])) if fin(r["season"]) else RAYA,
                      entero(r["pitcheos"]), vic(r["stuff_cdmx_plus_enc"], r["stuff_cdmx_lo"], r["stuff_cdmx_hi"], 1, False),
                      vic(r.get("stuff_plus_enc"), r.get("stuff_lo"), r.get("stuff_hi"), 1, False),
                      vic(r.get("delta_altitud"), r.get("delta_lo"), r.get("delta_hi"), 1),
                      nf(r.get("costo_cdmx_rv100"), 2, True),
                      f"{100 * r['gb_pct']:.0f}%" if fin(r.get("gb_pct")) else RAYA,
                      f"{100 * r['whiff_swing_pct']:.0f}%" if fin(r.get("whiff_swing_pct")) else RAYA])
        fondo.append(color_lado(r.get("delta_lo"), r.get("delta_hi")))
    cols = ["Pitcher", "Temp.", "Pitcheos", "Stuff+ en el Harp Helú [IC 95%]", "Stuff+ neutral [IC 95%]",
            "Efecto altitud [IC 95%]", "Costo en el Harp Helú (carreras por 100)", "Rodados por batazo",
            "Fallados por swing"]
    mt = (D.ins.get("parametros") or {}).get("min_pitcheos_top_cdmx")
    califican = int((pd.to_numeric(D.jug["pitcheos"], errors="coerce") >= mt).sum()) \
        if fin(mt) and "pitcheos" in D.jug else None
    nota = ("Puntos entre pitchers; valores encogidos hacia 100 según su confiabilidad. Efecto altitud = Stuff+ en el "
            "Harp Helú menos Stuff+ al nivel del mar (color: su intervalo queda a un lado de 0). Los IC solo incluyen "
            f"el ruido de muestreo de cada pitcher ({unidad_del_ruido(D)}), no el error del modelo")
    # si lo que se aparta un modelo de otro es tan grande como el intervalo impreso, la tabla lo dice ahi mismo
    e = error_del_modelo(D)
    grande = [f"{nf(x['error'], 1)} puntos en {nom} (su intervalo impreso equivale a {nf(x['sd_ic'], 1)})"
              for k, nom in (("stuff", "Stuff+"), ("delta", "el efecto altitud")) for x in [e.get(k)]
              if x and fin(x["sd_ic"]) and x["error"] >= x["sd_ic"]]
    nota += (", que en la medición indicativa de Limitaciones es, en desviaciones estándar, de " + juntar(grande)
             + ": estos intervalos subestiman la incertidumbre." if grande else ".")
    if califican is not None and califican <= len(t):
        nota += (f" Solo {califican} pitcher-temporadas llegan al mínimo de {entero(mt)} pitcheos: la tabla las lista "
                 "todas, sin seleccionar.")
    return [tabla(pd.DataFrame(filas, columns=cols), anchos=[x * inch for x in (1.0, .42, .55, 1.14, 1.14, 1.14, .64, .53, .54)],
                  fondo={cols[5]: fondo}, tam=7.2), Paragraph(nota, NOTA)]


def sec_recomendaciones(D):
    recs = [seguro(fn, D, nombre=fn.__name__) for fn in (rec_parque, rec_tipos, rec_riesgo_contacto, rec_formas, rec_ride, rec_rodados,
                                                         rec_mezcla, rec_pitchers, rec_scouting)]
    recs = [r for r in recs if r]
    if not recs:
        raise NoEstimable("sin resultados para recomendar")
    estilo = ParagraphStyle("rec", parent=P, spaceAfter=3)
    return [Paragraph(f"<b>{i}.</b> {x}", estilo) for i, x in enumerate(recs, 1)]


def sec_evidencia(D):
    def placebos():
        filas_pl = [(nom, uni, d, fila(D.el, var=var, pt="TODOS")) for nom, uni, d, var in
                    (("spin", "rpm", 0, "spin_rate"), ("velocidad de salida", "mph", 2, "velocity"))]
        partes = [frase_placebo(nom, r, uni, d, corta=True) for nom, uni, d, r in filas_pl if fin(r.get("beta"))]
        if not partes:
            raise NoEstimable("sin placebos")
        fiable = all(verdadero(r.get("ic_fiable")) for *_, r in filas_pl if fin(r.get("beta")))
        return ("Placebos (lo que el aire no debería mover; cifras en la sección 2): " + "; ".join(partes)
                + ("" if fiable else " (intervalos orientativos)") + ". <b>No verificado</b>: una calibración propia de "
                f"cada parque no se puede separar del aire con {'los ' + entero(D.k) if fin(D.k) else 'pocos'} niveles "
                "de densidad disponibles.")

    def transporte():
        partes, abs_, abs_1 = [], [], []
        for var, nom in (("ivb", "IVB"), ("hb_n", "HB")):
            r = fila(D.cf, bloque="transporte", pt="TODOS", variable=var)
            if fin(r.get("valor")):
                partes.append(f"{nom} {vic(r['valor'], r.get('ic_lo'), r.get('ic_hi'))} pulg, "
                              + frase_aceptacion("transporte", r["valor"], r.get("ic_lo"), r.get("ic_hi"),
                                                 r.get("ic_dentro_de_tolerancia"), corta=True))
                u = fila(D.cf, bloque="transporte_beta_1", pt="TODOS", variable=var)
                if fin(r.get("error_abs")):
                    abs_.append(f"{nf(r['error_abs'])} pulg en {nom}")
                    if fin(u.get("error_abs")):
                        abs_1.append(nf(u["error_abs"]))
        if not partes:
            raise NoEstimable("sin prueba de transporte")
        r = fila(D.cf, bloque="transporte", pt="TODOS")
        # el error medio puede ser ~0 con errores grandes de signo contrario entre celdas: se da tambien el absoluto
        absoluto = (" Error absoluto medio por celda: " + juntar(abs_)
                    + (f" (con β = 1, {juntar(abs_1)})" if len(abs_1) == len(abs_) else "") + ".") if abs_ else ""
        return ("Error medio, en la dirección del quiebre, al llevar a la banda alta lo lanzado al nivel del mar: "
                + "; ".join(partes) + f" (tolerancia {TOLERANCIA['transporte']:g} pulg; {entero(r.get('celdas'))} celdas, "
                f"{entero(r.get('pitchers'))} pitchers)." + absoluto + " Es una comprobación de ajuste: las celdas son "
                "las mismas con las que se estimó β.")

    def neutralidad():
        r = (fila(D.cf, bloque="neutralidad", pt="TODOS", variable="stuff_plus")
             or fila(D.cf, bloque="neutralidad", pt="TODOS"))
        if not fin(r.get("valor")):
            raise NoEstimable("sin prueba de neutralidad")
        txt = (f"Mismo pitcheo en banda alta y en banda mar: {vic(r['valor'], r.get('ic_lo'), r.get('ic_hi'))} puntos "
               "por pitcheo, " + frase_aceptacion("neutralidad", r["valor"], r.get("ic_lo"), r.get("ic_hi"),
                                                  r.get("ic_dentro_de_tolerancia")) + ".")
        p = fila(D.cf, bloque="neutralidad", pt="TODOS", variable="stuff_plus_pendiente")
        if fin(p.get("valor")):
            txt += (f" Con todas las bandas (pendiente): {vic(p['valor'], p.get('ic_lo'), p.get('ic_hi'))}, "
                    + frase_aceptacion("neutralidad", p["valor"], p.get("ic_lo"), p.get("ic_hi"),
                                       p.get("ic_dentro_de_tolerancia"), corta=True) + ".")
        if fin(r.get("factor_escala_pitcher")):
            txt += (f" Un punto por pitcheo son {nf(r['factor_escala_pitcher'], 1)} puntos entre pitchers, la escala de "
                    "los rankings.")
        e = fila(D.cf, bloque="entre_pitchers")
        return txt + (" Entre pitchers distintos: ver Limitaciones." if fin(e.get("valor")) else
                      " <b>No verificado</b> entre pitchers distintos.")

    def mecanica():
        a = D.adapt
        pide(a, "adaptacion.csv", "tipo", "variable", "diferencia", "ic_lo", "ic_hi")
        mec = a[a["tipo"].astype(str).str.startswith("Mec")]
        if not len(mec):
            raise NoEstimable("sin filas de mecánica")
        # rpm sin decimales, como en la tabla de la seccion 3; una fila que es la misma cantidad que un placebo lo dice
        cam = [f"{minuscula(r['variable'])} {frase_cambia(r)} "
               f"({vic(r['diferencia'], r['ic_lo'], r['ic_hi'], 0 if str(r.get('unidad')) == 'rpm' else 2)} "
               f"{r.get('unidad', '')}" + ("; misma cantidad que su placebo" if nota_de(r) else "") + ")"
               for _, r in mec.iterrows() if lado(r["ic_lo"], r["ic_hi"]) not in (0, None)]
        n = entero(pd.to_numeric(mec["pitchers"], errors="coerce").max()) if "pitchers" in mec else RAYA
        return ((f"De {len(mec)} variables de mecánica, {len(cam)} {'cambia' if len(cam) == 1 else 'cambian'} de forma "
                 "medible: " + "; ".join(cam) + f" ({n} pitchers). "
                 + ("El contrafactual no la mueve." if len(cam) == 1 else "El contrafactual no las mueve.")) if cam else
                f"Ninguna de las {len(mec)} variables de mecánica (velocidad, spin, extensión, punto de salida) cambia de "
                f"forma medible entre bandas ({n} pitchers).")

    def etiqueta():
        et = D.adapt[D.adapt["tipo"].astype(str) == "Etiqueta"] if "tipo" in D.adapt else pd.DataFrame()
        if not len(et):
            return "<b>No verificado</b>: la estabilidad de la etiqueta no es estimable con estos datos."
        migra = any(lado(r.get("ic_lo"), r.get("ic_hi")) not in (0, None) for _, r in et.iterrows())
        return (("<b>No se cumple</b>: " if migra else "No se detecta un cambio: ")
                + frase_etiqueta(et).split(": ", 1)[-1])

    def batazos():
        pide(D.pk, "val_park_holdout.csv", "escenario", "submodelo", "valuacion", "sesgo_por_100")
        partes = []
        for esc_ in dict.fromkeys(D.pk["escenario"].astype(str)):
            r = fila(D.pk, escenario=esc_, submodelo="pitching", valuacion="valor_por_densidad")
            if r:
                partes.append(("con el parque de casa" if esc_ == "sin_casa" else "con toda la banda alta")
                              + " fuera del entrenamiento, Pitching+ "
                              + frase_sesgo(r.get("sesgo_por_100"), r.get("sesgo_ci_lo"), r.get("sesgo_ci_hi")))
        if not partes:
            raise NoEstimable("sin validación por parque")
        d = (D.val.get("densidad") or {})
        txt = "; ".join(partes)
        return (txt[0].upper() + txt[1:] + ". <b>No verificado</b>: la linealidad misma (la recta entregada sale de "
                f"{entero(d.get('n_puntos'))} puntos de densidad).")

    def aire():
        est = [esc(e) for e, o in origen_densidad(D).items() if o == "estandar"]
        return (f"<b>No verificado</b> en modo categoría: se usa una densidad media por categoría ({entero(D.k)} "
                "categorías) y dentro de una categoría los parques difieren."
                + (f" La de {juntar(est)} no es una medición: es la atmósfera estándar corregida por el clima típico."
                   if est else ""))

    items = [("El cambio de movimiento viene del aire y no de la medición de cada parque", placebos),
             ("La β usada reproduce el movimiento en altitud", transporte),
             ("El Stuff+ neutral no depende de dónde se lanzó el pitcheo", neutralidad),
             ("El pitcher lanza igual en altitud (el contrafactual deja fijos velocidad, spin y punto de salida)", mecanica),
             ("La etiqueta del tipo de pitcheo no cambia con la altitud", etiqueta),
             ("El valor de los batazos cambia en línea recta con la densidad", batazos),
             ("La densidad asignada es la del aire del juego", aire)]
    chica = ParagraphStyle("ev", parent=CELDA, fontSize=7.5, leading=9.1)
    filas = [[celda(s, chica), celda(seguro(fn, defecto=NO_ESTIMABLE + ".", nombre="evidencia: " + fn.__name__), chica)]
             for s, fn in items]
    return [tabla(pd.DataFrame(filas, columns=["Supuesto", "Evidencia"]), anchos=[2.3 * inch, ANCHO - 2.3 * inch], izq=2,
                  tam=7.5, min_filas=3, extra=[("VALIGN", (0, 1), (-1, -1), "TOP")])]


def frase_entre_pitchers(cf):
    """ADENDA 2.e: lo que las pruebas dentro de la misma pitcher-temporada no ven, con las dos filas informativas
    del bloque entre_pitchers de val_contrafactual.csv:
      corr_stuff_plus_densidad         correlacion entre el Stuff+ neutral de una pitcher-temporada y la densidad
                                       media del aire donde lanzo
      cambio_stuff_plus_por_densidad   el MISMO pitcher de una temporada a la siguiente: puntos (entre pitchers) que
                                       cambia su Stuff+ por cada +0.1 de esa densidad
    Si puede ser composicion (quien lanza en altitud) o no lo dicen esa segunda fila y las pruebas de neutralidad,
    no un texto fijo. None si no hay correlacion estimada."""
    e = fila(cf, bloque="entre_pitchers", variable="corr_stuff_plus_densidad") or fila(cf, bloque="entre_pitchers")
    v, lo, hi = e.get("valor"), e.get("ic_lo"), e.get("ic_hi")
    if not fin(v):
        return None
    n = e.get("celdas") if fin(e.get("celdas")) else e.get("pitchers")
    cifra = f"r = {vic(v, lo, hi)}" + (f", {entero(n)} pitcher-temporadas" if fin(n) else "")
    s = lado(lo, hi)
    if s is None:
        return f"Entre pitchers, el Stuff+ neutral y la densidad media del aire donde lanzó cada uno: {cifra}."
    if s == 0:
        txt = ("Entre pitchers, el Stuff+ neutral no se relaciona de forma medible con la densidad media del aire donde "
               f"lanzó cada uno ({cifra}).")
        # la segunda fila puede decir otra cosa aunque la primera no se distinga de 0: se dice, no se calla
        c = fila(cf, bloque="entre_pitchers", variable="cambio_stuff_plus_por_densidad")
        sc = lado(c.get("ic_lo"), c.get("ic_hi")) if fin(c.get("valor")) else None
        if sc not in (None, 0):
            txt += (" En el mismo pitcher, de una temporada a la siguiente, sí aparece un cambio pequeño ("
                    f"{vic(c['valor'], c['ic_lo'], c['ic_hi'], 1)} puntos entre pitchers por cada +0.1 de densidad media"
                    + (f", {entero(c.get('celdas'))} pares de temporadas" if fin(c.get("celdas")) else "")
                    + "): su Stuff+ neutral es más " + ("alto" if sc < 0 else "bajo") + " cuando lanza en aire más "
                    "delgado. Esa comparación junta el aire con lo que cambia un pitcher de un año a otro (equipo, rol, "
                    "edad); dentro de la misma temporada la prueba de neutralidad es la de la sección 5.")
        return txt
    txt = ("Entre pitchers, el Stuff+ neutral es más " + ("alto" if s < 0 else "bajo") + " en quienes lanzaron en aire "
           f"más delgado ({cifra}).")
    c = fila(cf, bloque="entre_pitchers", variable="cambio_stuff_plus_por_densidad")
    sc = lado(c.get("ic_lo"), c.get("ic_hi")) if fin(c.get("valor")) else None
    mismo = sc is not None and sc == s                       # el mismo sentido, con un intervalo que excluye 0
    if sc is not None:
        cifra_c = (f"{vic(c['valor'], c['ic_lo'], c['ic_hi'], 1)} puntos entre pitchers por cada +0.1 de densidad media"
                   + (f", {entero(c.get('celdas'))} pares de temporadas" if fin(c.get("celdas")) else ""))
        txt += (f" El patrón también aparece en el mismo pitcher de una temporada a la siguiente ({cifra_c}): no es solo "
                "composición (quién lanza en altitud)." if mismo else
                f" En el mismo pitcher, de una temporada a la siguiente, el cambio va en sentido contrario ({cifra_c})."
                if sc != 0 else
                f" En el mismo pitcher, de una temporada a la siguiente, no se distingue un cambio ({cifra_c}).")
    neu = [frase_aceptacion("neutralidad", r["valor"], r.get("ic_lo"), r.get("ic_hi"), r.get("ic_dentro_de_tolerancia"),
                            corta=True)
           for r in (fila(cf, bloque="neutralidad", pt="TODOS", variable=x) for x in ("stuff_plus", "stuff_plus_pendiente"))
           if fin(r.get("valor"))]
    donde = "Dentro de la misma temporada" if sc is not None else "Dentro del mismo pitcher"
    if not neu:
        return txt + " No hay prueba de neutralidad con que separar composición de un defecto del transporte."
    if "fuera de tolerancia" in neu:
        return txt + f" {donde} la neutralidad tampoco pasa: el transporte deja parte del efecto del parque."
    if all(x == "equivalente" for x in neu):
        return txt + (f" {donde} la neutralidad sí es equivalente dentro de la tolerancia." if mismo else
                      f" {donde} la neutralidad es equivalente dentro de la tolerancia: puede ser composición (quién "
                      "lanza en altitud) y no un defecto del transporte; no se pudo separar.")
    n_eq = sum(x == "equivalente" for x in neu)              # la tabla de evidencia da las dos: aqui no se dice menos
    solo = ("pasa solo por la estimación puntual" if not n_eq else
            f"es equivalente en {n_eq} de sus {len(neu)} pruebas y en la otra pasa solo por la estimación puntual"
            if len(neu) - n_eq == 1 else
            f"pasa solo por la estimación puntual en {len(neu) - n_eq} de sus {len(neu)} pruebas")
    return txt + (f" {donde} la neutralidad {solo}: no se descarta que el transporte deje "
                  "parte del efecto del parque." if mismo else
                  f" {donde} la neutralidad {solo}: puede ser composición (quién lanza en "
                  "altitud), pero no se descarta un defecto del transporte.")


def sec_limitaciones(D):
    def etiqueta():
        et = D.adapt[D.adapt["tipo"].astype(str) == "Etiqueta"] if "tipo" in D.adapt else pd.DataFrame()
        dato = frase_etiqueta(et).split(": ", 1)[-1] if len(et) else ""
        dato = (" En estos datos, " + dato) if dato and not dato.startswith("no estimable") else \
            " Con estos datos no se pudo medir cuánto migra."
        # una fila de familia solo aguanta el cambio de etiqueta DENTRO de la familia, y hay tipos fuera de todas
        fuera = [f"{nombre_pt(p, articulo=True)} no entra en la media de ninguna familia"
                 + (f" y, cuando le faltan pitchers, usa la β de {nombre_pt(FAMILIA_DE[p], articulo=True)}"
                    if p in FAMILIA_DE else "")
                 for p in ORDEN_PT if not any(p in tipos for tipos in FAMILIAS.values())]
        return ("El dataset solo trae la etiqueta automática del pitcheo (AutoPitchType), que clasifica por movimiento: "
                "la etiqueta puede migrar con la altitud, y el contrafactual la deja fija." + dato + " Por eso las "
                "comparaciones por tipo son menos robustas que las de familia (rectas, quiebres, cambios), que no usan "
                "la etiqueta. Una fila de familia solo aguanta un cambio "
                "de etiqueta dentro de la familia" + ("; " + juntar(fuera) if fuera else "") + ".")

    def arrastre():
        vl = fila(D.el, var="velo_loss", pt="TODOS")
        dato = (f" (β del arrastre {vic(vl['beta'], vl.get('ci_lo'), vl.get('ci_hi'), 2, False)}"
                f"{orient(vl.get('ic_fiable'))})" if fin(vl.get("beta")) else "")
        return ("No se modela el efecto del menor arrastre sobre la velocidad de llegada al plato" + dato
                + ": el contrafactual no mueve la velocidad.")

    def modo():
        est = [esc(e) for e, o in origen_densidad(D).items() if o == "estandar"]
        return (f"Modo categoría: una sola densidad por categoría de altitud"
                + (f" (la de {juntar(est)} sale de la atmósfera estándar, no de una medición)" if est else "")
                + f", y «casa» junta todos los parques de la categoría {config.HOME_ALTITUDE}; los intervalos de la "
                f"física y de la recta de los batazos salen de {entero(D.k)} niveles de densidad y son orientativos.")

    def beta():
        r = fila(D.el, var="ivb", pt="FAM_RECTAS") or fila(D.el, var="ivb", pt="FF")
        if not fin(r.get("beta")):
            return None
        return (f"Las β medidas (IVB de las rectas {nf(r['beta'])}; {frase_beta_vs_1(r.get('ci_lo'), r.get('ci_hi'))}) "
                "se usan tal como salen de los datos, sin un mecanismo propuesto (sección 2).")

    def entre():
        return frase_entre_pitchers(D.cf)

    def calibracion():
        sx = fila(D.calr, objetivo="xrv", modelo="stuff", grupo="todos")
        k = calibracion_de(sx)
        if k not in ("exagera", "comprime"):
            return None
        return (f"El xRV del modelo Stuff {k} las diferencias fuera de muestra (pendiente {nf(sx['pendiente'])} "
                f"{ic(sx.get('pendiente_ci_lo'), sx.get('pendiente_ci_hi'))}): sus cifras absolutas en carreras son "
                "aproximadas.")

    def modelo():
        """Los intervalos por pitcher no traen el error del modelo: se dice cuanto midio score.acuerdo_modelos."""
        base = ("Los intervalos por pitcher, por arsenal y del perfil solo miden el ruido de muestreo "
                f"({unidad_del_ruido(D)}), no el error del modelo")
        e = error_del_modelo(D)
        if not e:
            return base + "; el desacuerdo entre modelos solo se midió por tipo y por forma."
        nom = {"stuff": "Stuff+", "delta": "el efecto altitud"}
        cada = [f"{nf(x['error'], 1)} puntos entre pitchers en {nom[k]}"
                + (f" (r = {nf(x['acuerdo'])} entre modelos" if fin(x["acuerdo"]) else " (")
                + (f"; el intervalo impreso, de 95%, equivale a {nf(x['sd_ic'], 1)})" if fin(x["sd_ic"]) else ")")
                for k, x in e.items()]
        mayor = any(fin(x["sd_ic"]) and x["error"] >= x["sd_ic"] for x in e.values())
        des = [f"{nf(x['rango'], 1)} puntos en {nom[k]}" for k, x in e.items() if fin(x["rango"])]
        return (base + ". Medición indicativa de ese error: con los "
                f"{entero(D.stab.get('acuerdo_modelos_k'))} modelos de la validación cruzada calificando las mismas "
                f"{entero(D.stab.get('acuerdo_modelos_pitcher_temporadas'))} pitcher-temporadas, el error típico de un "
                f"modelo (una desviación estándar) es de {juntar(cada)}"
                + (": los intervalos impresos subestiman la incertidumbre." if mayor else ".")
                + (f" Además el nivel medio de dos modelos llega a diferir {juntar(des)}: una diferencia de ese tamaño "
                   "entre pitchers calificados por modelos distintos es del modelo, no del pitcher." if des else ""))
    items = [seguro(fn, defecto=None, nombre="limitacion: " + fn.__name__) for fn in (etiqueta, arrastre, modo)]
    items += ["El valor de los batazos se supone lineal en la densidad del aire. El efecto medido incluye todo lo que "
              "distingue a esos parques, también cómo se guardan las pelotas: en Coors el humidor bajó los jonrones 25% "
              "(Nathan, 2011).",
              "El dataset no trae orden de pitcheos ni corredores. P(out dentro del turno) se arma con la cadena de "
              "cuentas de la liga y no ve corredores ni outs. La cadena de los pesos en carreras supone que la "
              "frecuencia de cada evento no depende de bases ni outs, ajusta el avance de corredores a promedios del "
              "dataset y no modela robos ni lanzamientos descontrolados."]
    items += [seguro(fn, defecto=None, nombre="limitacion: " + fn.__name__) for fn in (beta, entre, calibracion, modelo)]
    items.append("Ningún intervalo de Stuff+ incluye la incertidumbre de las β. El simulador del tablero usa otro modelo "
                 "(ajustado con todos los datos) que las tablas.")
    if D.rapido:
        items.append("Esta es una corrida de prueba con 30% de los pitchers: ninguna cifra es definitiva.")
    items = [x for x in items if x]
    return [Paragraph(" ".join(f"<b>({i})</b> {x}" for i, x in enumerate(items, 1)),
                      ParagraphStyle("lim", parent=P, fontSize=P.fontSize - .4, leading=P.leading - .6))]


# ── armado ──────────────────────────────────────────────────────────────────
def bloque(nombre, fn, D, titulo=None, estilo=H2, antes=None):
    """Un bloque del documento. Si no se puede escribir, deja su titulo y una linea 'No estimable con estos datos'."""
    cab = ([antes] if antes is not None else []) + ([Paragraph(titulo, estilo)] if titulo else [])
    try:
        cuerpo = fn(D)
        if not cuerpo:
            raise NoEstimable("sin contenido")
    except NoEstimable as e:
        aviso(f"bloque '{nombre}': no estimable ({e})")
        cuerpo = [Paragraph(f"{NO_ESTIMABLE} ({esc(str(e))}).", P)]
    except Exception as e:                 # noqa: BLE001  (un archivo de otra version no debe tumbar el documento)
        AVISOS.append(nombre)
        aviso(f"bloque '{nombre}': fallo inesperado ({type(e).__name__}: {e}); se imprime '{NO_ESTIMABLE}'", grave=True)
        cuerpo = [Paragraph(f"{NO_ESTIMABLE}.", P)]
    return cab + cuerpo


def historia(D):
    s = []
    # salto(h): si quedan menos de h pulgadas el bloque empieza en la pagina siguiente. En las secciones que abren
    # con una tabla, h cubre titulo + parrafo de entrada + encabezado + las primeras filas: ni el titulo ni el
    # parrafo se quedan solos al pie con su tabla en la pagina siguiente. Se encoge con el nivel (AJUSTE["saltos"]).
    salto = lambda h: CondPageBreak(h * inch * AJUSTE["saltos"])     # noqa: E731
    s += bloque("titulo", sec_titulo, D)
    s += bloque("resumen", sec_resumen, D, "Resumen ejecutivo")
    s += bloque("marco", sec_marco, D, "Marco teórico", H2)
    s += bloque("datos", sec_datos, D, "1. Datos y variables", H1)
    s += bloque("variables", sec_variables, D)
    # H2T va antes de una tabla larga que SI puede partirse entre paginas (con keepWithNext el titulo arrastraria
    # la tabla entera)
    s += bloque("fisica", sec_fisica_intro, D, "2. Lo que la altitud le hace a un pitcheo", H1, salto(1.15))
    s += bloque("figura de parques", sec_fisica_figura, D)
    s += bloque("elasticidades", sec_fisica_tabla, D)
    s += bloque("controles", sec_controles, D)
    s += bloque("lectura", sec_lectura, D)
    s += bloque("aire", sec_aire, D, "3. Aire, adaptación y modelo", H1, salto(1.25))
    s += bloque("adaptacion", sec_adaptacion, D, "¿Cambian los pitchers en altitud?", H2, salto(1.0))
    s += bloque("modelo", sec_modelo, D, "El modelo", H2, salto(1.0))
    s += bloque("validacion", sec_validacion, D, "4. Validación", H1, salto(2.3))
    s += bloque("validacion por parque", sec_val_parque, D)
    s += bloque("calibracion", sec_calibracion, D)
    s += bloque("contrafactual", sec_contrafactual, D, "5. ¿Es válido el contrafactual de altitud?", H1, salto(2.2))
    s += bloque("estabilidad", sec_estabilidad, D)
    s += bloque("h4", sec_h4, D)
    s += bloque("costo de parque", sec_costo, D, "6. Qué gana y qué pierde en el Harp Helú", H1, salto(1.5))
    s += bloque("efecto por tipo", sec_tipo, D)
    s += bloque("scores por parque", sec_parques_observado, D)
    s += bloque("mapa", sec_mapa, D)
    s += bloque("formas", sec_formas, D, "7. Formas dentro de cada pitcheo e hipótesis del reto", H1, salto(2.2))
    s += bloque("hipotesis", sec_hipotesis, D, "Hipótesis del reto", H2T, salto(2.6))
    s += bloque("perfil", sec_perfil, D, "8. Perfil físico ideal para el Harp Helú", H1, salto(2.1))
    s += bloque("top", sec_top, D, "Pitcher-temporadas con mayor Stuff+ en el Harp Helú", H2T, salto(1.9))
    s += bloque("recomendaciones", sec_recomendaciones, D, "9. Recomendaciones, evidencia y limitaciones", H1, salto(1.3))
    s += bloque("evidencia", sec_evidencia, D, "Evidencia de cada supuesto", H2T, salto(2.2))
    s += bloque("limitaciones", sec_limitaciones, D, "Limitaciones", H2, salto(1.0))
    s += bloque("referencias", sec_referencias, D, "Referencias", H2)
    return s


class Documento(SimpleDocTemplate):
    """SimpleDocTemplate que anota cuanto espacio queda en cada pagina (diagnostico con REPORT_DEBUG=1: sirve
    para ver si el numero de paginas esta inflado con huecos)."""

    def afterFlowable(self, flowable):
        libre = getattr(self, "libre", None)
        if libre is None:
            libre = self.libre = {}
        try:
            libre[self.page] = (self.frame._y - self.frame._y1p) / inch
            que = type(flowable).__name__
            if que == "Paragraph":
                que = flowable.getPlainText()[:28] if flowable.style.name.startswith("h") else "p"
            self.__dict__.setdefault("orden", []).append((self.page, que, libre[self.page]))
        except Exception:                  # noqa: BLE001  (detalle interno de reportlab: si cambia, no hay diagnostico)
            pass


def destino():
    """Si el PDF esta abierto en un visor (Windows lo bloquea) se escribe con otro nombre, y se avisa."""
    path = config.OUT / "documento_tecnico.pdf"
    try:
        with open(path, "ab"):
            pass
        return path
    except OSError:
        otro = config.OUT / "documento_tecnico_nuevo.pdf"
        aviso(f"no se puede escribir {path.name} (esta abierto en otro programa?): se escribe {otro.name}", grave=True)
        return otro


def main():
    t0 = time.time()
    config.OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9})
    D = Datos()
    path = destino()
    autor = config.EQUIPO if config.EQUIPO not in PLACEHOLDERS else "Stuff+ LMB"
    pie_txt = "Stuff+ LMB · altitud · " + (f"código {D.hash}" if D.hash else "código sin registro")
    if D.rapido:
        pie_txt += " · " + MARCA_RAPIDA
    if D.otra_version:
        pie_txt += " · validación de otra versión del código"

    def pie(c, d):
        c.setFont("Helvetica", 7)
        c.setFillColor(GRIS)
        c.drawString(MARGEN, .36 * inch, pie_txt)
        c.drawRightString(letter[0] - MARGEN, .36 * inch, str(d.page))
    depura = os.environ.get("REPORT_DEBUG") == "1"           # diagnostico: PDF sin comprimir y hueco de cada pagina

    def armar(nivel):
        """Escribe el PDF con ese nivel de compactacion y devuelve cuantas paginas salieron."""
        compactar(nivel)
        del AVISOS[:]
        doc = Documento(str(path), pagesize=letter, leftMargin=MARGEN, rightMargin=MARGEN,
                        topMargin=.55 * inch, bottomMargin=.62 * inch,
                        title="Stuff+ LMB calibrado por altitud", author=str(autor),
                        **({"pageCompression": 0} if depura else {}))
        doc.build(historia(D), onFirstPage=pie, onLaterPages=pie)
        if depura:
            print(f"nivel {nivel}: {doc.page} paginas; pulgadas libres al pie de cada una: "
                  + ", ".join(f"{p}: {v:.1f}" for p, v in sorted(getattr(doc, "libre", {}).items())))
            for p, que, v in getattr(doc, "orden", []):
                if que != "p":
                    print(f"   pag {p}  quedan {v:4.1f}  {que}".encode("ascii", "replace").decode("ascii"))
        return doc.page

    # Se arma en tamano normal; si pasa de la meta se prueba cada nivel mas compacto hasta alcanzarla. Si ninguno
    # la alcanza queda el menos compacto que cabe en el rango del reto (o, si ninguno cabe, el mas compacto).
    paginas = {}
    nivel = 0
    for nivel in range(len(NIVELES)):
        paginas[nivel] = armar(nivel)
        if paginas[nivel] <= META_PAGINAS:
            break
    else:
        caben = [k for k, v in paginas.items() if v <= PAGINAS[1]]
        elegido = min(caben) if caben else max(paginas)
        if elegido != nivel:
            nivel = elegido
            paginas[nivel] = armar(nivel)
    n = paginas[nivel]
    if len(paginas) > 1:
        aviso("paginas segun el nivel de compactacion: " + ", ".join(f"nivel {k}: {v}" for k, v in sorted(paginas.items()))
              + f"; queda el nivel {nivel} (meta: {META_PAGINAS} paginas; el reto acepta de {PAGINAS[0]} a {PAGINAS[1]})")
    compactar(0)
    if path.name == "documento_tecnico.pdf":   # un "_nuevo" de una corrida anterior ya no es el documento nuevo
        try:
            (config.OUT / "documento_tecnico_nuevo.pdf").unlink()
        except OSError:
            pass
    print(f"documento: {str(path).encode('ascii', 'replace').decode('ascii')} - {n} paginas - "
          f"{'CORRIDA DE PRUEBA' if D.rapido else 'corrida completa'}"
          + (f" - compactado (nivel {nivel})" if nivel else "") + f" - {time.time() - t0:.0f} s")
    graves = []
    if not PAGINAS[0] <= n <= PAGINAS[1]:
        graves.append(f"el documento tiene {n} paginas y el reto pide de {PAGINAS[0]} a {PAGINAS[1]}")
    if AVISOS:
        graves.append("bloques que no se pudieron escribir por un fallo inesperado: " + ", ".join(dict.fromkeys(AVISOS)))
    if graves:
        print("=" * 68)
        for g in graves:
            aviso(g, grave=True)
        print("=" * 68)
        if config.ESTRICTO:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
