"""
DASHBOARD — Stuff+ LMB calibrado por altitud.

Correr desde la carpeta del proyecto:    python -m streamlit run dashboard/app.py

Pestanas:
  1. Simulador            promedio de un pitcheo -> Stuff+ en el Harp Helu y al nivel del mar
  2. Mapa de movimiento   que formas rinden mas en cada aire, con el pitcheo simulado encima
  3. Ranking              Stuff+ de los pitchers y de los arsenales de la liga
  4. Altitud por pitcheo  que tipos ganan o pierden en el Harp Helu, con su intervalo
  5. Prospecto            arsenal completo -> su lugar entre los pitchers de la liga

DOS ESCALAS que no se comparan entre si (cada tabla dice cual usa):
  puntos por pitcheo      100 = pitcheo promedio de la liga en ese aire; 10 puntos = una
                          desviacion estandar entre PITCHEOS (la de las tablas)
  puntos entre pitchers   100 = pitcher promedio; 10 puntos = una desviacion estandar
                          entre PITCHERS (varias veces mas chica que la de arriba)

Ninguna frase que afirme una direccion ("gana", "pierde") o una equivalencia
("comparable") esta escrita a mano: sale de un numero de la corrida con su
intervalo o su medida (lectura_tipo, lectura_parque, lectura_recta,
lectura_ranking).
"""
import json
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure      # sin pyplot: no deja figuras abiertas entre una interaccion y otra
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config
import simulator as sim
from config import CDMX_M, SEA_LEVEL_M, CLIMA_TIPICO, RHO_CDMX, RHO_MAR, RANDOM_STATE, STADIUMS
from features import FASTBALLS

st.set_page_config(page_title="Stuff+ LMB · Altitud", layout="wide")

NOMBRE = {"FF": "Recta de 4 costuras", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
          "CU": "Curva", "CH": "Cambio", "FS": "Split"}
HARP, MAR = f"Harp Helú · {CDMX_M:,} m", "Nivel del mar"
ARCHIVOS = {"pitcheos": "scores_pitch.parquet", "pitchers": "scores_pitcher.csv",
            "arsenales": "scores_arsenal.csv", "tipos": "altitude_by_pitch_type.csv"}
COLS_MAPA = ["pt", "ivb_mar", "hb_mar", "ivb_cdmx", "hb_cdmx", "stuff_mar_plus", "stuff_cdmx_plus"]
MAX_PUNTOS_MAPA = 80_000
# El numero "entre pitchers" del Prospecto solo se llama comparable con el Ranking si, para los pitchers de
# la liga, los dos caminos coinciden con al menos esta correlacion (simulator.contra_el_ranking).
R_COMPARABLE = 0.8
# Colores: tinta y grises. Azul-gris-naranja para "arriba / abajo del promedio" (el gris es el promedio);
# verde y rojo SOLO donde un intervalo permite decir que algo gana o pierde.
TINTA, TENUE, REJA, FONDO = "#111111", "#6b6a66", "#e1e0d9", "#fcfcfb"
GANA, PIERDE, SIN_EVIDENCIA = "#15803D", "#B91C1C", "#b9b8b2"
ARRIBA_ABAJO = LinearSegmentedColormap.from_list("abajo_arriba", ["#c2410c", "#f0efec", "#1c5cab"])
ESCALA_PITCHEO = ("**Puntos por pitcheo**: 100 = el pitcheo promedio de la liga lanzado en ese mismo aire; "
                  "10 puntos = una desviación estándar entre los pitcheos de la liga en las tablas.")
# variables del modelo que comparan el pitcheo con la recta primaria (para lectura_recta)
CONTRA_LA_RECTA = {"velo_diff_vs_fb": "la diferencia de velocidad contra la recta",
                   "ivb_diff_vs_fb": "la diferencia de quiebre vertical contra la recta",
                   "hb_diff_vs_fb": "la diferencia de quiebre horizontal contra la recta",
                   "tunnel_dist": "la distancia entre su punto de salida y el de la recta"}
# por que un tipo no usa su propia elasticidad (columna `motivo` de out/altitude_elasticities.csv)
MOTIVO = {"igual_a_familia": "la suya no se distingue de esa", "pocos_pitchers": "pocos pitchers para medir la suya",
          "no_estimable": "quiebre medio casi 0 o que cambia de signo", "sin_error_estandar": "sin error estándar",
          "recortada": "recortada al límite de cordura"}
PRESTADA_DE = {"familia": "la de su familia", "liga": "la de toda la liga", "mov_total": "la del movimiento total",
               "fisica": "1.0 (proporcional a la densidad)"}


# ── datos ───────────────────────────────────────────────────────────────────
def version():
    """Cambia cuando el pipeline reescribe out/: obliga a releer sin reiniciar el dashboard."""
    return tuple((str(config.OUT / f), (config.OUT / f).stat().st_mtime_ns) for f in ARCHIVOS.values())


@st.cache_data(show_spinner="Leyendo los pitcheos…")
def pitcheos(version):
    """Los pitcheos del mapa. Es la tabla grande: solo la piden funciones con cache, para que mover un
    control no la vuelva a leer ni a recorrer."""
    return pd.read_parquet(config.OUT / ARCHIVOS["pitcheos"], columns=COLS_MAPA)


@st.cache_data(show_spinner="Leyendo los resultados del pipeline…")
def datos(version):
    """(pitchers, arsenales, efecto por tipo, resumen del mapa: pitcheos y quiebre mediano de cada tipo)."""
    jug = pd.read_csv(config.OUT / ARCHIVOS["pitchers"], dtype={"pitcher_id": str})
    ars = pd.read_csv(config.OUT / ARCHIVOS["arsenales"], dtype={"pitcher_id": str})
    tipo = pd.read_csv(config.OUT / ARCHIVOS["tipos"])
    g = pitcheos(version).groupby("pt")
    return jug, ars, tipo, g[COLS_MAPA[1:5]].median().assign(pitcheos=g.size())


@st.cache_data(show_spinner=False, max_entries=32)
def tabla_mapa(version, pt, paso):
    """El mapa como tabla: las 15 formas (celdas de `paso` pulgadas del quiebre en el Harp Helu) con mas pitcheos."""
    d = pitcheos(version)
    d = d[d["pt"] == pt]
    celda = d.assign(ivb=(d["ivb_cdmx"] / paso).round() * paso, hb=(d["hb_cdmx"] / paso).round() * paso)
    return (celda.groupby(["ivb", "hb"]).agg(pitcheos=("pt", "size"), harp=("stuff_cdmx_plus", "mean"),
                                            mar=("stuff_mar_plus", "mean")).reset_index()
            .sort_values("pitcheos", ascending=False).head(15).sort_values("harp", ascending=False))


def leer_csv(nombre):
    """Tabla opcional de out/: sin filas si falta, esta vacia o no se puede leer."""
    try:
        return pd.read_csv(config.OUT / nombre)
    except (OSError, ValueError):
        return pd.DataFrame()


def leer_json(nombre):
    try:
        return json.loads((config.OUT / nombre).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def alto(mensaje):
    st.error(mensaje)
    st.stop()


def exige(tabla, cols, archivo):
    faltan = [c for c in cols if c not in tabla]
    if faltan:
        alto(f"`out/{archivo}` es de una versión anterior del código (faltan las columnas {faltan}). "
             "Corre `python run_all.py --desde score` y recarga la página.")


# ── frases: siempre a partir de un numero con su intervalo ──────────────────
def finito(*v):
    return all(isinstance(x, (int, float, np.floating, np.integer)) and np.isfinite(x) for x in v)


def intervalo_tipo(r):
    """El mas ancho de los dos intervalos del efecto de un tipo: el de muestreo de pitchers y el que
    hay entre los modelos de la validacion cruzada (score.py explica por que hacen falta los dos)."""
    return (pd.Series([r.get("delta_lo"), r.get("delta_lo_folds")], dtype=float).min(),
            pd.Series([r.get("delta_hi"), r.get("delta_hi_folds")], dtype=float).max())


def lectura_tipo(r):
    """(clase, frase) del efecto de la altitud en un tipo de pitcheo, segun out/altitude_by_pitch_type.csv:
      gana / pierde            solo si la tabla lo marca `significativo` (los dos intervalos excluyen 0
                               y hay pitchers suficientes)
      sin diferencia medible   si el intervalo contiene 0
      evidencia insuficiente   si excluye 0 pero no alcanza el minimo de pitchers o falta un intervalo"""
    nombre, d = NOMBRE.get(r["pt"], r["pt"]), r.get("delta")
    (lo, hi), n = intervalo_tipo(r), r.get("pitchers")
    if not finito(d):
        return "sin_datos", f"{nombre}: sin datos."
    cifra = f"{d:+.1f} puntos por pitcheo" + (f", IC 95 % de {lo:+.1f} a {hi:+.1f}" if finito(lo, hi)
                                              else ", sin intervalo")
    if finito(lo, hi) and str(r.get("significativo")).strip().lower() == "true":
        clase = "gana" if d > 0 else "pierde"
        return clase, (f"{nombre}: {clase} valor frente a los demás pitcheos al pasar del nivel del mar al "
                       f"Harp Helú ({cifra}).")
    if finito(lo, hi) and lo <= 0 <= hi:
        return "sin_dif", f"{nombre}: sin diferencia medible entre el Harp Helú y el nivel del mar ({cifra})."
    return "insuficiente", (f"{nombre}: evidencia insuficiente para afirmar una dirección ({cifra}"
                            + (f"; {int(n)} pitchers" if finito(n) else "") + ").")


def _par(v):
    """Un intervalo [lo, hi] de un JSON, o (None, None) si no viene completo."""
    return tuple(v) if isinstance(v, (list, tuple)) and len(v) == 2 else (None, None)


def _parte(valor, ic, que):
    """Una parte del costo con lo que dice SU intervalo (sin afirmar el signo por adelantado)."""
    lo, hi = _par(ic)
    if not finito(lo, hi):
        return f"{valor:+.2f} {que} (sin intervalo)"
    return f"{valor:+.2f} [{lo:+.2f}, {hi:+.2f}] {que}" + (", intervalo que incluye 0" if lo <= 0 <= hi else "")


def lectura_parque(p):
    """Frase del costo ABSOLUTO de lanzar en el Harp Helu, con el intervalo que incluye la incertidumbre
    de la recta del valor de los batazos (ic95_recta); None si el archivo no trae los numeros. En la
    referencia observada, "casa" es la categoria de altitud del Harp Helu (config.HOME_ALTITUDE)."""
    costo, ic = p.get("costo_cdmx_vs_mar_rv100"), _par(p.get("ic95_recta"))
    if not (finito(costo) and finito(*ic)):
        return None
    cifra = (f"{costo:+.2f} carreras por 100 pitcheos, IC 95 % de {ic[0]:+.2f} a {ic[1]:+.2f}"
             + ("" if p.get("recta_ic_fiable") else f", orientativo: sale de {p.get('recta_n_puntos', '?')} "
                                                     "puntos de densidad"))
    if ic[0] > 0:
        frase = f"Lanzar en el Harp Helú en vez de al nivel del mar le cuesta al pitcher promedio ({cifra})."
    elif ic[1] < 0:
        frase = f"Lanzar en el Harp Helú en vez de al nivel del mar favorece al pitcher promedio ({cifra})."
    else:
        frase = f"Sin diferencia medible entre lanzar en el Harp Helú y al nivel del mar ({cifra})."
    mov, val = p.get("por_movimiento_rv100"), p.get("por_valor_batazos_rv100")
    if finito(mov, val):
        frase += (" Descomposición: "
                  + _parte(mov, p.get("por_movimiento_ic95"), "por el cambio de movimiento, con el valor de los "
                                                              "batazos fijo") + "; "
                  + _parte(val, p.get("por_valor_batazos_ic95_recta"), "por el cambio de valor de los batazos con la "
                                                                       "densidad del aire") + ".")
    obs, (olo, ohi) = p.get("observado_casa_menos_mar_rv100"), _par(p.get("observado_ic95"))
    if finito(obs):
        casa = f"la categoría de altitud {config.HOME_ALTITUDE} (todos sus parques, no solo el Harp Helú)"
        frase += (f" Como referencia, los pitcheos lanzados en {casa} permitieron {obs:+.2f} carreras por 100"
                  + (f" [{olo:+.2f}, {ohi:+.2f}]" if finito(olo, ohi) else "")
                  + " frente a los lanzados al nivel del mar; esa diferencia mezcla parque, equipos y bateadores.")
    return frase


def lectura_recta(imp):
    """Por que se pide la recta primaria. Solo dice que variable pesa mas si la tabla de importancia por
    permutacion de la corrida (out/val_importancia_stuff.csv) pone primero una que compara contra la recta."""
    frase = "El modelo compara cada pitcheo contra esa recta: diferencia de velocidad y de quiebre."
    if {"variable", "delta_logloss"} <= set(imp.columns):
        orden = imp.assign(d=pd.to_numeric(imp["delta_logloss"], errors="coerce")).dropna(subset=["d"])
        orden = orden.sort_values("d", ascending=False)
        if len(orden) and orden["d"].iloc[0] > 0 and str(orden["variable"].iloc[0]) in CONTRA_LA_RECTA:
            frase += (f" En esta corrida, {CONTRA_LA_RECTA[str(orden['variable'].iloc[0])]} es la variable con mayor "
                      "importancia por permutación en el modelo Stuff.")
    return frase


def lectura_ranking(c):
    """(clase, frase): que tanto se puede leer el Stuff+ "entre pitchers" de un arsenal tecleado junto al
    Ranking, segun simulator.contra_el_ranking() (los pitchers de la liga pasados por los dos caminos)."""
    if not finito(c.get("r"), c.get("dif_abs")):
        return "sin_datos", ("No hay pitchers suficientes en la liga para medir cuánto coincide este número con el "
                             "del Ranking: no lo leas como un lugar en esa tabla.")
    medida = (f"para los {c['n']} pitcher-temporadas de la liga, este camino y el Stuff+ Harp Helú del Ranking "
              f"coinciden con r = {c['r']:.2f} y difieren en promedio {c['dif_abs']:.1f} puntos")
    if c["r"] >= R_COMPARABLE:
        return "comparable", f"Comparable con el Ranking: en esta corrida, {medida}."
    return "orientativo", (f"Orientativo, NO es un lugar en el Ranking: en esta corrida, {medida}. El simulador usa el "
                           "modelo ajustado con todos los datos y promedios tecleados; el Ranking, los modelos de la "
                           "validación cruzada sobre los pitcheos reales.")


# ── figuras (PNG en cache: una interaccion no las vuelve a dibujar) ─────────
def _ejes(ax):
    ax.set_facecolor(FONDO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJA)
    ax.tick_params(colors=TENUE, labelsize=9)


def _png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor=FONDO)
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=64)
def png_mapa(version, pt, marca):
    """Stuff+ medio por forma del pitcheo, cada panel con SUS coordenadas (el quiebre transportado a
    ese aire). marca = ((hb, ivb) al nivel del mar, (hb, ivb) en el Harp Helu) del pitcheo simulado,
    redondeada a media pulgada por quien llama: asi mover un control no dibuja un mapa nuevo cada vez."""
    d = pitcheos(version)
    d = d[d["pt"] == pt].dropna()
    d = d.sample(min(len(d), MAX_PUNTOS_MAPA), random_state=RANDOM_STATE)
    x0, x1 = np.percentile(d["hb_mar"], [0.5, 99.5])
    y0, y1 = np.percentile(d["ivb_mar"], [0.5, 99.5])
    if marca:                                         # que el pitcheo simulado siempre quepa
        x0, x1 = min(x0, *(m[0] for m in marca)), max(x1, *(m[0] for m in marca))
        y0, y1 = min(y0, *(m[1] for m in marca)), max(y1, *(m[1] for m in marca))
    marco = (x0 - 2, x1 + 2, y0 - 2, y1 + 2)
    fig = Figure(figsize=(11, 4.9), dpi=110, layout="constrained")
    axs = fig.subplots(1, 2, sharex=True, sharey=True)
    for k, (ax, suf, titulo) in enumerate(zip(axs, ("mar", "cdmx"), (MAR, HARP))):
        _ejes(ax)
        celdas = ax.hexbin(d[f"hb_{suf}"], d[f"ivb_{suf}"], C=d[f"stuff_{suf}_plus"], reduce_C_function=np.mean,
                           gridsize=24, extent=marco, mincnt=max(5, len(d) // 1500), cmap=ARRIBA_ABAJO,
                           norm=TwoSlopeNorm(100, 85, 115), linewidths=0.6, edgecolors=FONDO)
        ax.axhline(0, color=REJA, lw=1, zorder=0)
        ax.axvline(0, color=REJA, lw=1, zorder=0)
        ax.set_title(titulo, loc="left", fontsize=11, color=TINTA)
        ax.set_xlabel("HB (pulg) · positivo = lado del brazo", fontsize=9, color=TENUE)
        if marca:
            if k == 1:                                # de donde venia: el mismo pitcheo al nivel del mar
                ax.annotate("", xy=marca[1], xytext=marca[0],
                            arrowprops={"arrowstyle": "->", "color": TINTA, "lw": 1.2})
                ax.plot(*marca[0], "o", ms=7, mfc="none", mec=TINTA, mew=1.2)
            ax.plot(*marca[k], "o", ms=9, mfc=TINTA, mec=FONDO, mew=2)
    axs[0].set_ylabel("IVB (pulg)", fontsize=9, color=TENUE)
    if marca:
        punto = lambda relleno, texto: Line2D([], [], marker="o", ls="", ms=8, mfc=relleno, mec=TINTA, label=texto)
        axs[1].legend(handles=[punto(TINTA, "pitcheo simulado"), punto("none", "el mismo al nivel del mar")],
                      loc="best", fontsize=8, frameon=False)
    barra = fig.colorbar(celdas, ax=axs, shrink=0.85, pad=0.02)
    barra.set_label("Stuff+ medio (100 = promedio de ese aire)", fontsize=9, color=TENUE)
    barra.outline.set_visible(False)
    barra.ax.tick_params(colors=TENUE, labelsize=9)
    return _png(fig)


@st.cache_data(show_spinner=False)
def png_tipos(version):
    """Efecto relativo de la altitud por tipo, con su intervalo. Color solo donde la tabla permite
    afirmar una direccion; gris en lo demas."""
    t = datos(version)[2]
    t = t.assign(delta=pd.to_numeric(t["delta"], errors="coerce"))     # una tabla sin filas llega como texto
    t = t[np.isfinite(t["delta"])].sort_values("delta")
    fig = Figure(figsize=(8.2, 0.52 * len(t) + 1.3), dpi=110, layout="constrained")
    ax = fig.subplots()
    _ejes(ax)
    if t.empty:
        ax.text(0.5, 0.5, "Sin datos por tipo de pitcheo", ha="center", va="center", color=TENUE,
                transform=ax.transAxes)
        return _png(fig)
    clase = [lectura_tipo(r)[0] for r in t.to_dict("records")]
    color = [{"gana": GANA, "pierde": PIERDE}.get(c, SIN_EVIDENCIA) for c in clase]
    lo, hi = zip(*(intervalo_tipo(r) for r in t.to_dict("records")))
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    y, d = np.arange(len(t)), t["delta"].to_numpy(float)
    ax.barh(y, d, height=0.5, color=color, zorder=2)
    con_ic = np.isfinite(lo) & np.isfinite(hi)
    ax.hlines(y[con_ic], lo[con_ic], hi[con_ic], color=TINTA, lw=1.2, zorder=3)
    ax.axvline(0, color=TINTA, lw=0.8, zorder=1)
    ax.xaxis.grid(True, color=REJA, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y, [NOMBRE.get(p, p) for p in t["pt"]], fontsize=10, color=TINTA)
    borde = float(np.abs(np.concatenate([d, lo[con_ic], hi[con_ic]])).max()) or 1.0
    ax.set_xlim(-1.25 * borde, 1.25 * borde)
    for yi, di, l, h in zip(y, d, lo, hi):            # el valor, fuera del intervalo para que no se encime
        if np.isfinite(di):
            fin = (h if di >= 0 else l) if np.isfinite(l) and np.isfinite(h) else di
            ax.text(fin + 0.02 * borde * (1 if di >= 0 else -1), yi, f"{di:+.1f}", va="center",
                    ha="left" if di >= 0 else "right", fontsize=9, color=TINTA)
    ax.set_xlabel("Cambio relativo de Stuff+ del mismo pitcheo, Harp Helú menos nivel del mar (puntos por pitcheo)",
                  fontsize=9, color=TENUE)
    marcas = [Patch(color=c, label=f"{k} (los dos intervalos excluyen 0)")
              for c, k in ((GANA, "gana"), (PIERDE, "pierde")) if k in clase]
    if set(clase) - {"gana", "pierde"}:
        marcas.append(Patch(color=SIN_EVIDENCIA, label="sin diferencia medible o evidencia insuficiente"))
    marcas.append(Line2D([], [], color=TINTA, lw=1.2, label="intervalo de 95 % (el más ancho de los dos)"))
    ax.legend(handles=marcas, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False)
    return _png(fig)


# ── controles compartidos ───────────────────────────────────────────────────
LUGARES = {"Harp Helú (CDMX)": (CDMX_M, RHO_CDMX, CLIMA_TIPICO["cdmx"]),
           MAR: (SEA_LEVEL_M, RHO_MAR, CLIMA_TIPICO["mar"]),
           "Otra altitud": (None, None, (25.0, 55.0))}
OTROS_PARQUES = ", ".join(f"{STADIUMS[k][0].split(' · ')[-1]} {STADIUMS[k][1]:,}" for k in (
    "monterrey", "revolucion", "panamericano", "hermanos_serdan") if k in STADIUMS)


def aire_de_medicion(clave, clima_tipico):
    """Controles de DONDE se midio -> (densidad relativa del aire, lugar). Con 'clima tipico' las dos
    referencias usan exactamente las densidades de las tablas (config.RHO_CDMX / RHO_MAR)."""
    c1, c2, c3, c4 = st.columns(4)
    lugar = c1.selectbox("¿Dónde se midió?", list(LUGARES), key=f"lugar_{clave}",
                         help="El mismo quiebre vale distinto según el aire: hay que decir dónde se midió.")
    alt, rho, (t0, h0) = LUGARES[lugar]
    if alt is None:
        alt = c2.number_input("Altitud (m)", 0, 3000, 520, 10, key=f"alt_{clave}",
                              help=f"Por ejemplo: {OTROS_PARQUES}.")
        rho = sim.densidad(alt)
    rho_lugar = rho                                   # el aire de ese lugar con clima tipico: de ahi parten los controles
    tipico = c2.checkbox("Clima típico de los juegos LMB", clima_tipico, key=f"tipico_{clave}",
                         help="Desmárcalo para dar la temperatura y la humedad de la medición.")
    if not tipico:
        temp = c3.number_input("Temperatura (°C)", 0.0, 42.0, float(t0), 0.5, key=f"temp_{clave}_{lugar}")
        hum = c4.number_input("Humedad relativa (%)", 5.0, 100.0, float(h0), 1.0, key=f"hum_{clave}_{lugar}")
        rho = sim.densidad(alt, temp, hum)
    st.caption(f"Densidad del aire de la medición: **{rho:.3f}** (1 = aire estándar al nivel del mar; "
               f"Harp Helú {RHO_CDMX:.3f}, parques al nivel del mar {RHO_MAR:.3f}).")
    return rho, lugar, rho_lugar, alt


def deslizador(col, etiqueta, lo, hi, valor, paso, clave, formato="%.1f"):
    valor = float(np.clip(round(float(valor) / paso) * paso, lo, hi))
    return col.slider(etiqueta, float(lo), float(hi), valor, float(paso), format=formato, key=clave)


def numero(col, etiqueta, lo, hi, valor, paso, clave):
    return col.number_input(etiqueta, float(lo), float(hi), float(np.clip(round(float(valor), 1), lo, hi)),
                            float(paso), format="%.1f", key=clave)


def tarjeta(col, titulo, r, en_su_tipo=None):
    col.metric(f"Stuff+ · {titulo}", f"{r['stuff_plus']:.0f}")
    if en_su_tipo:
        col.caption(en_su_tipo)
    col.caption(f"Densidad del aire {r['densidad']:.3f} · IVB {r['ivb']:+.1f} pulg · HB {r['hb']:+.1f} pulg")
    col.caption(f"Swing fallado {100 * r['whiff_por_swing']:.0f} % de los swings · rodado "
                f"{100 * r['rodado_por_batazo']:.0f} % de los batazos · línea o elevado duro "
                f"{100 * r['duro_por_batazo']:.0f} % de los batazos")


# ── arranque ────────────────────────────────────────────────────────────────
st.title("Stuff+ LMB calibrado por altitud")
st.caption("Calidad esperada de cada pitcheo por sus características físicas, en el Harp Helú y al nivel del mar · "
           "modelo entrenado con el dataset oficial de ISAC (stuff_model_df, LMB 2024-2026)")

faltan = sim.faltantes() + [f"out/{f}" for f in ARCHIVOS.values() if not (config.OUT / f).exists()]
if faltan:
    alto("Todavía no están los resultados del pipeline. Faltan: " + ", ".join(f"`{f}`" for f in faltan) + ".\n\n"
         "Corre primero `python run_all.py` (o `python run_all.py --rapido` para una prueba) y recarga la página.")
try:
    jug, ars, tipo, mapa = datos(version())
    R = sim.recursos()
    with st.spinner("Calificando los arsenales de la liga (solo la primera vez)…"):
        sim.dispersion()
        sim.liga_tecleada()
        CONTRA_RANKING = sim.contra_el_ranking()
except sim.SimuladorNoListo as e:
    alto(str(e))
except (KeyError, ValueError) as e:
    alto(f"Los archivos de `out/` son de una versión anterior del código ({e}). Corre "
         "`python run_all.py --desde score` y recarga la página.")
RANKING = ["pitcher_name", "season", "pitcheos", "stuff_plus_enc", "stuff_cdmx_plus_enc", "swstr_pct"]
exige(tipo, ["pt", "delta", "delta_lo", "delta_hi", "significativo", "pitchers"], ARCHIVOS["tipos"])
exige(jug, RANKING, ARCHIVOS["pitchers"])
exige(ars, RANKING + ["pt"], ARCHIVOS["arsenales"])
TIPOS_SIM = [p for p in NOMBRE if p in R["defaults"]]              # tipos con pitcheos en los datos
RHO_LIGA = R["valores"]["densidad"]["rho_liga"]
POR_TIPO = {r["pt"]: lectura_tipo(r) for r in tipo.to_dict("records")}
# De que corrida es lo que hay en out/: lo dice cada paso de run_info.json (la bandera de arriba del archivo
# solo describe la ultima vez que se llamo a run_all.py). Es la misma regla que usa el documento.
INFO = leer_json("run_info.json")
PASOS = {k: v for k, v in (INFO.get("pasos") or {}).items() if isinstance(v, dict)}
if INFO.get("rapido") or any(p.get("rapido") for p in PASOS.values()):
    st.warning("CORRIDA RÁPIDA (30 % de los pitchers): estos números solo sirven para comprobar que todo corre.")
elif "score" not in PASOS:
    st.warning("No se sabe con qué corrida se generaron estos resultados (`out/run_info.json` no registra el paso "
               "`score`). Antes de usarlos, corre `python run_all.py`.")

tabs = st.tabs(["Simulador", "Mapa de movimiento", "Ranking", "Altitud por pitcheo", "Prospecto"])

# ── 1. SIMULADOR ────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("Captura el **promedio** de un pitcheo tal como se midió. El simulador lo lleva al aire del Harp Helú "
                "y al del nivel del mar con la física medida en la LMB y lo califica en cada uno.")
    rho_med, lugar, rho_lugar, alt_med = aire_de_medicion("sim", clima_tipico=True)
    c1, c2, c3 = st.columns(3)
    pt = c1.selectbox("Tipo de pitcheo", TIPOS_SIM, format_func=NOMBRE.get, key="pt_sim")
    zurdo = c1.radio("Mano", ["Derecho", "Zurdo"], horizontal=True, key="mano_sim") == "Zurdo"
    # punto de partida: el pitcheo tipico de la liga con el quiebre que tendria en ese lugar
    tp = sim.tipico(pt, rho_lugar)
    k = f"{pt}_{lugar}" + ("" if LUGARES[lugar][0] is not None else f"_{alt_med}")   # otra altitud: otros controles
    velo = deslizador(c1, "Velocidad (mph)", 65, 103, tp["velo"], 0.5, f"velo_{k}")
    ivb = deslizador(c2, "IVB · quiebre vertical inducido (pulg)", -25, 25, tp["ivb"], 0.5, f"ivb_{k}")
    hb = deslizador(c2, "HB · quiebre horizontal (pulg), lado del brazo (+)", -25, 25, tp["hb"], 0.5, f"hb_{k}")
    spin = deslizador(c2, "Spin (rpm)", 500, 3500, tp["spin"], 25, f"spin_{k}", "%.0f")
    rel_h = deslizador(c3, "Altura de salida (pies)", 3.5, 7.5, tp["rel_h"], 0.1, f"relh_{k}")
    rel_s = deslizador(c3, "Lado de salida (pies), lado del brazo (+)", -1.5, 4.0, tp["rel_s"], 0.1, f"rels_{k}")
    ext = deslizador(c3, "Extensión (pies)", 4.5, 7.8, tp["ext"], 0.1, f"ext_{k}")
    c3.caption(f"Los controles empiezan en la mediana de la liga para {NOMBRE[pt].lower()} en ese lugar.")

    es_primaria = pt in FASTBALLS and c1.checkbox(
        "Es la recta primaria del pitcher", tp["fb_tipo"] == pt, key=f"primaria_{pt}",
        help="La recta que más usa. Los demás pitcheos se comparan contra ella (diferencia de velocidad y de "
             "quiebre).")
    recta = {"fb_tipo": pt}
    if not es_primaria:
        st.markdown("**Recta primaria del pitcher**, medida en el mismo lugar. "
                    + lectura_recta(leer_csv("val_importancia_stuff.csv")))
        r1, r2, r3, r4 = st.columns(4)
        opciones = [f for f in FASTBALLS if f != pt and f in R["defaults"]] or list(FASTBALLS)
        fb_tipo = r1.selectbox("Tipo de recta", opciones, format_func=NOMBRE.get, key=f"fbtipo_{k}",
                               index=opciones.index(tp["fb_tipo"]) if tp["fb_tipo"] in opciones else 0)
        recta = {"fb_tipo": fb_tipo,
                 "fb_velo": numero(r2, "Velocidad de la recta (mph)", 65, 103, tp["fb_velo"], 0.5, f"fbv_{k}"),
                 "fb_ivb": numero(r3, "IVB de la recta (pulg)", -10, 28, tp["fb_ivb"], 0.5, f"fbi_{k}"),
                 "fb_hb_brazo": numero(r4, "HB de la recta (pulg), lado del brazo (+)", -15, 28, tp["fb_hb"], 0.5,
                                       f"fbh_{k}")}

    rho_cdmx, rho_mar = RHO_CDMX, RHO_MAR
    with st.expander("Clima del juego en cada destino (la temperatura y la humedad cambian la densidad del aire)"):
        if not st.checkbox("Usar el clima promedio de los juegos LMB en cada lugar", True, key="clima_destino"):
            k1, k2 = st.columns(2)
            (tc0, hc0), (tm0, hm0) = CLIMA_TIPICO["cdmx"], CLIMA_TIPICO["mar"]
            t_c = k1.slider("Harp Helú · temperatura (°C)", 5.0, 32.0, float(tc0), 0.5, key="t_cdmx")
            h_c = k1.slider("Harp Helú · humedad (%)", 10.0, 100.0, float(hc0), 1.0, key="h_cdmx")
            # desde 8 C: con menos, el aire del nivel del mar ya es mas denso que el que cubre la escala del simulador
            t_m = k2.slider("Nivel del mar · temperatura (°C)", 8.0, 40.0, float(tm0), 0.5, key="t_mar")
            h_m = k2.slider("Nivel del mar · humedad (%)", 10.0, 100.0, float(hm0), 1.0, key="h_mar")
            rho_cdmx, rho_mar = sim.densidad(CDMX_M, t_c, h_c), sim.densidad(SEA_LEVEL_M, t_m, h_m)

    simulado = None
    try:
        captura = pd.DataFrame([{"pt": pt, "velo": velo, "ivb": ivb, "hb_brazo": hb, "spin": spin, "rel_h": rel_h,
                                 "rel_s_brazo": rel_s, "ext": ext, "zurdo": zurdo, **recta}])
        en_cdmx, en_mar = (t.iloc[0] for t in sim.calificar_promedio(captura, rho_med, rho_cdmx, rho_mar))
        media = lambda v: round(2 * float(v)) / 2          # a media pulgada: es la llave del mapa en cache
        simulado = {"pt": pt, "marca": ((media(en_mar["hb"]), media(en_mar["ivb"])),
                                        (media(en_cdmx["hb"]), media(en_cdmx["ivb"])))}
    except (ValueError, sim.SimuladorNoListo) as e:
        st.error(f"No se pudo calificar el pitcheo: {e}")
    if simulado:
        st.divider()
        a, b, c = st.columns(3)
        en_su_tipo = None
        if rho_cdmx == RHO_CDMX:                      # la liga de referencia esta calificada en ese mismo aire
            pct, n = sim.entre_los_de_su_tipo(pt, en_cdmx["score"])
            if finito(pct):
                en_su_tipo = (f"Percentil {pct:.0f} entre los {n} arsenales de {NOMBRE[pt].lower()} de la liga, "
                              "calificados de la misma forma")
        tarjeta(a, HARP, en_cdmx, en_su_tipo)
        tarjeta(b, MAR, en_mar)
        c.metric("Efecto de la altitud (relativo)", f"{en_cdmx['stuff_plus'] - en_mar['stuff_plus']:+.1f}",
                 help="Stuff+ en el Harp Helú menos Stuff+ al nivel del mar, en puntos por pitcheo. Cada aire tiene "
                      "su propio 100: mide si el pitcheo sube o baja FRENTE A LOS DEMÁS, no si rinde menos.")
        c.metric("Costo del Harp Helú (absoluto)", f"{100 * (en_mar['score'] - en_cdmx['score']):+.2f}",
                 help="Carreras esperadas por cada 100 pitcheos que cuesta lanzarlo en el Harp Helú en vez de al "
                      "nivel del mar. Incluye el efecto de parque, que es común a todos los pitcheos.")
        c.caption("Los dos números de este pitcheo son la estimación puntual del modelo, sin intervalo. Lo que sí "
                  "está medido con intervalo es el efecto promedio de su tipo en la liga. "
                  + POR_TIPO.get(pt, ("", "Sin datos de la liga para este tipo."))[1])
        ee = [en_cdmx["stuff_plus_ee"], en_mar["stuff_plus_ee"], en_mar["dif_plus_ee"], 100 * en_mar["dif_score_ee"]]
        if finito(*ee):
            c.caption(f"Error de simulación (error estándar por representar el promedio con {sim.N_NUBE} "
                      f"repeticiones; no es el error del modelo): ±{max(ee[0], ee[1]):.1f} en cada Stuff+, "
                      f"±{ee[2]:.1f} en el efecto y ±{ee[3]:.2f} en el costo.")
        sd_sim = R["escala"].get("simulador", {}).get("sd")
        with st.expander("Cómo leer estos números"):
            st.markdown(
                f"- {ESCALA_PITCHEO} No se compara con el Stuff+ entre pitchers de la pestaña Ranking."
                + (f" Con el modelo del simulador, una desviación estándar entre esos mismos pitcheos son "
                   f"{10 * sd_sim / R['escala']['stuff']['sd']:.1f} puntos." if finito(sd_sim) else "") + "\n"
                "- Se califica **como lo lanzaría un pitcher real**: el promedio que capturas más la dispersión con "
                f"que los pitchers de la liga repiten ese tipo de pitcheo ({sim.N_NUBE} repeticiones). Qué tan bien "
                "reproduce eso el Stuff+ medio real de un arsenal, contra calificar el promedio como si fuera un "
                "solo pitcheo, lo mide `python src/simulator.py` (sección 3).\n"
                "- El quiebre se lleva de un aire a otro con la elasticidad medida en la LMB; velocidad, spin y punto "
                "de salida no cambian. **No está modelado** que en aire delgado la bola pierde menos velocidad en el "
                "camino, ni la ubicación del pitcheo (eso es Location+).\n"
                "- La distancia entre el punto de salida del pitcheo y el de la recta (túnel) es un rasgo del "
                "repertorio: vale 0 para la recta primaria y, como aquí no se captura el punto de salida de la recta, "
                "para los demás pitcheos se usa la mediana de la liga para su tipo.\n"
                "- El simulador usa el modelo ajustado con todos los datos; las tablas usan los modelos de la "
                "validación cruzada (a cada pitcher lo califica uno que no lo vio). No dan el mismo número"
                + (f": en esta corrida, para los {CONTRA_RANKING['n']} pitcher-temporadas de la liga, el Stuff+ por "
                   f"este camino y el del Ranking coinciden con r = {CONTRA_RANKING['r']:.2f}"
                   if finito(CONTRA_RANKING.get("r")) else "")
                + ". La comparación pitcheo por pitcheo la imprime `python src/simulator.py` (sección 1).\n"
                "- Un pitcheo con todas sus características en la mediana no es «el pitcheo promedio»: su Stuff+ no "
                "tiene por qué ser 100.")

# ── 2. MAPA DE MOVIMIENTO ───────────────────────────────────────────────────
with tabs[1]:
    con_datos = [p for p in NOMBRE if p in mapa.index]
    if not con_datos:
        st.info("No hay pitcheos calificados que dibujar (`out/scores_pitch.parquet` no trae ninguno de los tipos).")
    else:
        partida = simulado["pt"] if simulado and simulado["pt"] in con_datos else con_datos[0]
        pt2 = st.selectbox("Tipo de pitcheo", con_datos, index=con_datos.index(partida), format_func=NOMBRE.get,
                           help="Empieza en el tipo del Simulador; si es el mismo, el pitcheo simulado aparece en el "
                                "mapa.")
        marca = simulado["marca"] if simulado and simulado["pt"] == pt2 else None
        st.image(png_mapa(version(), pt2, marca))
        med = mapa.loc[pt2]
        st.caption(
            f"{NOMBRE[pt2]}, {int(med['pitcheos']):,} pitcheos. Cada panel dibuja el quiebre que esos pitcheos "
            f"tendrían en ese aire (IVB mediano {med['ivb_mar']:+.1f} pulg al nivel del mar y {med['ivb_cdmx']:+.1f} "
            f"en el Harp Helú; HB {med['hb_mar']:+.1f} y {med['hb_cdmx']:+.1f}). El color es el Stuff+ medio de los "
            "pitcheos de cada celda, en puntos por pitcheo, y el 100 de cada panel es el promedio de ESE aire. "
            + POR_TIPO.get(pt2, ("", ""))[1])
        with st.expander("Ver el mapa como tabla (formas con más pitcheos)"):
            paso = 4
            st.caption(f"Quiebre en el Harp Helú, en celdas de {paso} pulgadas; Stuff+ medio en puntos por pitcheo.")
            st.dataframe(tabla_mapa(version(), pt2, paso), hide_index=True, column_config={
                "ivb": st.column_config.NumberColumn("IVB en el Harp Helú (pulg)", format="%.0f"),
                "hb": st.column_config.NumberColumn("HB en el Harp Helú (pulg)", format="%.0f"),
                "pitcheos": st.column_config.NumberColumn("Pitcheos", format="%d"),
                "harp": st.column_config.NumberColumn("Stuff+ Harp Helú", format="%.1f"),
                "mar": st.column_config.NumberColumn("Stuff+ nivel del mar", format="%.1f")})

# ── 3. RANKING ──────────────────────────────────────────────────────────────
with tabs[2]:
    f1, f2, f3 = st.columns(3)
    vista = f1.radio("Ver", ["Pitchers", "Arsenales (pitcher y tipo)"], horizontal=True, key="vista_rank")
    por_pitcher = vista == "Pitchers"
    base = jug if por_pitcher else ars
    grupo = "pitchers" if por_pitcher else "arsenales"
    minimo = config.MIN_PITCHES_PITCHER if por_pitcher else config.MIN_PITCHES_ARSENAL
    if base.empty:
        st.info(f"Ningún grupo llega a {minimo} pitcheos: no hay {grupo} que ordenar.")
    else:
        temporada = f2.selectbox("Temporada", sorted(base["season"].unique(), reverse=True), key="temp_rank")
        t = base[base["season"] == temporada]
        if not por_pitcher:
            tipos_rank = ["Todos"] + [p for p in NOMBRE if (t["pt"] == p).any()]
            cual = f3.selectbox("Tipo de pitcheo", tipos_rank, format_func=lambda p: NOMBRE.get(p, p),
                                key="tipo_rank")
            t = t if cual == "Todos" else t[t["pt"] == cual]
        # el error del modelo no esta en ningun intervalo por pitcher: score.py lo mide aparte (val_stability.json)
        estab = leer_json("val_stability.json")
        error = estab.get("error_modelo_stuff_pts")
        semi = ((jug["stuff_hi"] - jug["stuff_lo"]).mean() / 2 if {"stuff_lo", "stuff_hi"} <= set(jug.columns)
                else np.nan)
        # score.py toma el juego como unidad de muestreo solo si el dataset trae game_id (split_half_por = "juego");
        # sin val_stability.json (o sin esa clave) no se sabe cual fue la unidad: no se afirma ninguna
        ruido = {"juego": " (juegos y pitcheos)",
                 "pitcheo": " (pitcheos tomados como independientes, porque el dataset no trae el juego o le falta a "
                            "demasiados pitcheos: sale angosto)"}.get(str(estab.get("split_half_por")), "")
        st.caption(
            f"**Puntos entre {grupo}**: 100 = promedio de los {grupo} calificados ({minimo}+ pitcheos), 10 puntos = "
            f"una desviación estándar entre {grupo}. No se comparan con los puntos por pitcheo del Simulador. Cada "
            "valor está encogido hacia 100 según su confiabilidad; el intervalo de 95 % cubre el ruido de muestreo"
            f"{ruido}, no el error del modelo"
            # el error es UNA desviacion estandar y el intervalo es de 95 %: se comparan en la misma unidad
            + (f": en esta corrida el error típico de un modelo de la validación (una desviación estándar, medida "
               f"con los modelos que califican a los mismos pitchers) es de {error:.1f} puntos de Stuff+ por pitcher, "
               f"contra {semi / 1.96:.1f} del intervalo por pitcher (su semiancho medio, {semi:.1f}, entre 1.96)"
               if finito(error, semi) else "")
            + ". A cada pitcher lo califica un modelo que nunca lo vio.")
        entero = lambda col: (t[col].round() + 0.0).map("{:.0f}".format).astype(str)      # + 0.0: que no salga "-0"
        ic = lambda s: ("[" + entero(f"{s}_lo") + ", " + entero(f"{s}_hi") + "]"
                        if {f"{s}_lo", f"{s}_hi"} <= set(t.columns) else "")
        en_pct = lambda col: 100 * t[col] if col in t else np.nan
        ver = pd.DataFrame({
            "Pitcher": t["pitcher_name"],
            **({} if por_pitcher else {"Pitcheo": t["pt"].map(NOMBRE).fillna(t["pt"])}),
            "Pitcheos": t["pitcheos"],
            "Stuff+ Harp Helú": t["stuff_cdmx_plus_enc"], "IC 95 % Harp Helú": ic("stuff_cdmx"),
            "Stuff+ neutral": t["stuff_plus_enc"], "IC 95 % neutral": ic("stuff"),
            "Efecto altitud": t.get("delta_altitud"), "IC 95 % efecto": ic("delta"),
            "Location+": t.get("location_plus_enc"),
            "Pitching+ neutral": t.get("pitching_neutral_plus_enc"),
            "IC 95 % Pitching+ neutral": ic("pitching_neutral"),
            "Pitching+ en su parque": t.get("pitching_plus_enc"),
            "Swing fallado / pitcheo (%)": en_pct("swstr_pct"), "Swing fallado / swing (%)": en_pct("whiff_swing_pct"),
            "Chase (%)": en_pct("chase_pct"), "Rodados / batazo (%)": en_pct("gb_pct"),
            "Contacto duro (%)": en_pct("hard_pct"),
            "Carreras / 100": t.get("rv100")}).sort_values("Stuff+ Harp Helú", ascending=False)
        formato = {"Pitcheos": "%d", "Carreras / 100": "%.2f"}
        st.dataframe(ver, hide_index=True, column_config={
            c: st.column_config.NumberColumn(c, format=formato.get(c, "%.1f"))
            for c in ver.columns if ver[c].dtype.kind in "fiu"})
        st.caption("Stuff+ neutral: el pitcheo llevado a la densidad media de la liga (compara pitchers sin importar "
                   "dónde lanzaron). Efecto altitud: Stuff+ Harp Helú menos Stuff+ nivel del mar del mismo pitcher, "
                   "relativo a la liga. Pitching+ neutral: con los batazos valuados a la densidad media de la liga, "
                   "para comparar pitchers de parques distintos; Pitching+ en su parque: valuados en el aire donde "
                   "lanzó, así que incluye el efecto del parque. Chase: swings a pitcheos fuera de la zona. Contacto "
                   "duro: batazos de 95+ mph entre los batazos con velocidad de salida medida. Carreras / 100: valor "
                   "observado; negativo favorece al pitcher.")

# ── 4. ALTITUD POR PITCHEO ──────────────────────────────────────────────────
with tabs[3]:
    st.image(png_tipos(version()))
    st.caption("Efecto RELATIVO, en puntos por pitcheo: cuánto sube o baja cada tipo frente a los demás pitcheos al "
               "pasar del nivel del mar al Harp Helú (en la liga promedia 0). La barra tiene color solo cuando los "
               "dos intervalos, el de muestreo de pitchers y el que hay entre los modelos de la validación, "
               "excluyen 0.")
    st.markdown("\n".join(f"- {frase}" for _, frase in POR_TIPO.values()))
    frase = lectura_parque(leer_json("altitude_park_effect.json"))
    st.markdown(f"**Efecto absoluto.** {frase}" if frase else
                "**Efecto absoluto.** No estimable con estos datos (`out/altitude_park_effect.json`).")
    with st.expander("Tabla completa por tipo"):
        st.caption("delta = efecto relativo en puntos por pitcheo (mov = por el cambio de quiebre, valor = por lo que "
                   "valen sus batazos en aire delgado); costo = carreras por 100 pitcheos, absoluto.")
        st.dataframe(tipo.assign(pt=tipo["pt"].map(NOMBRE).fillna(tipo["pt"])), hide_index=True, column_config={
            c: st.column_config.NumberColumn(c, format="%d" if c in ("pitcheos", "pitchers") else "%.2f")
            for c in tipo.columns if tipo[c].dtype.kind in "fiu"})
    fisica = leer_csv("altitude_elasticities.csv")
    if {"var", "pt", "media_mar", "media_cdmx", "beta_usada", "fuente"} <= set(fisica.columns):
        with st.expander("Cuánto quiebre se pierde: la física que usa el simulador"):
            f = fisica[fisica["var"].isin(["ivb", "hb_n"]) & fisica["pt"].isin(list(NOMBRE))]
            fuente = f["fuente"].astype(str)
            propia = fuente.str.startswith("propia")
            origen = ("prestada: " + fuente.map(PRESTADA_DE).fillna(fuente)).mask(propia, "propia del tipo")
            nota = "«Prestada» = el tipo no usa su propia elasticidad sino la de su familia o la de la liga."
            if "motivo" in f:                         # por que se presto: no siempre es por falta de datos
                motivo = f["motivo"].map(MOTIVO)
                origen = origen.mask(motivo.notna() & ~propia, origen + " (" + motivo.fillna("") + ")")
                nota = ("«Prestada» = el tipo no usa su propia elasticidad sino la de su familia o la de la liga; "
                        "entre paréntesis, por qué.")
            ver = pd.DataFrame({
                "Pitcheo": f["pt"].map(NOMBRE),
                "Quiebre": f["var"].map({"ivb": "IVB", "hb_n": "HB (lado del brazo +)"}),
                "Nivel del mar (pulg)": f["media_mar"], "Harp Helú (pulg)": f["media_cdmx"],
                "Cambio (pulg)": f["media_cdmx"] - f["media_mar"]})
            if {"delta_lo", "delta_hi"} <= set(f.columns):
                par = lambda col: f[col].map(lambda v: f"{v:+.1f}" if finito(v) else "s/d")
                ver["IC 95 % del cambio"] = "[" + par("delta_lo") + ", " + par("delta_hi") + "]"
                if "ic_fiable_usada" in f:
                    fiable = f["ic_fiable_usada"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
                    ver["IC 95 % del cambio"] = (ver["IC 95 % del cambio"]
                                                 + fiable.map({True: "", False: " (orientativo)"}))
                    if not fiable.all():
                        nota += (" Un intervalo «orientativo» sale de muy pocos niveles de densidad: indica el orden "
                                 "de magnitud, no una precisión.")
            ver["Elasticidad usada"], ver["Origen"] = f["beta_usada"], origen
            st.caption("quiebre en el destino = quiebre medido × (densidad destino / densidad de la medición) ^ "
                       "elasticidad. El intervalo del cambio solo lleva la incertidumbre de la elasticidad. " + nota)
            st.dataframe(ver, hide_index=True, column_config={
                c: st.column_config.NumberColumn(c, format="%.2f") for c in ver.columns if ver[c].dtype.kind == "f"})

# ── 5. PROSPECTO ────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("Captura el arsenal completo de un prospecto o agente libre: el **promedio** de cada pitcheo y cuánto "
                "lo usa. Primero, dónde y con qué clima se midió el bullpen.")
    rho_p = aire_de_medicion("pros", clima_tipico=False)[0]
    p1, p2, p3, p4 = st.columns(4)
    zurdo_p = p1.radio("Mano", ["Derecho", "Zurdo"], horizontal=True, key="mano_pros") == "Zurdo"
    tp = sim.tipico(TIPOS_SIM[0], RHO_CDMX)                         # punto de partida: la mediana de la liga
    rel_h_p = numero(p2, "Altura de salida (pies)", 3.5, 7.5, tp["rel_h"], 0.1, "relh_pros")
    rel_s_p = numero(p3, "Lado de salida (pies), lado del brazo (+)", -1.5, 4.0, tp["rel_s"], 0.1, "rels_pros")
    ext_p = numero(p4, "Extensión (pies)", 4.5, 7.8, tp["ext"], 0.1, "ext_pros")
    inicial = []
    for p, uso in (("FF", 50), ("SL", 30), ("CH", 20)):           # punto de partida: pitcheos tipicos en el Harp Helu
        if p in R["defaults"]:
            tp = sim.tipico(p, RHO_CDMX)
            inicial.append({"pt": p, "uso": uso, "velo": round(tp["velo"], 1), "ivb": round(tp["ivb"], 1),
                            "hb": round(tp["hb"], 1), "spin": int(round(tp["spin"], -1))})
    inicial = pd.DataFrame(inicial, columns=["pt", "uso", "velo", "ivb", "hb", "spin"])
    numerica = st.column_config.NumberColumn
    filas = st.data_editor(inicial, num_rows="dynamic", hide_index=True, key="arsenal_pros", column_config={
        "pt": st.column_config.SelectboxColumn("Pitcheo", options=TIPOS_SIM, required=True,
                                               help=" · ".join(f"{p} = {NOMBRE[p]}" for p in TIPOS_SIM)),
        "uso": numerica("Uso (%)", min_value=1, max_value=100, step=1, required=True),
        "velo": numerica("Velocidad (mph)", min_value=60.0, max_value=105.0, step=0.1, format="%.1f", required=True),
        "ivb": numerica("IVB (pulg)", min_value=-30.0, max_value=30.0, step=0.1, format="%.1f", required=True),
        "hb": numerica("HB (pulg), lado del brazo (+)", min_value=-30.0, max_value=30.0, step=0.1, format="%.1f",
                       required=True),
        "spin": numerica("Spin (rpm)", min_value=500, max_value=3600, step=10, required=True)})
    st.caption("Pitcheo: " + " · ".join(f"{p} = {NOMBRE[p]}" for p in TIPOS_SIM) + ". La tabla viene con valores "
               "típicos de la liga en el Harp Helú: cámbialos por los del prospecto. El uso se normaliza si no suma 100.")
    filas = pd.DataFrame(filas).dropna(subset=["pt", "uso", "velo", "ivb", "hb", "spin"])
    filas = filas[filas["pt"].isin(TIPOS_SIM) & (pd.to_numeric(filas["uso"], errors="coerce") > 0)]
    if filas.empty:
        st.info("Captura al menos un pitcheo completo (tipo, uso, velocidad, IVB, HB y spin) para calificar el arsenal.")
    else:
        try:
            pp, pa = sim.arsenales(filas.rename(columns={"hb": "hb_brazo"}).assign(
                rel_h=rel_h_p, rel_s_brazo=rel_s_p, ext=ext_p, zurdo=zurdo_p), rho_p)
        except (ValueError, sim.SimuladorNoListo) as e:
            pp = None
            st.error(f"No se pudo calificar el arsenal: {e}")
        if pp is not None:
            total = pa.iloc[0]
            plus, pct, n = sim.entre_pitchers(total["score_cdmx"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Stuff+ del arsenal en el Harp Helú", f"{plus:.0f}" if finito(plus) else "sin datos",
                      help="Puntos ENTRE PITCHERS (la escala de la pestaña Ranking), no los puntos por pitcheo de la "
                           "tabla de abajo. Qué tanto coincide con el Ranking en esta corrida se dice debajo.")
            if finito(plus):
                cubre = sim.liga_tecleada()[1]["cobertura"].mean()
                m1.caption(f"Puntos entre pitchers · percentil {pct:.0f} entre {n} pitcher-temporadas de la liga con "
                           f"{config.MIN_PITCHES_PITCHER}+ pitcheos, calificados por este mismo camino: el promedio "
                           f"de cada uno de sus pitcheos con {config.MIN_PITCHES_ARSENAL}+ lanzamientos (en promedio, "
                           f"el {100 * cubre:.0f} % de lo que lanzan).")
                m1.caption(lectura_ranking(CONTRA_RANKING)[1])
            else:
                m1.caption(f"Solo hay {n} arsenales de la liga para comparar: no alcanza para una escala.")
            m2.metric("Efecto de la altitud (relativo)", f"{total['delta']:+.1f}",
                      help="Puntos por pitcheo, pesando cada pitcheo por su uso: Harp Helú menos nivel del mar.")
            m3.metric("Costo del Harp Helú (absoluto)", f"{total['costo_cdmx_rv100']:+.2f}",
                      help="Carreras por 100 pitcheos frente a lanzar al nivel del mar; incluye el efecto de parque.")
            if finito(total["delta_ee"], total["costo_ee"]):
                m2.caption(f"Error de simulación (error estándar por representar cada promedio con {sim.N_NUBE} "
                           f"repeticiones; no es el error del modelo): ±{total['delta_ee']:.1f} en el efecto y "
                           f"±{total['costo_ee']:.2f} en el costo.")
            st.caption("Se captura un solo punto de salida para todo el arsenal, así que la distancia entre el punto "
                       "de salida de cada pitcheo y el de la recta (túnel) no se conoce: vale 0 para la recta "
                       "primaria y la mediana de la liga para cada uno de los demás tipos.")
            if (pp["recta_ref"] == "tipica").any():
                st.caption("El arsenal no trae recta (FF, SI o FC): cada pitcheo se compara contra la recta típica de "
                           "quienes lo lanzan en la liga.")
            en_su_tipo = [sim.entre_los_de_su_tipo(p, v)[0] for p, v in zip(pp["pt"], pp["score_cdmx"])]
            ver = pd.DataFrame({
                "Pitcheo": pp["pt"].map(NOMBRE), "Uso (%)": 100 * pp["peso"],
                "Stuff+ Harp Helú": pp["stuff_cdmx"], "Percentil en su tipo": en_su_tipo,
                "Stuff+ nivel del mar": pp["stuff_mar"], "Efecto altitud": pp["delta"],
                "Costo (carreras / 100)": pp["costo_cdmx_rv100"],
                "IVB en el Harp Helú": pp["ivb_cdmx"], "HB en el Harp Helú": pp["hb_cdmx"],
                "Swing fallado / swing (%)": 100 * pp["whiff_por_swing"],
                "Rodados / batazo (%)": 100 * pp["rodado_por_batazo"]})
            st.caption(f"Por pitcheo. {ESCALA_PITCHEO} Percentil en su tipo: lugar entre los arsenales de ese tipo en "
                       "la liga. Las tasas son las que espera el modelo en el Harp Helú.")
            formato = {"Costo (carreras / 100)": "%.2f", "Percentil en su tipo": "%.0f", "Uso (%)": "%.0f"}
            st.dataframe(ver, hide_index=True, column_config={
                c: st.column_config.NumberColumn(c, format=formato.get(c, "%.1f")) for c in ver.columns if c != "Pitcheo"})
            st.markdown("\n".join("- " + POR_TIPO.get(p, ("", NOMBRE[p] + ": sin datos de la liga."))[1]
                                  for p in dict.fromkeys(pp["pt"])))
            st.caption("Las frases de arriba son el efecto promedio de cada TIPO en la liga, con su intervalo; los "
                       "números del prospecto son la estimación puntual del modelo. Es una aproximación: se califica "
                       "el promedio de cada pitcheo con la dispersión típica de la liga, no los pitcheos reales del "
                       "prospecto.")
