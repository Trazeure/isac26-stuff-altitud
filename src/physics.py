"""
EFECTO EMPIRICO DE LA ALTITUD EN LA LMB: cuanto cambia cada pitcheo con el aire.

DISENO: cada pitcher contra si mismo. Un grupo g es un pitcher x temporada x
tipo de pitcheo; en cada estadio donde lanzo 15+ pitcheos se promedia la
variable (una CELDA) y se ajusta

      y_gs = M_g * exp(beta * (x_gs - xbar)) + error       x = log(densidad del aire)

M_g es el nivel propio del grupo, CON SU SIGNO, y beta la ELASTICIDAD: cuanto
cambia la variable, en %, por cada 1% de densidad. Si la fuerza del aire es
proporcional a la densidad, beta ~ 1. No se usa log|x|: en los pitcheos cuyo
quiebre cruza el cero (slider, cutter) el valor absoluto es ruido y diluia beta.

INCERTIDUMBRE: la densidad cambia entre ESTADIOS (en el dataset oficial, entre
sus 3 categorias de altitud, que hacen de estadio), no entre pitchers, y cada
parque trae su propio sesgo de medicion. El intervalo principal es un jackknife
que quita un estadio (una categoria) a la vez; con menos de 6 se escribe pero
se marca ic_fiable = False. El bootstrap de pitchers se guarda aparte: es mas
angosto porque ignora ese sesgo de parque.

FILAS: 7 tipos x (ivb, hb_n, mov_total); TODOS (liga); y FAMILIAS con efecto
fijo pitcher x temporada, sin usar la etiqueta: AutoPitchType clasifica por
movimiento y en altitud las etiquetas migran (un sinker que corre menos pasa a
"recta"), lo que contamina la beta por tipo. Una fila de familia aguanta el
cambio de etiqueta DENTRO de la familia; no aguanta un cambio real de mezcla ni
una etiqueta que cruza de una familia a otra. Por eso el cutter (FC) no entra
en la media de ninguna familia: queda entre la recta y el slider y su parte
dentro de SL+CU+FC cambia con la altitud, asi que con el adentro la beta del HB
de los quiebres puede salir del rango de las de sus propios tipos. Cuando le
faltan pitchers, FC toma prestada la beta de las rectas (es una recta en
features.FASTBALLS).

beta_usada / fuente: UNICO lugar donde se decide que elasticidad mueve los
pitcheos en el contrafactual (score.py, el simulador y la web leen esa columna).
beta_usada_lo / beta_usada_hi son el intervalo de ESA beta (el de la fila de
donde salio: una beta prestada no hereda el intervalo de la fila que la usa) y
delta_lo / delta_hi el mismo cambio en pulgadas calculado con esos extremos.

CONTROLES: velo_loss (arrastre: mph que pierde la bola de la mano al plato) y
dos PLACEBOS, spin rate y velocidad de salida, que no dependen del aire: su
beta mide cuanto sesgo de estadio le queda al diseno. Se reportan los dos, con
su intervalo y su tamano, salgan como salgan.

PARA LEER UNA BETA DEL IVB MAYOR QUE 1 hay dos filas mas. zone_time2 es el
tiempo de vuelo al cuadrado: en aire delgado la bola frena menos y llega
antes. Es DESCRIPTIVA y NO se suma al 1.0 de "fuerza proporcional a la
densidad": la fuerza de Magnus va con densidad x velocidad^2 x C_L y la
velocidad media es distancia / tiempo, asi que quiebre = 1/2 a t^2 va con
densidad x C_L x distancia^2. Lo que el vuelo mas corto le quita al quiebre se
lo devuelve la mayor presion dinamica (la velocidad media al cuadrado tiene,
salvo por la extension, la beta de zone_time2 con el signo contrario).
`sustentacion` lo mide directo en las rectas: aceleracion vertical inducida
(az0 + g) entre velocidad media al cuadrado, es decir densidad x C_L sin tiempo
de vuelo ni velocidad; si la fuerza fuera proporcional a la densidad su beta
seria ~1. Cuando sale tan arriba de 1 como la del IVB, el exceso NO es tiempo
de vuelo: queda sin explicar (puede ser calibracion de cada parque).

CON POCOS ESTADIOS (las 3 categorias del dataset oficial) el jackknife es muy
ancho (t con 2 grados de libertad, y quitar un nivel deja poco contraste de
densidad): puede dejar `significativo` apagado en un efecto real, y ahi un
placebo "no significativo" no prueba nada. Por eso cada fila lleva ademas
signo_estable: quitando cualquier estadio y remuestreando pitchers, la beta no
cambia de signo. No es una prueba de significancia; dice si la DIRECCION
depende de un solo nivel de densidad.
"""
import json
import time
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from scipy.optimize import minimize_scalar

# config.OUT se lee al usarse, para que una prueba pueda apuntar a otra carpeta
import config
from config import RANDOM_STATE, RAPIDO, RHO_CDMX, RHO_MAR, BETA_LIMITES, MIN_PITCHERS_BETA
from features import TIPOS, es_jonron

MOVIMIENTO = ("ivb", "hb_n", "mov_total")
# Tipos que entran en la MEDIA de cada familia, y de que familia toma prestado
# cada tipo. FC no esta en ninguna media (ver FILAS en el docstring).
FAMILIAS = {"FAM_RECTAS": ("FF", "SI"), "FAM_QUIEBRE": ("SL", "CU"), "FAM_CAMBIO": ("CH", "FS")}
FAMILIA_DE = {"FF": "FAM_RECTAS", "SI": "FAM_RECTAS", "FC": "FAM_RECTAS", "SL": "FAM_QUIEBRE",
              "CU": "FAM_QUIEBRE", "CH": "FAM_CAMBIO", "FS": "FAM_CAMBIO"}

MIN_CELDA = 15             # pitcheos para que un grupo x estadio cuente como celda
MIN_PITCHERS = 5           # con menos pitchers no se estima nada
MIN_ESTADIOS_IC = 6        # con menos estadios el intervalo es solo orientativo
NIVEL_MIN = 3.0            # pulgadas: debajo de esto el quiebre medio es ~0 y beta no esta identificada
SIGNO_MAX = 0.25           # fraccion maxima del peso con el signo contrario al de la mayoria
BUSQUEDA = (-1.0, 4.0)     # intervalo donde se busca beta
N_BOOT = 70 if RAPIDO else 200

NECESARIAS = ["pitcher_id", "season", "pt", "stadium", "altitude_m", "air_density", "banda",
              "ivb", "hb_n", "mov_total", "velocity", "spin_rate", "extension", "release_height",
              "rel_side_n", "outcome", "run_value", "y_swing", "y_whiff", "y_bip", "y_gb"]
OPCIONALES = ["game_id", "plate_speed", "zone_time", "az0", "exit_velo", "launch_angle", "hit_distance",
              "play_result", "home_run"]
CONTROLES = ("velo_loss", "zone_time2")
PLACEBOS = ("spin_rate", "velocity")
LECTURA = ("FAM_RECTAS", "FF")       # filas con el control `sustentacion` (aceleracion inducida positiva)
GRAVEDAD = 32.174                    # pies/s2
DIST_PLATO = 60.5 - 17 / 12          # pies de la goma al frente del plato, donde termina ZoneTime (diccionario oficial)

COLUMNAS = ["var", "pt", "beta", "se", "ci_lo", "ci_hi", "ic_fiable", "ci_lo_boot", "ci_hi_boot",
            "loo_min", "loo_max", "n_clusters", "estimable", "significativo", "pitchers", "celdas",
            "pitcheos", "media_liga", "media_mar", "media_cdmx", "delta_cdmx_vs_mar", "beta_usada",
            "fuente", "placebo",
            # diagnosticos de por que una fila es o no estimable / propia
            "nivel_abs", "signo_opuesto", "peso_propio", "motivo",
            # direccion estable (ver el docstring) e intervalo de lo que SI se uso
            "signo_estable", "beta_usada_lo", "beta_usada_hi", "ic_fiable_usada", "delta_lo", "delta_hi"]
FILA_VACIA = {"beta": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "ic_fiable": False,
              "ci_lo_boot": np.nan, "ci_hi_boot": np.nan, "loo_min": np.nan, "loo_max": np.nan,
              "n_clusters": 0, "estimable": False, "significativo": False, "pitchers": 0, "celdas": 0,
              "pitcheos": 0, "media_liga": np.nan, "media_mar": np.nan, "media_cdmx": np.nan,
              "delta_cdmx_vs_mar": np.nan, "beta_usada": np.nan, "fuente": "no_estimable",
              "placebo": False, "nivel_abs": np.nan, "signo_opuesto": np.nan, "peso_propio": np.nan,
              "motivo": "", "signo_estable": False, "beta_usada_lo": np.nan, "beta_usada_hi": np.nan,
              "ic_fiable_usada": False, "delta_lo": np.nan, "delta_hi": np.nan}

MECANICA = [("velocity", "Velocidad", "mph", 1), ("spin_rate", "Spin", "rpm", 1),
            ("extension", "Extensión", "pulg", 12), ("release_height", "Altura de salida", "pulg", 12),
            ("rel_side_n", "Lado de salida", "pulg", 12)]
COLS_ADAPTACION = ["tipo", "variable", "unidad", "diferencia", "ic_lo", "ic_hi", "pitchers",
                   "significativo", "n_temporadas", "nota"]
# la velocidad y el spin de la mano son tambien los placebos de la tabla de elasticidades
NOTA_PLACEBO = "misma cantidad que el placebo: no separa adaptación del pitcher de calibración del parque"


def _ascii(x):
    """Texto sin acentos para la consola de Windows."""
    return unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()


# ── datos ────────────────────────────────────────────────────────────────────
def leer():
    """out/features.parquet, solo las columnas que usa este paso (el archivo trae mas de 120)."""
    ruta = config.OUT / "features.parquet"
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}: corre primero  {config.comando('features')}")
    hay = set(pq.read_schema(ruta).names)
    faltan = [c for c in NECESARIAS if c not in hay]
    if faltan:
        raise SystemExit(f"A {ruta.name} le faltan las columnas {faltan}: vuelve a correr  "
                         + config.comando("features"))
    return preparar(pd.read_parquet(ruta, columns=NECESARIAS + [c for c in OPCIONALES if c in hay]))


def preparar(df):
    """Columnas auxiliares y llaves enteras (una sola vez): despues todo el
    remuestreo trabaja con numpy sobre la tabla de celdas, nunca sobre pitcheos."""
    df = df.copy()
    df["x"] = np.log(df["air_density"].astype(float))
    llegada = pd.to_numeric(df["plate_speed"], errors="coerce") if "plate_speed" in df else np.nan
    df["velo_loss"] = df["velocity"] - llegada
    vuelo = (pd.to_numeric(df["zone_time"], errors="coerce") if "zone_time" in df
             else pd.Series(np.nan, index=df.index))
    vuelo = vuelo.where(vuelo > 0)
    df["zone_time2"] = vuelo ** 2
    # aceleracion vertical inducida entre velocidad media^2 (x 1000 para leerla): ver el docstring
    inducida = pd.to_numeric(df["az0"], errors="coerce") + GRAVEDAD if "az0" in df else np.nan
    df["sustentacion"] = 1000 * inducida / ((DIST_PLATO - df["extension"]) / vuelo) ** 2
    df["pit"] = pd.factorize(df["pitcher_id"])[0]
    df["est"] = pd.factorize(df["stadium"])[0]
    df["tipo"] = pd.factorize(df["pt"])[0]
    df["g_tipo"] = df.groupby(["pitcher_id", "season", "pt"], sort=False).ngroup()
    df["g_temp"] = df.groupby(["pitcher_id", "season"], sort=False).ngroup()
    return df


def _tipos_de(pt):
    """Tipos de pitcheo que entran en una fila: un tipo, una familia o TODOS."""
    return FAMILIAS.get(pt, TIPOS if pt == "TODOS" else (pt,))


def celdas(df, var, pt="TODOS"):
    """Tabla de celdas grupo x estadio de `var`: y = media CON SIGNO, x = media
    de log(densidad), n = pitcheos. Grupo = pitcher x temporada x tipo; en las
    familias, pitcher x temporada (sin etiqueta). Solo celdas con MIN_CELDA
    pitcheos y grupos con dos o mas estadios de densidad distinta."""
    grupo = "g_temp" if pt in FAMILIAS else "g_tipo"
    d = df.loc[df["pt"].isin(_tipos_de(pt)) & np.isfinite(df[var]), [grupo, "est", "pit", "tipo", "x", var]]
    c = (d.groupby([grupo, "est"], sort=False)
           .agg(y=(var, "mean"), x=("x", "mean"), n=(var, "size"), pit=("pit", "first"), tipo=("tipo", "first"))
           .reset_index().rename(columns={grupo: "g"}))
    c = c[c["n"] >= MIN_CELDA]
    c = c[_con_contraste(pd.factorize(c["g"])[0], c["x"].to_numpy())]
    # `tipo` es con quien se compara el signo del grupo: en una familia, con todos
    return c.assign(g=pd.factorize(c["g"])[0], tipo=0 if pt in FAMILIAS else c["tipo"]).reset_index(drop=True)


# ── estimador ────────────────────────────────────────────────────────────────
def _con_contraste(g, x):
    """Celdas de grupos con dos o mas estadios de densidad distinta: los demas
    no dicen nada de beta (su nivel M_g absorbe todo)."""
    if not len(g):
        return np.zeros(0, dtype=bool)
    lo, hi = np.full(g.max() + 1, np.inf), np.full(g.max() + 1, -np.inf)
    np.minimum.at(lo, g, x)
    np.maximum.at(hi, g, x)
    return (hi - lo)[g] > 1e-9


def _sumas(beta, g, x, y, n, G):
    """Por grupo: sum(n y r) y sum(n r^2), con r = exp(beta x). Su cociente es M_g."""
    r = np.exp(beta * x)
    return np.bincount(g, n * y * r, G), np.bincount(g, n * r * r, G)


def ajustar(g, x, y, n, peso=None):
    """beta que minimiza la suma de cuadrados ponderada de y = M_g * exp(beta x).
    Para cada beta el mejor M_g tiene formula cerrada (M_g = sum(n y r) / sum(n r^2)),
    asi que solo se busca en una dimension. Con M_g perfilado,
        SSE(beta) = sum(n y^2) - sum_g [sum(n y r)^2 / sum(n r^2)]
    y basta maximizar el segundo termino. `peso`: veces que entra cada GRUPO
    (bootstrap de pitchers). x debe venir centrada."""
    if not len(g):
        return np.nan
    G = g.max() + 1 if peso is None else len(peso)

    def menos_explicado(beta):
        sxy, srr = _sumas(beta, g, x, y, n, G)
        q = np.divide(sxy * sxy, srr, out=np.zeros(G), where=srr > 0)
        return -float(q.sum() if peso is None else q @ peso)

    return float(minimize_scalar(menos_explicado, bounds=BUSQUEDA, method="bounded",
                                 options={"xatol": 1e-6}).x)


def jackknife(est, replicas):
    """Error estandar e IC 95% con las replicas 'sin el estadio k':
    se = sqrt((K-1)/K * sum((b_k - media)^2)), IC = est +/- t(K-1) * se."""
    r = np.asarray(replicas, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or not np.isfinite(est):
        return np.nan, np.nan, np.nan
    se = float(np.sqrt((len(r) - 1) / len(r) * np.sum((r - r.mean()) ** 2)))
    t = float(stats.t.ppf(0.975, len(r) - 1))
    return se, float(est - t * se), float(est + t * se)


def elasticidad(df, var, pt="TODOS", n_boot=N_BOOT):
    """Una fila de la tabla: elasticidad de `var` a la densidad para un tipo de
    pitcheo, una familia (FAM_*) o toda la liga (TODOS). Nunca devuelve None: si
    no se puede estimar, la fila sale con beta vacia y estimable = False."""
    fila = dict(FILA_VACIA, var=var, pt=pt)
    if var not in df or not df[var].notna().any():
        return fila
    c = celdas(df, var, pt)
    g, y, n, est = c["g"].to_numpy(), c["y"].to_numpy(), c["n"].to_numpy(float), c["est"].to_numpy()
    pit = pd.factorize(c["pit"])[0]
    P, estadios = (pit.max() + 1 if len(c) else 0), np.unique(est)
    fila.update(pitchers=int(P), celdas=int(len(c)), pitcheos=int(n.sum()), n_clusters=int(len(estadios)))
    if P < MIN_PITCHERS:
        return fila
    x = c["x"].to_numpy() - np.average(c["x"], weights=n)     # centrada: M_g = nivel a la densidad media
    G = g.max() + 1
    beta = ajustar(g, x, y, n)

    # ¿se puede leer como elasticidad? Si el nivel de los grupos es ~0 o cambia
    # de signo de un pitcher a otro (HB del cutter, IVB del slider), no hay
    # nivel que escalar. El signo se compara dentro de cada tipo de pitcheo: en
    # TODOS conviven a proposito pitcheos de signo contrario.
    sxy, srr = _sumas(beta, g, x, y, n, G)
    nivel, peso_g = sxy / srr, np.bincount(g, n, G)
    tipo_g = np.zeros(G, dtype=int)
    tipo_g[g] = c["tipo"].to_numpy()
    positivo, total = np.bincount(tipo_g, peso_g * (nivel > 0)), np.bincount(tipo_g, peso_g)
    nivel_abs = float(np.average(np.abs(nivel), weights=peso_g))
    opuesto = float(np.minimum(positivo, total - positivo).sum() / peso_g.sum())
    fila.update(nivel_abs=nivel_abs, signo_opuesto=opuesto)
    if min(beta - BUSQUEDA[0], BUSQUEDA[1] - beta) < 1e-3:
        return fila                        # la busqueda acabo en un extremo: no hay optimo, no hay beta
    estimable = bool(opuesto <= SIGNO_MAX and (var not in ("ivb", "hb_n") or nivel_abs >= NIVEL_MIN))

    # jackknife por estadio: se quita uno, se vuelve a pedir contraste y se reajusta
    sin_k = []
    for k in estadios:
        m = est != k
        m[m] = _con_contraste(g[m], x[m])
        sin_k.append(ajustar(g[m], x[m], y[m], n[m]))
    se, lo, hi = jackknife(beta, sin_k)
    completas = bool(np.isfinite(sin_k).all())

    # bootstrap de pitchers: cada grupo entra tantas veces como salio su pitcher
    pit_g = np.zeros(G, dtype=int)
    pit_g[g] = pit
    rng = np.random.default_rng(RANDOM_STATE)
    boot = [ajustar(g, x, y, n, rng.multinomial(P, np.full(P, 1.0 / P))[pit_g].astype(float))
            for _ in range(n_boot)]
    lo_b, hi_b = np.percentile(boot, [2.5, 97.5]) if n_boot else (np.nan, np.nan)
    # misma direccion sin cualquiera de los estadios y en el remuestreo de pitchers (NaN no cuenta como igual)
    estable = bool(completas and beta != 0 and all(np.sign(v) == np.sign(beta) for v in (*sin_k, lo_b, hi_b)))

    fila.update(beta=beta, se=se, ci_lo=lo, ci_hi=hi, ci_lo_boot=float(lo_b), ci_hi_boot=float(hi_b),
                ic_fiable=bool(len(estadios) >= MIN_ESTADIOS_IC and completas),
                loo_min=float(np.nanmin(sin_k)) if np.isfinite(sin_k).any() else np.nan,
                loo_max=float(np.nanmax(sin_k)) if np.isfinite(sin_k).any() else np.nan,
                estimable=estimable, signo_estable=estable,
                significativo=bool(estimable and P >= MIN_PITCHERS_BETA and (lo > 0 or hi < 0)))
    return fila


# ── que elasticidad se usa ───────────────────────────────────────────────────
def elegir_betas(res):
    """Llena beta_usada, fuente y peso_propio. Para cada tipo y variable de movimiento:
      'propia_encogida'  estimable y con MIN_PITCHERS_BETA pitchers: su beta
            encogida hacia la de su familia (Bayes empirico),
                b_fam + k (b - b_fam),   k = tau2 / (tau2 + se^2) = peso_propio
            tau2 = cuanto difieren DE VERDAD los tipos de su familia, quitado el
            ruido: max(0, media(d^2 - se^2)) ponderada por pitchers. Si tau2 = 0
            los tipos no se distinguen de su familia y pasan al caso 'familia'.
      'mov_total'  no estimable por quiebre ~0 o de signo mixto: la beta_usada
            del movimiento total de ese mismo tipo (ahi si hay nivel que escalar).
      'familia'  pocos pitchers, o sin diferencia medible: la beta de su familia.
      'liga'     la familia tampoco es usable: la de toda la liga (TODOS).
      'fisica'   nada usable: 1.0, fuerza proporcional a la densidad.
    Familias y liga llevan 'propia' si su beta es usable. Al final todo se
    recorta a config.BETA_LIMITES. Los controles y los placebos llevan su beta
    sin tocar: no mueven ningun pitcheo.

    Tambien llena el intervalo de la beta QUE SE USA (beta_usada_lo / _hi, con
    ic_fiable_usada): el de la fila de donde salio (familia, liga, mov_total),
    recortado igual que ella; para 'propia_encogida', la combinacion de los dos
    (k veces el propio y 1 - k veces el de su familia, sumados en cuadratura);
    para 'fisica' no hay. Y `motivo`, por que la fila no usa su propia beta:
    pocos_pitchers | no_estimable | igual_a_familia (tau2 = 0: no se distingue
    de su familia o, si esa no es usable, de la liga; `fuente` dice de cual) |
    sin_error_estandar | recortada (a BETA_LIMITES) | vacio (usa la suya)."""
    res = res.copy()
    f = res.set_index(["var", "pt"])
    usable = (f["estimable"] & (f["pitchers"] >= MIN_PITCHERS_BETA) & np.isfinite(f["beta"])).to_dict()
    b, se = f["beta"].to_dict(), f["se"].to_dict()
    propio = {k: (lo, hi, bool(ok)) for k, lo, hi, ok in zip(f.index, f["ci_lo"], f["ci_hi"], f["ic_fiable"])}
    sin_ic = (np.nan, np.nan, False)
    usada = {k: (b[k], "propia", 1.0) for k in b if k[0] not in MOVIMIENTO and np.isfinite(b[k])}
    ic = {k: propio[k] for k in usada}                # intervalo de la beta que se usa, con la misma llave

    for var in ("mov_total", "ivb", "hb_n"):          # mov_total primero: es el respaldo de ivb / hb_n
        liga = (b[var, "TODOS"], "propia", 1.0) if usable[var, "TODOS"] else (1.0, "fisica", 0.0)
        usada[var, "TODOS"], ic[var, "TODOS"] = liga, propio[var, "TODOS"] if liga[1] == "propia" else sin_ic
        for fam in FAMILIAS:
            if usable[var, fam]:
                usada[var, fam], ic[var, fam] = (b[var, fam], "propia", 1.0), propio[var, fam]
            else:
                usada[var, fam] = (liga[0], "liga" if liga[1] == "propia" else "fisica", 0.0)
                ic[var, fam] = ic[var, "TODOS"]
        meta = {pt: usada[var, FAMILIA_DE[pt]] for pt in TIPOS}
        con = [pt for pt in TIPOS if usable[var, pt] and se[var, pt] > 0]
        tau2 = 0.0
        if con:
            w = np.array([f.loc[(var, pt), "pitchers"] for pt in con], dtype=float)
            d2 = np.array([(b[var, pt] - meta[pt][0]) ** 2 for pt in con])
            tau2 = max(0.0, float(np.average(d2 - np.array([se[var, pt] ** 2 for pt in con]), weights=w)))
        print(f"  {var}: varianza real entre tipos (tau2) = {tau2:.4f} con {len(con)} tipos estimables")
        for pt in TIPOS:
            b_fam, f_fam, _ = meta[pt]
            lo_fam, hi_fam, ok_fam = ic[var, FAMILIA_DE[pt]]
            k = tau2 / (tau2 + se[var, pt] ** 2) if pt in con else 0.0
            if k > 0:
                valor, (lo, hi, ok) = b_fam + k * (b[var, pt] - b_fam), propio[var, pt]
                usada[var, pt] = (valor, "propia_encogida", k)
                ic[var, pt] = (valor - np.hypot(k * (b[var, pt] - lo), (1 - k) * (b_fam - lo_fam)),
                               valor + np.hypot(k * (hi - b[var, pt]), (1 - k) * (hi_fam - b_fam)), ok and ok_fam)
            elif var != "mov_total" and not f.loc[(var, pt), "estimable"] and f.loc[("mov_total", pt), "estimable"]:
                usada[var, pt], ic[var, pt] = (usada["mov_total", pt][0], "mov_total", 0.0), ic["mov_total", pt]
            else:
                usada[var, pt] = (b_fam, "familia" if f_fam == "propia" else f_fam, 0.0)
                ic[var, pt] = (lo_fam, hi_fam, ok_fam)

    for i, r in res.iterrows():
        llave = (r["var"], r["pt"])
        valor, fuente, k = usada.get(llave, (np.nan, "no_estimable", np.nan))
        lo, hi, ok = ic.get(llave, sin_ic)
        motivo = ""
        if r["var"] in MOVIMIENTO:
            recortada = float(np.clip(valor, *BETA_LIMITES))
            if not fuente.startswith("propia") or recortada != valor:
                motivo, texto = _motivo(r, fuente)
                print("  " + _ascii(f"{r['pt']:11s} {r['var']:9s}: beta propia {r['beta']:6.2f} "
                                    f"({texto}) -> usa {recortada:.2f} [{fuente}]"
                                    + (f", recortada desde {valor:.2f}" if recortada != valor else "")))
            valor, lo, hi = recortada, float(np.clip(lo, *BETA_LIMITES)), float(np.clip(hi, *BETA_LIMITES))
        for col, v in (("beta_usada", valor), ("fuente", fuente), ("peso_propio", k), ("motivo", motivo),
                       ("beta_usada_lo", lo), ("beta_usada_hi", hi), ("ic_fiable_usada", bool(ok))):
            res.at[i, col] = v
    return res


def _motivo(r, fuente):
    """Por que una fila no usa su propia beta: (clave para la columna `motivo`
    del CSV, texto para el mensaje de consola)."""
    if fuente.startswith("propia"):
        return "recortada", "fuera de los limites de cordura"
    if r["pitchers"] < MIN_PITCHERS:
        return "pocos_pitchers", f"{int(r['pitchers'])} pitchers: no se estima"
    if not r["estimable"]:
        return "no_estimable", (f"no estimable: nivel medio {r['nivel_abs']:.1f}, "
                                f"{100 * r['signo_opuesto']:.0f}% del peso con signo contrario"
                                + ("" if np.isfinite(r["beta"]) else ", sin optimo en el intervalo de busqueda"))
    if r["pitchers"] < MIN_PITCHERS_BETA:
        return "pocos_pitchers", f"{int(r['pitchers'])} pitchers, se piden {MIN_PITCHERS_BETA}"
    if not r["se"] > 0:
        return "sin_error_estandar", "sin error estandar"
    return "igual_a_familia", "no se distingue de su familia"


def niveles(df, res):
    """media_liga: media con signo en el aire donde se lanzo cada pitcheo.
    media_mar / media_cdmx: esos mismos pitcheos llevados a RHO_MAR / RHO_CDMX
    con beta_usada. delta_cdmx_vs_mar: la diferencia, en unidades de la variable.
    delta_lo / delta_hi: la misma diferencia con los extremos del intervalo de
    beta_usada (solo la incertidumbre de la elasticidad; la del nivel medio es
    despreciable a su lado)."""
    res = res.copy()
    rho = df["air_density"].to_numpy(float)
    for i, r in res.iterrows():
        if r["var"] not in df:
            continue
        v = df[r["var"]].to_numpy(float)
        m = df["pt"].isin(_tipos_de(r["pt"])).to_numpy() & np.isfinite(v)
        if not m.any():
            continue
        res.loc[i, "media_liga"] = v[m].mean()
        llevar = lambda destino, beta: float((v[m] * (destino / rho[m]) ** beta).mean())
        if np.isfinite(r["beta_usada"]):
            mar, cdmx = llevar(RHO_MAR, r["beta_usada"]), llevar(RHO_CDMX, r["beta_usada"])
            res.loc[i, ["media_mar", "media_cdmx", "delta_cdmx_vs_mar"]] = [mar, cdmx, cdmx - mar]
            extremos = (r["beta_usada_lo"], r["beta_usada_hi"])
            if np.isfinite(extremos).all():
                res.loc[i, ["delta_lo", "delta_hi"]] = sorted(
                    llevar(RHO_CDMX, beta) - llevar(RHO_MAR, beta) for beta in extremos)
    return res


def tabla_elasticidades(df):
    """Todas las filas de out/altitude_elasticities.csv."""
    filas = [elasticidad(df, var, pt) for pt in TIPOS for var in MOVIMIENTO]
    filas += [elasticidad(df, var, "TODOS") for var in MOVIMIENTO]
    filas += [elasticidad(df, var, fam) for fam in FAMILIAS for var in MOVIMIENTO]
    filas += [elasticidad(df, var) for var in CONTROLES]
    filas += [dict(elasticidad(df, var), placebo=True) for var in PLACEBOS]
    filas += [elasticidad(df, "sustentacion", pt) for pt in LECTURA]
    print("Elasticidad que usa el contrafactual (beta_usada):")
    return niveles(df, elegir_betas(pd.DataFrame(filas)))[COLUMNAS]


# ── batazos ──────────────────────────────────────────────────────────────────
def _pendiente_en_bins(b, x, y):
    """Pendiente de y sobre x quitando la media de cada bin."""
    if not len(b):
        return np.nan
    n = np.maximum(np.bincount(b), 1)
    xd, yd = x - (np.bincount(b, x) / n)[b], y - (np.bincount(b, y) / n)[b]
    den = float(xd @ xd)
    return float(xd @ yd) / den if den > 0 else np.nan


def batted_ball_carry(df):
    """A igual velocidad de salida y angulo, ¿cuanto mas viaja un elevado en el
    Harp Helu que al nivel del mar? Distancia sobre densidad dentro de bins de
    3 mph x 5 grados. Es la diferencia TOTAL entre parques a igual batazo (sobre
    todo densidad; tambien viento y humedad de la bola, que el dataset no trae
    y no se pueden separar).
    IC: jackknife por estadio."""
    out = {"pies_extra_cdmx_vs_mar": np.nan, "ic95": [np.nan, np.nan], "ic_fiable": False,
           "batazos": 0, "n_clusters": 0}
    if any(c not in df for c in ("exit_velo", "launch_angle", "hit_distance")):
        return out
    d = df.loc[df["y_bip"] == 1, ["exit_velo", "launch_angle", "hit_distance", "air_density", "est"]]
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    d = d[d["launch_angle"].between(15, 45) & (d["exit_velo"] >= 80)]
    estadios = d["est"].unique()
    out.update(batazos=int(len(d)), n_clusters=int(len(estadios)))
    if len(d) < 200 or len(estadios) < 2:
        return out
    b = pd.factorize((d["exit_velo"] // 3) * 100 + d["launch_angle"] // 5)[0]
    x, y, est = d["air_density"].to_numpy(float), d["hit_distance"].to_numpy(float), d["est"].to_numpy()
    paso = RHO_CDMX - RHO_MAR                                   # negativo: menos aire
    pies = _pendiente_en_bins(b, x, y) * paso
    sin_k = [_pendiente_en_bins(pd.factorize(b[est != k])[0], x[est != k], y[est != k]) * paso for k in estadios]
    _, lo, hi = jackknife(pies, sin_k)
    out.update(pies_extra_cdmx_vs_mar=pies, ic95=[lo, hi],
               ic_fiable=bool(len(estadios) >= MIN_ESTADIOS_IC and np.isfinite(sin_k).all()))
    return out


# ── adaptacion del pitcher ───────────────────────────────────────────────────
def _boot_pitcher(pitcher, dif, w, n_boot=N_BOOT):
    """Media ponderada de `dif` con IC 95% remuestreando pitchers completos
    (con las sumas por pitcher: no hace falta rearmar la tabla)."""
    cod = pd.factorize(np.asarray(pitcher))[0]
    P = cod.max() + 1
    num, den = np.bincount(cod, np.asarray(w, float) * np.asarray(dif, float), P), np.bincount(cod, w, P)
    veces = np.random.default_rng(RANDOM_STATE).multinomial(P, np.full(P, 1.0 / P), size=n_boot)
    lo, hi = np.percentile((veces @ num) / (veces @ den), [2.5, 97.5])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def adaptacion(df, min_n=MIN_CELDA, min_mezcla=60, min_pitchers=20):
    """¿El pitcher CAMBIA en altitud? Cada pitcher-temporada contra si mismo,
    banda 'alta' menos banda 'mar' (columna banda de features):
      Mecánica  velocidad, spin, extension, altura y lado de salida del mismo
                pitcheo (celdas pitcher-temporada-tipo con min_n en las dos bandas)
      Mezcla    puntos de uso de cada tipo sobre un conjunto FIJO de
                pitcher-temporadas (min_mezcla pitcheos en las dos bandas); un
                tipo que no se lanzo cuenta como 0%, asi que las diferencias suman 0
      Etiqueta  ¿cambia la etiqueta AutoPitchType de cada pitcheo? Parte de FF
                dentro de FF+SI, y cambio de velocidad del 'slider' menos el de
                la recta: si la etiqueta fuera estable, ambos serian ~0.
    IC: bootstrap de pitchers. Con solo dos bandas no hay como agrupar por
    estadio, asi que estos intervalos NO incluyen el sesgo de medicion de cada
    parque (calibracion del sistema de medicion): una diferencia chica de
    mecanica puede ser eso. En particular Velocidad y Spin son la misma
    cantidad que los dos placebos de la tabla de elasticidades (columna nota):
    un cambio ahi no distingue al pitcher que se adapta del parque que mide
    distinto."""
    d = df[df["banda"].isin(["alta", "mar"])]
    d = d.assign(alta=(d["banda"] == "alta").astype(int))
    filas = []
    if d["alta"].nunique() < 2:
        return pd.DataFrame(filas, columns=COLS_ADAPTACION)

    def fila(tipo, variable, unidad, dif, w, nota=""):
        """dif y w: Series con pitcher_id y season en el indice."""
        dif, w = dif.dropna(), w[dif.notna()]
        pitcher = dif.index.get_level_values("pitcher_id")
        if pitcher.nunique() < min_pitchers:
            return
        est, lo, hi = _boot_pitcher(pitcher, dif, w)
        filas.append({"tipo": tipo, "variable": variable, "unidad": unidad, "diferencia": est,
                      "ic_lo": lo, "ic_hi": hi, "pitchers": int(pitcher.nunique()),
                      "significativo": bool(lo > 0 or hi < 0),
                      "n_temporadas": len(set(zip(pitcher, dif.index.get_level_values("season")))),
                      "nota": nota})

    # mecanica del mismo pitcheo
    cambio = {}
    mec = [m for m in MECANICA if m[0] in d]
    c = d.groupby(["pitcher_id", "season", "pt", "alta"])[[m[0] for m in mec]].agg(["mean", "count"]).unstack("alta")
    for var, nombre, unidad, k in mec:
        media, n = c[var]["mean"], c[var]["count"]
        ok = (n[0] >= min_n) & (n[1] >= min_n)
        cambio[var] = (k * (media[1] - media[0])[ok], np.minimum(n[0], n[1])[ok])
        fila("Mecánica", nombre, unidad, *cambio[var], nota=NOTA_PLACEBO if var in PLACEBOS else "")

    # mezcla sobre un conjunto fijo de pitcher-temporadas
    n = (d.groupby(["pitcher_id", "season", "alta", "pt"]).size().unstack("pt", fill_value=0)
          .reindex(columns=list(TIPOS), fill_value=0))
    tot = n.sum(axis=1).unstack("alta")
    fijos = tot.index[(tot[0] >= min_mezcla) & (tot[1] >= min_mezcla)]
    uso = n.div(n.sum(axis=1), axis=0)
    dif = 100 * (uso.xs(1, level="alta").reindex(fijos) - uso.xs(0, level="alta").reindex(fijos))
    for pt in TIPOS:
        fila("Mezcla", pt, "puntos %", dif[pt], tot.loc[fijos].min(axis=1))
    suma = sum(f["diferencia"] for f in filas if f["tipo"] == "Mezcla")
    assert abs(suma) < 1e-6, f"las diferencias de mezcla deben sumar 0 y suman {suma}"

    # estabilidad de la etiqueta AutoPitchType
    rectas = (n["FF"] + n["SI"]).unstack("alta")
    ok = (rectas[0] >= 2 * min_n) & (rectas[1] >= 2 * min_n)
    parte_ff = (n["FF"] / (n["FF"] + n["SI"])).unstack("alta")
    fila("Etiqueta", "FF dentro de FF+SI", "puntos %", 100 * (parte_ff[1] - parte_ff[0])[ok], rectas.min(axis=1)[ok])
    if "velocity" in cambio:
        dv, w = (s.unstack("pt") for s in cambio["velocity"])
        if {"SL", "FF"} <= set(dv.columns):
            fila("Etiqueta", "Velocidad SL menos FF", "mph", dv["SL"] - dv["FF"], w[["SL", "FF"]].min(axis=1))
    return pd.DataFrame(filas, columns=COLS_ADAPTACION)


# ── tablas por estadio ───────────────────────────────────────────────────────
def park_table(df):
    """Resumen por estadio (o por categoria de altitud): lo que el club ve en la
    cancha. Son promedios CRUDOS: incluyen al staff local, no solo al parque."""
    d = pd.DataFrame({
        "stadium": df["stadium"], "banda": df["banda"], "altitud": df["altitude_m"], "densidad": df["air_density"],
        "rv": df["run_value"], "hr": es_jonron(df) & (df["y_bip"] == 1),
        "fb": df["outcome"].isin(["ip_fb_soft", "ip_fb_hard"]),
        "gb": df["y_gb"], "ip": df["y_bip"], "swings": df["y_swing"], "whiffs": df["y_whiff"],
        "ivb_ff": df["ivb"].where(df["pt"] == "FF"), "loss": df["velo_loss"]})
    t = d.groupby("stadium").agg(
        banda=("banda", "first"), altitud=("altitud", "first"), densidad=("densidad", "mean"),
        pitcheos=("rv", "size"), rv_por_100=("rv", "mean"), hr=("hr", "sum"), fb=("fb", "sum"),
        gb=("gb", "sum"), ip=("ip", "sum"), swings=("swings", "sum"), whiffs=("whiffs", "sum"),
        ivb_ff=("ivb_ff", "mean"), perdida_velo=("loss", "mean")).reset_index()
    t["rv_por_100"] *= 100
    t["whiff"] = t["whiffs"] / t["pitcheos"]                 # por pitcheo (swinging strike)
    t["whiff_swing_pct"] = t["whiffs"] / t["swings"]         # por swing
    # todos los HR entre elevados (convencion usual); vacio si el dataset no marca los jonrones
    t["hr_por_fb"] = t["hr"] / t["fb"] if {"play_result", "home_run"} & set(df.columns) else np.nan
    t["gb_pct"] = t["gb"] / t["ip"]                          # rodados por batazo en juego
    return t.sort_values("altitud", ascending=False)


# ── corrida ──────────────────────────────────────────────────────────────────
def _control(res, var, pt="TODOS"):
    """Un control o placebo para el resumen: beta; ci_lo / ci_hi (jackknife por
    estadio, orientativo si ic_fiable es False); ci_*_boot (remuestreo de
    pitchers: ignora el sesgo de cada parque); loo_* (rango de la beta al quitar
    un estadio); delta_* (CDMX menos nivel del mar en unidades de la variable,
    con los extremos del jackknife)."""
    r = res[(res["var"] == var) & (res["pt"] == pt)].iloc[0]
    numeros = ("ci_lo_boot", "ci_hi_boot", "loo_min", "loo_max", "delta_lo", "delta_hi")
    return {"beta": float(r["beta"]), "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]),
            "ic_fiable": bool(r["ic_fiable"]), "significativo": bool(r["significativo"]),
            "delta_cdmx_vs_mar": float(r["delta_cdmx_vs_mar"]), **{c: float(r[c]) for c in numeros},
            "signo_estable": bool(r["signo_estable"]), "n_clusters": int(r["n_clusters"])}


def main():
    pd.set_option("display.width", 220)
    t0 = time.time()
    df = leer()
    n_est = int(df["stadium"].nunique())
    print(f"{len(df):,} pitcheos | modo categoria: {n_est} categorias de altitud")
    if n_est < MIN_ESTADIOS_IC:
        print(f"AVISO: con {n_est} niveles de densidad los intervalos de la fisica son solo "
              "orientativos (ic_fiable = False)")

    res = tabla_elasticidades(df)
    res.to_csv(config.OUT / "altitude_elasticities.csv", index=False)
    print("\nELASTICIDADES (IC = jackknife por estadio; delta_lo / delta_hi = el cambio con el intervalo de "
          "beta_usada):")
    print(_ascii(res[["pt", "var", "beta", "ci_lo", "ci_hi", "ic_fiable", "estimable", "significativo",
                      "signo_estable", "pitchers", "media_mar", "delta_cdmx_vs_mar", "delta_lo", "delta_hi",
                      "beta_usada", "fuente"]].round(3).to_string(index=False)))

    carry = batted_ball_carry(df)
    print("\nELEVADOS, pies extra en el Harp Helu vs nivel del mar:", carry)

    adapt = adaptacion(df)
    adapt.to_csv(config.OUT / "adaptacion.csv", index=False)
    print("\nADAPTACION (banda alta - banda mar, mismo pitcher):")
    print(_ascii(adapt.drop(columns="nota").round(3).to_string(index=False)) if len(adapt) else
          "  sin pitchers suficientes en las dos bandas")
    if len(adapt) and (adapt["nota"] != "").any():
        print("  OJO: Velocidad y Spin son la misma cantidad que los placebos: un cambio ahi puede ser "
              "calibracion del parque y no adaptacion del pitcher")

    parks = park_table(df)
    parks.to_csv(config.OUT / "park_table.csv", index=False)
    print("\nPARQUES (promedios crudos):")
    print(_ascii(parks[["stadium", "banda", "altitud", "densidad", "pitcheos", "rv_por_100", "hr_por_fb",
                        "gb_pct", "whiff_swing_pct", "ivb_ff", "perdida_velo"]].round(3).to_string(index=False)))

    perdida = _control(res, "velo_loss")
    perdida["mph_cdmx_vs_mar"] = perdida["delta_cdmx_vs_mar"]
    resumen = {"rho_cdmx": RHO_CDMX, "rho_mar": RHO_MAR, "rho_liga": float(df["air_density"].mean()),
               "ratio_cdmx_mar": RHO_CDMX / RHO_MAR, "carry": carry,
               "modo": "categoria", "n_estadios": n_est,
               "placebo": {v: _control(res, v) for v in PLACEBOS}, "velo_loss": perdida,
               "zone_time2": _control(res, "zone_time2"),
               "sustentacion": {pt: _control(res, "sustentacion", pt) for pt in LECTURA}}
    with open(config.OUT / "altitude_summary.json", "w") as fh:
        json.dump(resumen, fh, indent=1)
    print("\nCONTROLES: arrastre", perdida, "| tiempo de vuelo^2", resumen["zone_time2"],
          "| sustentacion", resumen["sustentacion"], "| placebos", resumen["placebo"])
    print(f"physics listo en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
