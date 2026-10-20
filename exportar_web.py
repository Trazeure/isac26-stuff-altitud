"""Datos para la pagina web de resultados (solo lee; no cambia nada del proyecto).

Uso, desde la carpeta del proyecto, DESPUES de la corrida completa:
    python exportar_web.py

Escribe web_datos.zip con JSON compactos: resumen de la validacion, efecto por
tipo, rankings por pitcher y por arsenal (ids anonimos de ISAC), trayectorias
promedio (ajuste de trayectoria de 9 parametros, columnas x0..az0 del dataset)
para el 3D, mapas de calor y una
rejilla de Stuff+ del simulador. No lleva pitcheos sueltos ni datos crudos.
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
import config                      # noqa: E402
import simulator as sim            # noqa: E402

OUT, MODELS = config.OUT, config.MODELS
TIPOS = ["FF", "SI", "FC", "SL", "CU", "CH", "FS"]
NUEVE = ["x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0"]
MIN_CELDA = 25
CELDA_MOV, CELDA_ZONA = 1.5, 0.25     # pulgadas de quiebre y pies de ubicacion por celda


def r(x, d=2):
    """Numero corto para JSON (None si no es finito)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, d) if np.isfinite(x) else None


def tabla(nombre, cols=None, dec=3):
    p = OUT / nombre
    if not p.exists():
        return []
    t = pd.read_csv(p)
    if cols:
        t = t[[c for c in cols if c in t.columns]]
    out = []
    for fila in t.to_dict("records"):
        out.append({k: (r(v, dec) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)
                        else (None if (isinstance(v, float) and not np.isfinite(v)) else v))
                    for k, v in fila.items()})
    return out


def limpiar(o):
    """Lo que json no acepta (NaN, infinitos, tipos de numpy) a valores simples."""
    if isinstance(o, dict):
        return {str(k): limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpiar(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if o is None or isinstance(o, str):
        return o
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return str(o)


def leer_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def resumen():
    info = leer_json(OUT / "run_info.json")
    pasos = info.get("pasos") or {}
    return {
        "codigo": (info.get("hash_codigo") or next((v.get("hash_codigo") for v in pasos.values()
                                                   if isinstance(v, dict) and v.get("hash_codigo")), "") or "")[:8],
        "modo_rapido": bool(config.RAPIDO),
        "equipo": config.EQUIPO, "integrantes": list(config.INTEGRANTES) if not isinstance(config.INTEGRANTES, str)
        else [config.INTEGRANTES], "institucion": config.INSTITUCION,
        "rho": {"cdmx": config.RHO_CDMX, "mar": config.RHO_MAR, "categorias": config.ALTITUDE_RHO,
                "metros": config.ALTITUDE_M, "casa": config.HOME_ALTITUDE},
        "validacion": tabla("val_resumen.csv", dec=5), "temporada": tabla("val_season_holdout.csv", dec=5),
        "parque_fuera": tabla("val_park_holdout.csv", dec=5), "calibracion": tabla("val_calibracion_resumen.csv", dec=5),
        "contrafactual": [f for f in tabla("val_contrafactual.csv", dec=5)
                          if f.get("pt") in ("TODOS", "FAM_RECTAS", "FAM_QUIEBRE", "FAM_CAMBIO", "REPERTORIO")
                          or f.get("bloque") == "entre_pitchers"],
        "elasticidades": tabla("altitude_elasticities.csv", dec=5), "por_tipo": tabla("altitude_by_pitch_type.csv", dec=5),
        "parques": tabla("park_table.csv", dec=5), "adaptacion": tabla("adaptacion.csv", dec=5),
        "h4": tabla("val_h4_ablation.csv", dec=5), "importancia": tabla("val_importancia_stuff.csv", dec=5),
        "formas": tabla("insights_formas.csv", dec=5), "perfil": tabla("insights_perfil.csv", dec=5),
        "mezcla": tabla("insights_mix.csv", dec=5), "efecto_parque": leer_json(OUT / "altitude_park_effect.json"),
        "estabilidad": leer_json(OUT / "val_stability.json"), "insights": leer_json(OUT / "insights.json"),
        "limpieza": leer_json(OUT / "limpieza.json"), "valores": leer_json(MODELS / "outcome_values.json"),
    }


COLS_PIT = ["pitcher_id", "season", "pitcheos", "rho", "pct_alta", "lefty", "velo", "velo_recta", "rv100",
            "swstr_pct", "whiff_swing_pct", "chase_pct", "gb_pct", "hard_pct",
            "stuff_plus_enc", "stuff_lo", "stuff_hi", "stuff_cdmx_plus_enc", "stuff_cdmx_lo", "stuff_cdmx_hi",
            "stuff_mar_plus", "location_plus_enc", "location_lo", "location_hi", "pitching_plus_enc", "pitching_lo",
            "pitching_hi", "pitching_neutral_plus_enc", "delta_altitud", "delta_lo", "delta_hi", "costo_cdmx_rv100"]
COLS_ARS = ["pitcher_id", "season", "pt", "pitcheos", "rho", "uso", "velo", "ivb", "hb", "ivb_cdmx", "hb_cdmx", "spin",
            "ext", "rel_h", "rel_s", "swstr_pct", "whiff_swing_pct", "gb_pct", "hard_pct", "stuff_plus_enc",
            "stuff_lo", "stuff_hi", "stuff_cdmx_plus_enc", "stuff_cdmx_lo", "stuff_cdmx_hi", "stuff_mar_plus",
            "delta_altitud", "delta_lo", "delta_hi", "costo_cdmx_rv100"]


def trayectorias(sp):
    """Promedio del ajuste de trayectoria de 9 parametros (columnas x0..az0 del dataset) por tipo y banda, y por
    arsenal, en el marco de un derecho (a los zurdos se les voltea lo horizontal). Sale del archivo oficial de
    ISAC en data/, unido a los scores por su PitchUID."""
    crudos = sorted(p for p in config.DATA_FILE.iterdir() if p.name.lower().endswith(".parquet"))
    import pyarrow.parquet as pq
    partes = []
    for p in crudos:
        hay = set(pq.read_schema(p).names)
        partes.append(pd.read_parquet(p, columns=[c for c in ["PitchUID"] + NUEVE if c in hay]))
    if not partes or not set(NUEVE) <= set(partes[0].columns):
        return {}, None
    t = pd.concat(partes, ignore_index=True).rename(columns={"PitchUID": "pitch_id"})
    t["pitch_id"] = t["pitch_id"].astype(str)
    d = sp[["pitch_id", "pitcher_id", "season", "pt", "banda", "air_density", "lefty"]].copy()
    d["pitch_id"] = d["pitch_id"].astype(str)
    d = d.merge(t, on="pitch_id", how="inner")
    for c in NUEVE:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=NUEVE)
    zurdo = d["lefty"].astype(int) == 1
    for c in ("x0", "vx0", "ax0"):
        d.loc[zurdo, c] = -d.loc[zurdo, c]
    tipos = {}
    for (pt, banda), g in d.groupby(["pt", "banda"]):
        if len(g) >= 200:
            tipos.setdefault(pt, {})[banda] = {"n": int(len(g)), "rho": r(g["air_density"].mean(), 4),
                                              **{c: r(g[c].mean(), 3) for c in NUEVE}}
    ars = d.groupby(["pitcher_id", "season", "pt"])
    medias = ars[NUEVE + ["air_density"]].mean()
    medias["n"] = ars.size()
    medias = medias[medias["n"] >= config.MIN_PITCHES_ARSENAL].reset_index()
    return tipos, medias


def mapas(sp):
    """Stuff+ medio por celda de quiebre (Harp Helu y nivel del mar) y Location+ / swing fallado por celda de
    ubicacion, por tipo de pitcheo. Solo celdas con MIN_CELDA+ pitcheos."""
    out = {"celda_mov": CELDA_MOV, "celda_zona": CELDA_ZONA, "tipos": {}}
    for pt in TIPOS:
        g = sp[sp["pt"] == pt]
        if len(g) < 500:
            continue
        m = {}
        for lugar, (ci, ch, cs) in {"cdmx": ("ivb_cdmx", "hb_cdmx", "stuff_cdmx_plus"),
                                    "mar": ("ivb_mar", "hb_mar", "stuff_mar_plus")}.items():
            if not {ci, ch, cs} <= set(g.columns):
                continue
            k = pd.DataFrame({"i": np.floor(g[ci] / CELDA_MOV), "h": np.floor(g[ch] / CELDA_MOV), "s": g[cs]}).dropna()
            c = k.groupby(["i", "h"])["s"].agg(["mean", "size"]).reset_index()
            c = c[c["size"] >= MIN_CELDA]
            m[lugar] = [[int(a), int(b), r(s, 1), int(n)] for a, b, s, n in c.itertuples(index=False)]
        if {"plate_x_n", "plate_z", "location_plus"} <= set(g.columns):
            k = pd.DataFrame({"x": np.floor(g["plate_x_n"] / CELDA_ZONA), "z": np.floor(g["plate_z"] / CELDA_ZONA),
                              "l": g["location_plus"], "w": g["y_whiff"] if "y_whiff" in g else np.nan,
                              "sw": g["y_swing"] if "y_swing" in g else np.nan}).dropna(subset=["x", "z", "l"])
            c = k.groupby(["x", "z"]).agg(l=("l", "mean"), w=("w", "sum"), sw=("sw", "sum"), n=("l", "size"))
            c = c[c["n"] >= MIN_CELDA].reset_index()
            m["zona"] = [[int(a), int(b), r(l, 1), r(w / sw if sw else np.nan, 3), int(n)]
                         for a, b, l, w, sw, n in c.itertuples(index=False)]
        out["tipos"][pt] = m
    return out


def rejilla_simulador(sp):
    """Stuff+ (puntos por pitcheo) del modelo del simulador sobre una rejilla de velocidad, IVB y HB por tipo,
    con lo demas en el pitcheo tipico de la liga y el quiebre medido a la densidad media de la liga. La pagina
    interpola en esta rejilla: no corre el modelo."""
    R = sim.recursos()
    rho_liga = R["valores"]["densidad"]["rho_liga"]
    out = {"rho_liga": rho_liga, "beta": {pt: {k: r(v, 4) for k, v in R["beta"].get(pt, {}).items()}
                                          for pt in TIPOS}, "tipos": {}}
    for pt in TIPOS:
        g = sp[sp["pt"] == pt]
        if len(g) < 500:
            continue
        tp = sim.tipico(pt, rho_liga)
        q = lambda c, a, b: np.nanpercentile(g[c], [a, b])          # noqa: E731
        v0, v1 = q("velocity", 3, 97)
        i0, i1 = q("ivb", 3, 97)
        h0, h1 = q("hb_n", 3, 97)
        V = np.round(np.linspace(v0, v1, 9) * 2) / 2
        Iv = np.round(np.linspace(i0, i1, 13) * 2) / 2
        Hb = np.round(np.linspace(h0, h1, 13) * 2) / 2
        vv, ii, hh = np.meshgrid(V, Iv, Hb, indexing="ij")
        n = vv.size
        recta_propia = pt in ("FF", "SI")
        filas = pd.DataFrame({
            "pt": pt, "velo": vv.ravel(), "ivb": ii.ravel(), "hb_brazo": hh.ravel(), "spin": tp["spin"],
            "rel_h": tp["rel_h"], "rel_s_brazo": tp["rel_s"], "ext": tp["ext"], "zurdo": False,
            "fb_tipo": None if recta_propia else tp["fb_tipo"],
            "fb_velo": None if recta_propia else tp["fb_velo"],
            "fb_ivb": None if recta_propia else tp["fb_ivb"],
            "fb_hb_brazo": None if recta_propia else tp["fb_hb"],
            "fb_rel_h": None, "fb_rel_s_brazo": None, "rel_h_tipo": None, "rel_s_tipo": None}, index=range(n))
        base = sim.filas_pitcheo(filas, rho_liga)
        cd = sim.calificar(base, config.RHO_CDMX)
        mr = sim.calificar(base, config.RHO_MAR)
        out["tipos"][pt] = {
            "velo": V.tolist(), "ivb": Iv.tolist(), "hb": Hb.tolist(),
            "tipico": {k: (r(v, 3) if isinstance(v, (int, float, np.floating)) else v) for k, v in tp.items()},
            "cdmx": [r(x, 1) for x in cd["stuff_plus"]], "mar": [r(x, 1) for x in mr["stuff_plus"]],
            "whiff_cdmx": [r(x, 3) for x in cd["whiff_por_swing"]], "whiff_mar": [r(x, 3) for x in mr["whiff_por_swing"]],
            "gb_cdmx": [r(x, 3) for x in cd["rodado_por_batazo"]], "gb_mar": [r(x, 3) for x in mr["rodado_por_batazo"]],
            "pct_cdmx": [r(x, 1) for x in cd["percentil"]], "pct_mar": [r(x, 1) for x in mr["percentil"]]}
        print(f"  rejilla {pt}: {n} celdas")
    return out


def main():
    p = OUT / "scores_pitch.parquet"
    if not p.exists():
        raise SystemExit("Falta out/scores_pitch.parquet: corre primero  python run_all.py")
    cols = ["pitch_id", "pitcher_id", "season", "pt", "banda", "air_density", "lefty", "velocity", "ivb", "hb_n",
            "ivb_cdmx", "hb_cdmx", "ivb_mar", "hb_mar", "stuff_cdmx_plus", "stuff_mar_plus", "plate_x_n", "plate_z",
            "location_plus", "y_whiff", "y_swing"]
    import pyarrow.parquet as pq
    hay = set(pq.read_schema(p).names)
    sp = pd.read_parquet(p, columns=[c for c in cols if c in hay])
    print(f"scores_pitch: {len(sp):,} pitcheos")
    datos = {"resumen.json": resumen()}
    pit = tabla("scores_pitcher.csv", COLS_PIT)
    datos["pitchers.json"] = pit
    tipos3d, medias = trayectorias(sp)
    ars = pd.read_csv(OUT / "scores_arsenal.csv")
    ars = ars[[c for c in COLS_ARS if c in ars.columns]]
    if medias is not None:
        ars = ars.merge(medias.rename(columns={"air_density": "rho_9p", "n": "n_9p"}),
                        on=["pitcher_id", "season", "pt"], how="left")
    datos["arsenales.json"] = [{k: (r(v, 3) if isinstance(v, (int, float, np.floating, np.integer)) and not
                                    isinstance(v, bool) else v) for k, v in f.items()}
                               for f in ars.to_dict("records")]
    datos["trayectorias.json"] = tipos3d
    print(f"trayectorias: {sum(len(v) for v in tipos3d.values())} tipo-banda | arsenales con 9P: "
          f"{0 if medias is None else len(medias):,}")
    datos["mapas.json"] = mapas(sp)
    print("rejilla del simulador:")
    datos["simulador.json"] = rejilla_simulador(sp)
    destino = RAIZ / "web_datos.zip"
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for nombre, obj in datos.items():
            z.writestr(nombre, json.dumps(limpiar(obj), ensure_ascii=True, separators=(",", ":"), allow_nan=False))
    print(f"listo: {destino.name} ({destino.stat().st_size / 1e6:.1f} MB). Mandalo tal cual.")


if __name__ == "__main__":
    main()
