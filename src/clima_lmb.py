"""
CLIMA REAL DE LOS JUEGOS LMB -> densidad del aire real (no atmosfera estandar).

Utilidad aparte (la corrida normal no la usa): de aqui salen las constantes de
densidad y clima de config.py. Solo fuentes publicas y gratuitas, sin llave:
  1. Calendario publico de la LMB en MLB Stats API (sportId 23): juegos con
     estado final, estadio (venue) y hora programada (gameDate, en UTC).
     Se consulta por http (por https ha respondido 406).
  2. Open-Meteo Archive, modelo ERA5 (reanalisis de ECMWF / Copernicus):
     temperatura, humedad relativa y presion en superficie, hora por hora, en
     las coordenadas de cada estadio.
  3. Open-Meteo Elevation (modelo digital del terreno Copernicus GLO-90):
     elevacion de cada estadio.

Clima de cada juego = el de la mitad del juego, 90 min despues de la hora
programada, interpolado entre las dos horas del reanalisis que la rodean.

Densidad del aire humedo (kg/m^3):
     rho = Pd / (Rd*T) + Pv / (Rv*T)
     Pv = HR * es(T),  es = 6.1078 * 10^(7.5 t / (t + 237.3))  [hPa, Tetens]
     Pd = P - Pv,  Rd = 287.05,  Rv = 461.495,  T en Kelvin
  rho_rel = rho / 1.225 (aire estandar a nivel del mar)

Salida (en carpeta_salida):
  clima_por_juego.csv        un renglon por juego final con estadio conocido
  clima_resumen_estadio.csv  elevacion Copernicus y promedios de cada estadio
  constantes.json            ALTITUDE_M, ALTITUDE_RHO, RHO_CDMX, RHO_MAR,
                             CLIMA_TIPICO, CLIMA_FACTOR y la elevacion de cada
                             estadio (STADIUMS), para pasarlos a config.py
  crudo/                     respuestas de las APIs (json.gz): si ya estan, se
                             reusan y no se vuelve a consultar

Uso:  python clima_lmb.py carpeta_salida [temporadas]
      temporadas por omision: 2024 2025 2026 (tambien se acepta "2024,2025")
"""
import gzip
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

STATSAPI = "http://statsapi.mlb.com/api/v1"
ARCHIVO = "https://archive-api.open-meteo.com/v1/archive"
ELEVACION = "https://api.open-meteo.com/v1/elevation"
TEMPORADAS = (2024, 2025, 2026)

PAUSA_S = 1.0           # pausa antes de cada consulta
PAUSA_METEO_S = 2.0     # las consultas largas pesan mas en el limite de uso de Open-Meteo
ESPERA_429_S = 65       # si el servicio pide bajar el ritmo (HTTP 429)
MAX_CONSULTAS = 115     # tope de consultas por corrida (las de crudo/ no cuentan)
MITAD_MIN = 90          # minutos de la hora programada a la mitad del juego
DOBLE_JUEGO2_H = 3      # juego 2 de doble cartelera sin hora: hora del juego 1 + 3 h

# coordenadas aproximadas de cada estadio (la rejilla de ERA5 es de 0.25 grados, ~28 km)
COORDS = {
    "harp_helu": (19.404, -99.086), "hermanos_serdan": (19.075, -98.171),
    "francisco_villa": (24.033, -104.646), "romo_chavez": (21.882, -102.283),
    "conspiradores": (20.590, -100.390), "domingo_santana": (21.115, -101.653),
    "francisco_madero": (25.434, -100.976), "panamericano": (20.700, -103.463),
    "eduardo_vasconcelos": (17.075, -96.718), "monumental_chihuahua": (28.668, -106.072),
    "revolucion": (25.546, -103.443), "monclova": (26.905, -101.418),
    "monterrey": (25.716, -100.297), "la_junta": (27.484, -99.506),
    "unitrade": (27.555, -99.476), "chevron": (32.510, -116.995),
    "beto_avila_cancun": (21.152, -86.836), "beto_avila_veracruz": (19.180, -96.137),
    "nelson_barrera": (19.833, -90.535), "centenario_27": (17.985, -92.927),
    "kukulcan": (20.946, -89.627), "kanasin": (20.937, -89.553),
}
HOME_STADIUM = "harp_helu"

# categoria de altitud de ISAC (columna altitude_category) de cada estadio. El
# diccionario de ISAC no dice que parques caen en cada una: es el mismo reparto
# de config.py. No Altitude = el resto, parques de menos de LIMITE_SIN_ALTITUD_M
# (se verifica con la elevacion Copernicus).
CATEGORIAS = ("No Altitude", "Medium Altitude", "Extreme Altitude")
CATEGORIA = {
    **dict.fromkeys(("harp_helu", "hermanos_serdan", "francisco_villa", "romo_chavez",
                     "conspiradores", "domingo_santana"), "Extreme Altitude"),
    **dict.fromkeys(("revolucion", "monumental_chihuahua", "francisco_madero",
                     "eduardo_vasconcelos", "panamericano"), "Medium Altitude"),
}
LIMITE_SIN_ALTITUD_M = 700
LIMITE_MAR_M = 200      # RHO_MAR y CLIMA_TIPICO["mar"]: parques de menos de 200 m

# nombre del estadio en la API de la LMB -> clave del estadio (por palabra clave)
VENUE_KEYS = [("harp hel", "harp_helu"), ("serdan", "hermanos_serdan"), ("francisco villa", "francisco_villa"),
              ("romo", "romo_chavez"), ("conspiradores", "conspiradores"),
              ("domingo santana", "domingo_santana"), ("madero", "francisco_madero"),
              ("panamericano", "panamericano"), ("vasconcelos", "eduardo_vasconcelos"),
              ("chihuahua", "monumental_chihuahua"), ("revoluci", "revolucion"),
              ("kickapoo", "monclova"), ("monclova", "monclova"), ("junta", "la_junta"),
              ("uni-trade", "unitrade"), ("toros mobil", "chevron"), ("chevron", "chevron"),
              ("tijuana", "chevron"), ("nelson barrera", "nelson_barrera"),
              ("centenario", "centenario_27"), ("kukulc", "kukulcan"), ("kanas", "kanasin"),
              ("cervera", "kanasin"), ("yu'va", "eduardo_vasconcelos"),
              ("kukulk", "kukulcan"), ("nuevo laredo", "la_junta")]

_consultas = 0


def categoria(estadio):
    return CATEGORIA.get(estadio, "No Altitude")


def _ascii(x):
    return str(x).encode("ascii", "replace").decode()


def _utc(x):
    return pd.to_datetime(pd.Series(x), utc=True, errors="coerce").dt.as_unit("s")


def _bajar(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.load(r)


def get(url, nombre, crudo, pausa=PAUSA_S, intentos=3):
    """JSON de url. Si crudo/<nombre>.json.gz ya tiene esa misma url, se lee de
    ahi sin consultar. Solo se reintenta ante 429, 5xx o fallas de red."""
    global _consultas
    arch = Path(crudo) / f"{nombre}.json.gz"
    if arch.exists():
        with gzip.open(arch, "rt", encoding="utf-8") as f:
            guardado = json.load(f)
        if guardado.get("url") == url:
            return guardado["datos"]
    for i in range(intentos):
        if _consultas >= MAX_CONSULTAS:
            raise SystemExit(f"Tope de {MAX_CONSULTAS} consultas: no se consulta {url}")
        time.sleep(pausa)
        _consultas += 1
        try:
            datos = _bajar(url)
            break
        except urllib.error.HTTPError as e:
            if i < intentos - 1 and (e.code == 429 or e.code >= 500):
                time.sleep(ESPERA_429_S if e.code == 429 else 10 * (i + 1))
                continue
            raise SystemExit(f"HTTP {e.code} en {url}: {e.read()[:300]!r}")
        except (urllib.error.URLError, OSError, ValueError):
            if i == intentos - 1:
                raise
            time.sleep(10 * (i + 1))
    tmp = arch.with_name(arch.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump({"url": url, "datos": datos}, f)
    tmp.replace(arch)
    return datos


def _sin_acentos(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def venue_a_estadio(nombre, ciudad):
    n, c = _sin_acentos(nombre), _sin_acentos(ciudad)
    if "beto" in n and "vila" in n:
        return "beto_avila_cancun" if "canc" in c or "sherwin" in n else "beto_avila_veracruz"
    if "monterrey" in c or "monterrey" in n:
        return "monterrey"
    if "oaxaca" in c:
        return "eduardo_vasconcelos"          # 2026: "Estadio Yu'Va"
    if "kanasin" in c:
        return "kanasin"
    for k, cod in VENUE_KEYS:
        if k in n:
            return cod
    return None


def calendario_lmb(temporadas, crudo):
    """Todos los juegos del calendario publico de la LMB (cualquier estado)."""
    filas = []
    for t in temporadas:
        u = (f"{STATSAPI}/schedule?sportId=23&startDate={t}-03-01&endDate={t}-11-30"
             f"&gameType=R,F,D,L,W&hydrate=weather,venue(location)")
        for d in get(u, f"calendario_{t}", crudo).get("dates", []):
            for g in d.get("games", []):
                v = g.get("venue") or {}
                st = g.get("status") or {}
                filas.append({"gamePk": g.get("gamePk"), "temporada": t, "tipo": g.get("gameType"),
                              "fecha": g.get("officialDate") or d.get("date"),
                              "gameDate": g.get("gameDate"), "venue": v.get("name"),
                              "estadio": venue_a_estadio(v.get("name"), (v.get("location") or {}).get("city")),
                              "estado": st.get("detailedState"), "hora_tbd": bool(st.get("startTimeTBD")),
                              "juego_num": g.get("gameNumber"),
                              "temp_api_f": (g.get("weather") or {}).get("temp")})
    c = pd.DataFrame(filas)
    if c.empty:
        raise SystemExit(f"El calendario no trajo juegos para {list(temporadas)}")
    c["gameDate"] = _utc(c["gameDate"])
    c["juego_num"] = pd.to_numeric(c["juego_num"], errors="coerce")
    return c


VARS = ("temperature_2m", "relative_humidity_2m", "surface_pressure")


def meteo(estadio, inicio, fin, crudo):
    """Clima horario (UTC) del reanalisis ERA5 en las coordenadas del estadio."""
    lat, lon = COORDS[estadio]
    u = (f"{ARCHIVO}?latitude={lat}&longitude={lon}&start_date={inicio}&end_date={fin}"
         f"&hourly={','.join(VARS)}&models=era5&timezone=UTC")
    r = get(u, f"meteo_{estadio}_{inicio[:4]}", crudo, pausa=PAUSA_METEO_S)
    h = r["hourly"]

    def col(v):                              # con un solo modelo la API no le pone sufijo
        k = v if v in h else next(k for k in h if k.startswith(v))
        return pd.to_numeric(pd.Series(h[k]), errors="coerce").to_numpy(float)

    m = pd.DataFrame({"hora": _utc(h["time"]), "temp_c": col(VARS[0]), "hr_pct": col(VARS[1]),
                      "presion_hpa": col(VARS[2])})
    vacias = [v for v in ("temp_c", "hr_pct", "presion_hpa") if m[v].isna().all()]
    if vacias:                               # mejor parar antes de gastar mas consultas
        raise SystemExit(f"Open-Meteo no trajo {vacias} para {estadio} ({inicio} a {fin})")
    m["estadio"] = estadio
    m["elev_meteo_m"] = r.get("elevation")   # elevacion a la que Open-Meteo ajusta el dato
    return m


def elevaciones(crudo):
    """Elevacion de cada estadio en el modelo digital del terreno (Copernicus
    GLO-90, servicio de elevacion de Open-Meteo)."""
    cods = list(COORDS)
    lat = ",".join(str(COORDS[c][0]) for c in cods)
    lon = ",".join(str(COORDS[c][1]) for c in cods)
    e = get(f"{ELEVACION}?latitude={lat}&longitude={lon}", "elevacion", crudo)["elevation"]
    return pd.DataFrame({"estadio": cods, "categoria": [categoria(c) for c in cods],
                         "lat": [COORDS[c][0] for c in cods], "lon": [COORDS[c][1] for c in cods],
                         "elevacion_m": np.asarray(e, float)})


def densidad(temp_c, hr_pct, presion_hpa):
    t = np.asarray(temp_c, float)
    es = 6.1078 * 10 ** (7.5 * t / (t + 237.3))
    pv = np.asarray(hr_pct, float) / 100 * es
    pd_ = np.asarray(presion_hpa, float) - pv
    T = t + 273.15
    return (pd_ * 100) / (287.05 * T) + (pv * 100) / (461.495 * T)


def estandar_rel(alt_m):
    """Densidad relativa de la atmosfera estandar ICAO (igual que features.air_density_ratio)."""
    return (1.0 - 2.25577e-5 * np.asarray(alt_m, float)) ** 4.2559


def presion_estandar(alt_m):
    """Presion de la atmosfera estandar ICAO, hPa (igual que features.densidad_aire)."""
    return 1013.25 * (1.0 - 2.25577e-5 * np.asarray(alt_m, float)) ** 5.25588


def _media(s, nd):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    return round(float(s.mean()), nd) if len(s) else None


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def _clima(g):
    """Promedios de clima de un grupo de juegos (solo los que tienen clima)."""
    c = g.dropna(subset=["rho_rel"])
    return {"juegos": int(len(g)), "juegos_con_clima": int(len(c)),
            "rho_rel": _media(c["rho_rel"], 4), "temp_c": _media(c["temp_c"], 1),
            "hr_pct": _media(c["hr_pct"], 1), "presion_hpa": _media(c["presion_hpa"], 1)}


def main(salida, temporadas=TEMPORADAS):
    salida = Path(salida)
    crudo = salida / "crudo"
    crudo.mkdir(parents=True, exist_ok=True)
    temporadas = sorted(int(t) for t in temporadas)

    # 1. juegos finales del calendario publico, con estadio
    cal = calendario_lmb(temporadas, crudo)
    fin = cal[cal["estado"].fillna("").astype(str).str.startswith(("Final", "Game Over", "Completed"))]
    duplicados = int(fin["gamePk"].duplicated().sum())
    fin = fin.drop_duplicates("gamePk", keep="first")
    sin_est = fin[fin["estadio"].isna()]
    venues_sin = {str(k): int(v) for k, v in sin_est["venue"].fillna("(sin nombre)").value_counts().items()}
    j = fin.dropna(subset=["estadio"]).reset_index(drop=True)
    if j.empty:
        raise SystemExit("Ningun juego final con estadio conocido")
    print(f"calendario LMB {temporadas}: {len(cal):,} juegos, {len(fin):,} finales "
          f"({duplicados} gamePk repetidos), {len(sin_est):,} sin estadio")
    if venues_sin:
        print("  estadios de la API sin clave (se ignoran):", _ascii(venues_sin))

    # 2. hora de la mitad del juego. Sin hora definida (startTimeTBD) solo se
    #    estima el juego 2 de una doble cartelera: empieza despues del juego 1
    #    (en el calendario suele traer la hora del juego 1 + 5 min). Los demas
    #    juegos sin hora se quedan sin clima.
    uno = (j.loc[j["juego_num"].eq(1) & ~j["hora_tbd"], ["fecha", "estadio", "gameDate"]]
           .drop_duplicates(["fecha", "estadio"]).rename(columns={"gameDate": "inicio_j1"}))
    j = j.merge(uno, on=["fecha", "estadio"], how="left")
    estimar = j["hora_tbd"] & j["juego_num"].eq(2)
    j["inicio"] = j["gameDate"].mask(j["hora_tbd"])
    j["inicio"] = j["inicio"].fillna((j["inicio_j1"] + pd.Timedelta(hours=DOBLE_JUEGO2_H)).where(estimar))
    j["mitad"] = (j["inicio"] + pd.Timedelta(minutes=MITAD_MIN)).dt.as_unit("s")

    # 3. clima horario: una consulta por estadio y temporada, solo en las fechas
    #    de sus juegos (el archivo ERA5 se publica con unos dias de retraso:
    #    se pide hasta hace 7 dias)
    corte = (pd.Timestamp.now("UTC") - pd.Timedelta(days=7)).normalize()
    met = []
    for (est, t), g in j.dropna(subset=["mitad"]).groupby(["estadio", "temporada"]):
        ini = g["mitad"].min().normalize() - pd.Timedelta(days=1)
        fin_ = min(g["mitad"].max().normalize() + pd.Timedelta(days=1), corte)
        if fin_ < ini:
            continue
        met.append(meteo(est, ini.strftime("%Y-%m-%d"), fin_.strftime("%Y-%m-%d"), crudo))
    if not met:
        raise SystemExit("No hubo clima para ningun juego")
    met = pd.concat(met, ignore_index=True).drop_duplicates(["estadio", "hora"])

    # 4. interpolacion lineal entre la hora anterior y la siguiente a la mitad del juego
    h0 = j["mitad"].dt.floor("h")
    w = ((j["mitad"] - h0) / pd.Timedelta(hours=1)).to_numpy(float)

    def en(hora):
        return j[["estadio"]].assign(hora=hora.dt.as_unit("s")).merge(met, on=["estadio", "hora"], how="left")

    a, b = en(h0), en(h0 + pd.Timedelta(hours=1))
    for v in ("temp_c", "hr_pct", "presion_hpa"):
        va, vb = a[v].to_numpy(float), b[v].to_numpy(float)
        j[v] = np.where(w == 0, va, (1 - w) * va + w * vb)
    j["rho_rel"] = densidad(j["temp_c"], j["hr_pct"], j["presion_hpa"]) / 1.225

    # 5. elevacion y atmosfera estandar en la elevacion de cada juego
    est = elevaciones(crudo)
    altos = est[(est["categoria"] == "No Altitude") & (est["elevacion_m"] >= LIMITE_SIN_ALTITUD_M)]
    if len(altos):
        print(f"AVISO: parques 'No Altitude' de {LIMITE_SIN_ALTITUD_M} m o mas:", altos["estadio"].tolist())
    j = j.merge(est[["estadio", "categoria", "elevacion_m"]], on="estadio", how="left")
    j["factor_clima"] = j["rho_rel"] / estandar_rel(j["elevacion_m"])
    j["presion_vs_std"] = j["presion_hpa"] / presion_estandar(j["elevacion_m"])

    j["hora_mitad_utc"] = j["mitad"].dt.strftime("%Y-%m-%dT%H:%MZ")
    cols = ["gamePk", "estadio", "fecha", "temp_c", "hr_pct", "presion_hpa", "rho_rel",
            "temporada", "tipo", "categoria", "elevacion_m", "hora_mitad_utc"]
    (j[cols].sort_values(["fecha", "gamePk"])
     .round({"temp_c": 2, "hr_pct": 1, "presion_hpa": 2, "rho_rel": 5})
     .to_csv(salida / "clima_por_juego.csv", index=False))

    c = j.dropna(subset=["rho_rel"])
    agg = c.groupby("estadio").agg(juegos_con_clima=("gamePk", "size"), temp_c=("temp_c", "mean"),
                                   hr_pct=("hr_pct", "mean"), presion_hpa=("presion_hpa", "mean"),
                                   presion_vs_std=("presion_vs_std", "mean"), rho_rel=("rho_rel", "mean"),
                                   rho_rel_min=("rho_rel", "min"), rho_rel_max=("rho_rel", "max"),
                                   factor_clima=("factor_clima", "mean"))
    r = (est.merge(j.groupby("estadio").size().rename("juegos"), left_on="estadio", right_index=True, how="left")
         .merge(agg, left_on="estadio", right_index=True, how="left")
         .merge(met.groupby("estadio")["elev_meteo_m"].first(), left_on="estadio", right_index=True, how="left"))
    r["juegos"] = r["juegos"].fillna(0).astype(int)
    r["juegos_con_clima"] = r["juegos_con_clima"].fillna(0).astype(int)
    r["rho_std_atm"] = estandar_rel(r["elevacion_m"])
    r = r.sort_values("elevacion_m", ascending=False)
    r.round(4).to_csv(salida / "clima_resumen_estadio.csv", index=False)

    # 6. constantes para config.py
    por_t = j.groupby("temporada").size()
    cats = {}
    for cat in CATEGORIAS:
        g = j[j["categoria"] == cat]
        n_t = g.groupby("temporada").size().reindex(por_t.index, fill_value=0)
        cats[cat] = {"estadios": [k for k in COORDS if categoria(k) == cat],
                     "pct_juegos": round(100 * len(g) / len(j), 1),
                     "juegos_temporada_regular": int(g["tipo"].eq("R").sum()),
                     "pct_temporada_regular": round(100 * g["tipo"].eq("R").sum()
                                                    / max(int(j["tipo"].eq("R").sum()), 1), 1),
                     "juegos_por_temporada": {str(t): int(n) for t, n in n_t.items()},
                     "pct_por_temporada": {str(t): round(100 * n / por_t[t], 1) for t, n in n_t.items()},
                     "elevacion_m_por_juego": _media(g["elevacion_m"], 0),
                     **_clima(g)}
    casa = j[j["estadio"] == HOME_STADIUM]
    mar_est = est.loc[est["elevacion_m"] < LIMITE_MAR_M, "estadio"].tolist()
    mar = j[j["estadio"].isin(mar_est)]
    k_casa, k_mar = _clima(casa), _clima(mar)
    x = c["elevacion_m"].to_numpy(float) / 2240.0
    y = c["presion_vs_std"].to_numpy(float) - 1.0
    t_api = (pd.to_numeric(j["temp_api_f"], errors="coerce") - 32) * 5 / 9
    ok = t_api.notna() & j["temp_c"].notna()

    k = {
        "ALTITUDE_M": {cat: int(cats[cat]["elevacion_m_por_juego"]) if cats[cat]["juegos"] else None
                       for cat in CATEGORIAS},
        "ALTITUDE_RHO": {cat: cats[cat]["rho_rel"] for cat in CATEGORIAS},
        "RHO_CDMX": k_casa["rho_rel"],
        "RHO_MAR": k_mar["rho_rel"],
        "CLIMA_TIPICO": {"cdmx": [k_casa["temp_c"], k_casa["hr_pct"]], "mar": [k_mar["temp_c"], k_mar["hr_pct"]]},
        "CLIMA_FACTOR": _media(c["factor_clima"], 4),
        "PRESION_CORRECCION_2240": round(float((x * y).sum() / (x * x).sum()), 4) if len(c) else None,
        "STADIUMS_ELEVACION_M": {e: int(round(m)) for e, m in zip(est["estadio"], est["elevacion_m"])},
        "categorias": cats,
        "cdmx": {"estadio": HOME_STADIUM, **k_casa},
        "mar": {"estadios": mar_est, **k_mar},
        "control": {"temporadas": temporadas, "juegos_calendario": int(len(cal)),
                    "juegos_finales": int(len(fin)), "gamePk_repetidos": duplicados,
                    "sin_estadio": int(len(sin_est)), "venues_sin_estadio": venues_sin,
                    "con_estadio": int(len(j)), "hora_tbd": int(j["hora_tbd"].sum()),
                    "hora_estimada_doble_cartelera": int((estimar & j["inicio"].notna()).sum()),
                    "sin_hora": int(j["inicio"].isna().sum()), "sin_clima": int(j["rho_rel"].isna().sum()),
                    "temp_reportada_vs_reanalisis": {
                        "juegos": int(ok.sum()),
                        "correlacion": _corr(t_api[ok], j.loc[ok, "temp_c"]),
                        "diferencia_media_c": _media(t_api[ok] - j.loc[ok, "temp_c"], 2)},
                    "consultas_en_esta_corrida": _consultas},
        "metodo": {
            "juegos": "calendario publico LMB, MLB Stats API sportId 23, gameType R,F,D,L,W, estado final",
            "hora": f"hora programada (gameDate, UTC) + {MITAD_MIN} min, interpolada entre horas del reanalisis",
            "clima": "Open-Meteo Archive, models=era5: temperature_2m, relative_humidity_2m, surface_pressure",
            "densidad": "aire humedo (Tetens), rho_rel = rho / 1.225 kg/m3",
            "elevacion": "Open-Meteo Elevation (Copernicus GLO-90) en las coordenadas de cada estadio",
            "ALTITUDE_M": "elevacion promedio por juego de los parques de cada categoria",
            "ALTITUDE_RHO": "rho_rel promedio de los juegos de cada categoria",
            "RHO_CDMX": f"rho_rel promedio de los juegos en {HOME_STADIUM}",
            "RHO_MAR": f"rho_rel promedio de los juegos en parques de menos de {LIMITE_MAR_M} m",
            "CLIMA_FACTOR": "promedio por juego de rho_rel / atmosfera estandar ICAO en la elevacion del estadio",
            "PRESION_CORRECCION_2240": "k de minimos cuadrados (por el origen) en presion / presion estandar "
                                       "= 1 + k * elevacion / 2240 (features.densidad_aire)",
        },
    }
    with open(salida / "constantes.json", "w", encoding="utf-8") as f:
        json.dump(k, f, ensure_ascii=False, indent=2)

    pd.set_option("display.width", 200)
    print(f"con estadio: {len(j):,} juegos; sin hora: {k['control']['sin_hora']}; "
          f"sin clima: {k['control']['sin_clima']}; hora estimada (doble cartelera): "
          f"{k['control']['hora_estimada_doble_cartelera']}")
    print(r[["estadio", "categoria", "elevacion_m", "elev_meteo_m", "juegos", "juegos_con_clima", "temp_c",
             "hr_pct", "presion_hpa", "rho_rel", "factor_clima"]].round(3).to_string(index=False))
    for cat in CATEGORIAS:
        print(f"  {cat:17s} {cats[cat]['juegos']:5d} juegos ({cats[cat]['pct_juegos']:4.1f} %)  "
              f"{k['ALTITUDE_M'][cat]} m  rho_rel {cats[cat]['rho_rel']}")
    print(f"RHO_CDMX {k['RHO_CDMX']}  RHO_MAR {k['RHO_MAR']}  CLIMA_TIPICO {k['CLIMA_TIPICO']}  "
          f"CLIMA_FACTOR {k['CLIMA_FACTOR']}  PRESION_CORRECCION_2240 {k['PRESION_CORRECCION_2240']}")
    print(f"consultas a las APIs en esta corrida: {_consultas}")


def _temporadas(args):
    t = [int(x) for a in args for x in a.replace(",", " ").split()]
    return t or list(TEMPORADAS)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], _temporadas(sys.argv[2:]))
