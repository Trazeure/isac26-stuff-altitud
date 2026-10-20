"""
Corre todo el pipeline en orden (Windows, Mac o Linux).

    python run_all.py --rapido          # prueba de que todo corre (30% de los pitchers, ~10-15 min)
    python run_all.py                   # corrida completa (~1-1.5 h)
    python run_all.py --estricto        # corrida de ENTREGA: los avisos graves detienen el pipeline
    python run_all.py --desde score     # retomar desde un paso (usa lo ya calculado antes)
    python run_all.py --solo insights report   # solo esos pasos (separados por espacio o por coma)

Pasos: features, physics, valores, validar, final, score, insights, report.
La validacion ya no es opcional: los scores que se entregan salen de los
modelos de la validacion cruzada (cada pitcher lo califica un modelo que nunca
lo vio), asi que 'validar' siempre corre antes de 'score'.

Un argumento que no se reconoce detiene la corrida ANTES de empezar: un
"--rapdio" mal escrito no debe lanzar la corrida completa.

out/run_info.json guarda, por paso, con que codigo y en que modo se calculo lo
que hay en out/ y models/. Al retomar (--desde / --solo) se avisa si lo que no
se vuelve a calcular viene de otro codigo, de otro modo (rapido / completo), de
un paso que no termino o de antes de la ultima corrida de un paso anterior.
"""
import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "out"
PASOS = [
    ("features", ["features.py"]),
    ("physics",  ["physics.py"]),
    ("valores",  ["model.py", "valores"]),
    ("validar",  ["model.py", "validar"]),
    ("final",    ["model.py", "final"]),
    ("score",    ["score.py"]),
    ("insights", ["insights.py"]),
    ("report",   ["report.py"]),
]
NOMBRES = [n for n, _ in PASOS]
INFO = OUT / "run_info.json"
PLACEHOLDERS = {"Nombre del equipo", "Integrante 1", "Integrante 2", "Universidad"}
# El codigo que calcula las cifras. La huella NO incluye report.py, el dashboard, el simulador ni este archivo:
# retocar como se presenta un numero no vuelve viejo lo ya calculado.
CODIGO_DE_DATOS = ("config.py", "features.py", "physics.py", "model.py", "score.py", "insights.py")
PORTADA = ("EQUIPO", "INTEGRANTES", "INSTITUCION")
EXTENSIONES = (".csv", ".csv.gz", ".parquet")                # las mismas que lee features.load_raw
REQUERIDAS = {"pandas": "pandas", "numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn",
              "pyarrow": "pyarrow", "joblib": "joblib", "matplotlib": "matplotlib", "reportlab": "reportlab"}
SKLEARN_MINIMO = (1, 4)                                      # el mismo que exige model.py
# Pasos que, con --estricto, pueden detenerse DESPUES de escribir todo (y lo explican en su propio mensaje):
# el archivo que escriben al final dice si eso fue lo que paso.
ESCRIBE_AL_FINAL = {"score": ("scores_pitch.parquet",),
                    "report": ("documento_tecnico.pdf", "documento_tecnico_nuevo.pdf")}


def _ascii(x):
    """Texto seguro para la consola de Windows."""
    return str(x).encode("ascii", "replace").decode("ascii")


def _sin_portada(texto):
    """config.py sin las asignaciones de la portada, con todas sus lineas (la lista de integrantes puede ocupar
    varias): llenar la portada no debe marcar como viejo lo ya calculado."""
    try:
        arbol = ast.parse(texto)
    except SyntaxError:
        return texto
    fuera = set()
    for nodo in arbol.body:
        destinos = getattr(nodo, "targets", None) or [getattr(nodo, "target", None)]
        if isinstance(nodo, (ast.Assign, ast.AnnAssign)) and any(getattr(d, "id", None) in PORTADA for d in destinos):
            fuera.update(range(nodo.lineno, nodo.end_lineno + 1))
    return "\n".join(linea for k, linea in enumerate(texto.split("\n"), 1) if k not in fuera)


def hash_codigo():
    """Huella del codigo que calcula las cifras (CODIGO_DE_DATOS): si cambia uno de esos .py, cambia. El
    documento la imprime en el pie para saber con que version se genero cada cifra. Es igual en Windows y
    Linux (saltos de linea) y no cambia si un editor guarda el archivo con BOM."""
    h = hashlib.sha1()
    for nombre in CODIGO_DE_DATOS:
        p = SRC / nombre
        if not p.exists():
            continue
        texto = p.read_bytes().decode("utf-8-sig", "replace").replace("\r\n", "\n")
        h.update(nombre.encode())
        h.update((_sin_portada(texto) if nombre == "config.py" else texto).encode("utf-8"))
    return h.hexdigest()


def versiones():
    v = {"python": platform.python_version()}
    for lib in (*REQUERIDAS, "lightgbm"):
        try:
            v[lib] = __import__(lib).__version__ if lib != "reportlab" else __import__(lib).Version
        except Exception:                       # noqa: BLE001
            pass
    return v


def leer_info():
    try:
        info = json.loads(INFO.read_text(encoding="utf-8"))
        return info if isinstance(info, dict) else {}
    except Exception:                           # noqa: BLE001
        return {}


def es_rapido(nombre, registro):
    """Lo que ese paso dejo en disco viene, o puede venir, de una corrida rapida? Un paso de DATOS que no
    termino pudo dejar los archivos de la corrida anterior: cuenta tambien el modo de aquella. El paso
    report no hereda nada: el documento que escribe es de la corrida que lo escribe."""
    return bool(registro.get("rapido")) or (nombre != "report" and registro.get("terminado") is False
                                            and bool(registro.get("antes_rapido")))


def guardar_info(info):
    """El modo de arriba del archivo describe lo que hay en disco (algun paso de corrida rapida), no la ultima
    llamada: es lo que leen el documento (mientras se escribe), el dashboard y la lista de control."""
    info["rapido"] = any(es_rapido(n, p) for n, p in info.get("pasos", {}).items())
    OUT.mkdir(exist_ok=True)
    INFO.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")


def uso(msg):
    print(_ascii(msg))
    print("Pasos validos:", ", ".join(NOMBRES))
    print("Ejemplos:  python run_all.py --rapido   |   python run_all.py --estricto   |   "
          "python run_all.py --desde score   |   python run_all.py --solo insights report")
    sys.exit(2)


def leer_argumentos(args):
    """(rapido, estricto, pasos). Acepta --rapido, --estricto, --desde PASO y --solo PASOS (separados por
    espacio o por coma: PowerShell parte "a,b" en dos argumentos), tambien con "=". Todo lo demas detiene la
    corrida antes de empezar."""
    rapido = estricto = False
    solo = desde = None
    i = 0
    while i < len(args):
        clave, igual, valor = args[i].partition("=")
        if args[i] in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        if args[i] == "--sin-validar":
            uso("--sin-validar ya no existe: los scores salen de los modelos de la validacion cruzada.\n"
                "Para una prueba corta usa --rapido.")
        if args[i] == "--rapido":
            rapido = True
        elif args[i] == "--estricto":
            estricto = True
        elif clave in ("--solo", "--desde"):
            dados = [valor] if igual else []
            while i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
                dados.append(args[i])
            dados = [p for d in dados for p in d.replace(",", " ").split()]
            malos = [p for p in dados if p not in NOMBRES]
            if not dados or malos:
                uso(f"{clave} necesita pasos validos (no reconozco: {malos or 'ninguno dado'})")
            if clave == "--desde" and len(dados) != 1:
                uso(f"--desde necesita UN paso (recibi: {dados}). Para elegir varios usa --solo")
            solo, desde = (dados, desde) if clave == "--solo" else (solo, dados[0])
        else:
            uso(f"No reconozco el argumento {args[i]!a}: no se corrio nada.")
        i += 1
    if solo and desde:
        uso("--solo y --desde no se usan juntos: elige uno")
    if solo:
        return rapido, estricto, [n for n in NOMBRES if n in solo]
    return rapido, estricto, NOMBRES[NOMBRES.index(desde):] if desde else list(NOMBRES)


def revisar_antes(pasos, cfg, rapido, estricto, vers):
    """Lo que conviene saber ANTES de una corrida de mas de una hora."""
    faltan = [pip for lib, pip in REQUERIDAS.items() if lib not in vers]
    if faltan:
        print(f"No se pudo importar: {', '.join(faltan)}. Instala lo que pide el proyecto y vuelve a correr:\n"
              "    python -m pip install -r requirements.txt")
        sys.exit(2)
    if tuple(int(x) for x in re.findall(r"\d+", vers["sklearn"])[:2]) < SKLEARN_MINIMO:
        print(f"scikit-learn {vers['sklearn']} es anterior a {'.'.join(map(str, SKLEARN_MINIMO))}, la minima con que "
              "corre model.py. Actualiza y vuelve a correr:\n    python -m pip install -r requirements.txt")
        sys.exit(2)
    avisos = []
    datos = Path(cfg.DATA_FILE)
    archivos = (sorted(p for p in datos.iterdir() if p.name.lower().endswith(EXTENSIONES))
                if datos.is_dir() else ([datos] if datos.exists() else []))
    if not archivos and "features" in pasos:
        print(f"No hay archivos de datos en {_ascii(datos)}. Copia ahi el dataset (csv, csv.gz o parquet).")
        sys.exit(2)
    if "features" in pasos:
        print(f"Archivos de datos ({len(archivos)}): " + ", ".join(_ascii(p.name) for p in archivos))
        if len({re.sub(r"\d+", "#", p.name.lower()) for p in archivos}) > 1:
            print("OJO: los nombres parecen de mas de un origen y se leen TODOS juntos. Deja en data/ solo "
                  "las partes de stuff_model_df.")
    equipo = {cfg.EQUIPO, cfg.INSTITUCION, *cfg.INTEGRANTES}
    if equipo & PLACEHOLDERS:
        avisos.append("La portada del documento saldra con datos de ejemplo: llena EQUIPO, INTEGRANTES e "
                      "INSTITUCION en src/config.py")
    if rapido and estricto:
        avisos.append("--rapido y --estricto juntos: la corrida de entrega debe ser la completa")
    return [p.name for p in archivos], avisos


def revisar_lo_previo(pasos, previo, codigo, rapido):
    """Avisos sobre lo que esta corrida NO vuelve a calcular y va a usar (los pasos anteriores al ultimo que
    se pide): debe haber terminado, con este codigo, en este modo y despues del paso que le precede."""
    registro = previo.get("pasos") or {}
    avisos, reciente = [], 0.0
    for n in NOMBRES[:NOMBRES.index(pasos[-1])]:
        if n in pasos:                          # se vuelve a calcular ahora: lo que sigue debe ser posterior
            reciente = float("inf")
            continue
        r = registro.get(n)
        if not isinstance(r, dict):
            avisos.append(f"no hay registro de que '{n}' se haya corrido con run_all.py (out/run_info.json)")
            continue
        if r.get("terminado") is False:
            avisos.append(f"'{n}' no termino la ultima vez que se corrio: sus archivos pueden estar a medias")
        if r.get("hash_codigo") != codigo:
            avisos.append(f"'{n}' se calculo con otra version del codigo ({str(r.get('hash_codigo'))[:8]}; "
                          f"ahora {codigo[:8]})")
        if es_rapido(n, r) != rapido:
            avisos.append(f"'{n}' viene de una corrida {'RAPIDA' if es_rapido(n, r) else 'COMPLETA'} y ahora "
                          f"pides {'RAPIDA' if rapido else 'COMPLETA'}")
        hecho = r.get("epoca")
        if isinstance(hecho, (int, float)):     # los registros de versiones anteriores no traen la hora exacta
            if hecho < reciente:
                avisos.append(f"'{n}' se calculo ANTES de la ultima corrida de un paso anterior: no la refleja")
            reciente = max(reciente, hecho)
    if avisos:
        primero = next(n for n in NOMBRES if any(f"'{n}'" in a for a in avisos))
        avisos.append(f"lo seguro es volver a calcular desde ahi:  python run_all.py {'--rapido ' if rapido else ''}"
                      f"--desde {primero}")
    return avisos


def paginas_pdf(path):
    try:
        return path.read_bytes().count(b"/Type /Page") - path.read_bytes().count(b"/Type /Pages")
    except Exception:                           # noqa: BLE001
        return None


def _marcas(nombres):
    """{archivo de out/: hora de modificacion} de los que existen y no estan vacios. Comparar las marcas de
    antes y de despues de un paso dice si lo reescribio, sin depender del reloj."""
    return {f: (OUT / f).stat().st_mtime_ns for f in nombres if (OUT / f).exists() and (OUT / f).stat().st_size > 0}


def documento(antes=None):
    """(ruta, es de esta corrida) del PDF mas reciente de out/. report.py escribe documento_tecnico_nuevo.pdf
    cuando documento_tecnico.pdf esta abierto en un visor (Windows lo bloquea): por eso se busca el mas nuevo.
    `antes` = las marcas de los PDF justo antes del paso report, o None si no se corrio ahora."""
    ahora = _marcas(ESCRIBE_AL_FINAL["report"])
    if not ahora:
        return None, False
    nombre = max(ahora, key=ahora.get)
    return OUT / nombre, antes is not None and ahora[nombre] != antes.get(nombre)


def lista_de_control(pdf=None, de_ahora=False):
    """Resumen final leido de out/: lo que hay que revisar antes de entregar. Cada renglon va por separado:
    un archivo que falta o que no se puede leer se dice y no tumba el final de una corrida de mas de una hora."""
    try:
        import pandas as pd
    except Exception:                           # noqa: BLE001
        return
    print("\n" + "=" * 68 + "\nLISTA DE CONTROL (revisala antes de entregar)\n" + "=" * 68)

    def csv(nombre):
        p = OUT / nombre
        return pd.read_csv(p) if p.exists() else None

    def es(serie, *valores):
        return serie.astype(str).str.strip().str.lower().isin(valores)

    def si(serie):
        return es(serie, "true", "1", "1.0")

    def contrafactual():
        cf = csv("val_contrafactual.csv")
        if cf is None or not {"bloque", "pasa"} <= set(cf.columns):
            return
        for bloque in ("neutralidad", "transporte", "resultados"):
            b = cf[cf["bloque"] == bloque]
            # sin veredicto: las filas 'sin datos' y, aparte, las de resultados que no podian fallar (score.py
            # las deja con pasa vacio y con_potencia False)
            sin_potencia = (int((b["pasa"].isna() & es(b["con_potencia"], "false", "0", "0.0")).sum())
                            if "con_potencia" in b else 0)
            cola = f" | sin potencia para fallar (sin veredicto): {sin_potencia}" if sin_potencia else ""
            b = b[b["pasa"].notna()]
            if not len(b):
                print(f"  contrafactual / {bloque:12s}: sin datos suficientes para la prueba{cola}")
                continue
            ok = si(b["pasa"])
            fallan = [" ".join(str(r.get(c, "")) for c in ("modelo", "pt", "variable") if c in b)
                      for _, r in b[~ok].iterrows()]
            dentro = (f" ({int((ok & si(b['ic_dentro_de_tolerancia'])).sum())} con todo el intervalo dentro de la "
                      "tolerancia)" if "ic_dentro_de_tolerancia" in b and bloque != "resultados" else "")
            print(f"  contrafactual / {bloque:12s}: pasan {int(ok.sum())} de {len(b)}{dentro}{cola}"
                  + (f"  <-- NO pasan: {_ascii('; '.join(fallan[:6]))}" if fallan else ""))

    def elasticidades():
        el = csv("altitude_elasticities.csv")
        if el is None or not {"fuente", "var", "pt"} <= set(el.columns):
            return
        mov = el[el["var"].isin(["ivb", "hb_n"]) & ~el["pt"].astype(str).str.startswith(("TODOS", "FAM"))]
        origen = mov["fuente"].astype(str).str.replace("_encogida", "", regex=False).value_counts()
        print("  elasticidades del movimiento (origen del valor usado): "
              + ", ".join(f"{k} {v}" for k, v in origen.items()))
        if "ic_fiable" in mov and not si(mov["ic_fiable"]).any():
            print("    (pocos niveles de densidad: los intervalos de la fisica son orientativos)")

    def calibracion():
        cal = csv("val_calibracion_resumen.csv")
        if cal is None or not {"ece", "objetivo", "modelo"} <= set(cal.columns):
            return
        c = cal[cal["objetivo"] != "xrv"]           # el xRV esta en carreras, no en probabilidad
        if "grupo" in c:
            c = c[c["grupo"] == "todos"]
        c = c.dropna(subset=["ece"])
        if len(c):
            r = c.sort_values("ece", ascending=False).iloc[0]
            ruido = r.get("ece_ruido")              # el ECE que daria el azar con ese tamano de muestra
            print(f"  calibracion: peor ECE = {r['ece']:.4f}"
                  + (f" (por azar se espera {ruido:.4f})" if ruido == ruido and ruido is not None else "")
                  + f" en {r['objetivo']} / {r['modelo']}"
                  + (f" | pendiente {r['pendiente']:.2f}" if "pendiente" in c and r["pendiente"] == r["pendiente"]
                     else ""))

    def validacion():
        res = csv("val_resumen.csv")
        if res is None or not {"modelo", "logloss"} <= set(res.columns):
            return

        def mejora(r, contra):
            v = r.get(f"mejora_vs_{contra}_pct")
            if v is None or v != v:
                return ""
            lo, hi = r.get(f"mejora_vs_{contra}_ci_lo"), r.get(f"mejora_vs_{contra}_ci_hi")
            ic = f" [{lo:+.1f}, {hi:+.1f}]" if None not in (lo, hi) and lo == lo and hi == hi else ""
            return f" | vs nulo {'por cuenta' if contra == 'cuenta' else 'global'} {v:+.1f}%{ic}"
        for _, r in res.iterrows():
            if str(r["modelo"]).startswith("nulo"):
                continue
            # Stuff+ no ve la cuenta: su nulo es el global; con la cuenta anadida se mide contra el del reto
            contra = "global" if str(r["modelo"]) == "stuff" else "cuenta"
            print(f"  validacion {str(r['modelo']):13s}: log-loss {r['logloss']:.4f}{mejora(r, contra)}")

    def el_documento():
        if pdf is None:
            print("  documento: todavia no existe (se genera en el paso report)")
            return
        n = paginas_pdf(pdf)
        nota = "" if n is None or 6 <= n <= 10 else "  <-- FUERA DEL RANGO 6-10 QUE PIDE EL RETO"
        print(f"  documento: {_ascii(pdf)} ({n} paginas){nota}"
              + ("" if de_ahora else "  <-- NO es de esta corrida (no se corrio 'report' o no se pudo escribir)"))

    def la_corrida():
        info = leer_info()
        if not info:
            return
        if info.get("archivos_de_datos"):
            print("  datos: " + _ascii(", ".join(map(str, info["archivos_de_datos"]))))
        print(f"  codigo {str(info.get('hash_codigo', '?'))[:8]} | modo {'RAPIDO' if info.get('rapido') else 'COMPLETO'}"
              f" | backend {info.get('backend', '?')}")
        a_medias = [n for n in NOMBRES if (info.get("pasos") or {}).get(n, {}).get("terminado") is False]
        if a_medias:
            print(f"  OJO: no terminaron: {', '.join(a_medias)}")
        if info.get("rapido"):
            print("  OJO: hay resultados de una corrida RAPIDA (30% de los pitchers). Para entregar: "
                  "python run_all.py --estricto")

    for archivo, renglon in (("val_contrafactual.csv", contrafactual), ("altitude_elasticities.csv", elasticidades),
                             ("val_calibracion_resumen.csv", calibracion), ("val_resumen.csv", validacion),
                             ("el documento", el_documento), ("run_info.json", la_corrida)):
        try:
            renglon()
        except Exception as e:                  # noqa: BLE001
            print(f"  no se pudo leer {archivo}: {_ascii(type(e).__name__)} ({_ascii(e)[:120]})")


def main():
    rapido, estricto, pasos = leer_argumentos(sys.argv[1:])
    vers = versiones()                          # antes de importar config.py, que ya importa pandas
    faltan = [pip for lib, pip in REQUERIDAS.items() if lib not in vers]
    if faltan:
        print(f"No se pudo importar: {', '.join(faltan)}. Instala lo que pide el proyecto y vuelve a correr:\n"
              "    python -m pip install -r requirements.txt")
        sys.exit(2)
    sys.path.insert(0, str(SRC))
    import config as cfg

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"           # la consola de Windows no siempre es UTF-8
    env.pop("RAPIDO", None)                     # un 'set RAPIDO=1' olvidado no debe achicar la corrida completa
    env.pop("ESTRICTO", None)
    if rapido:
        env["RAPIDO"] = "1"
    if estricto:
        env["ESTRICTO"] = "1"

    archivos, avisos = revisar_antes(pasos, cfg, rapido, estricto, vers)
    info, codigo = leer_info(), hash_codigo()
    avisos += revisar_lo_previo(pasos, info, codigo, rapido)
    for a in avisos:
        print("AVISO:", a)
    if estricto and avisos:
        print("\n--estricto: corrige los avisos de arriba y vuelve a correr (sin --estricto la corrida sigue "
              "con esos avisos).")
        sys.exit(2)
    # Los registros de los pasos NO se borran al empezar: describen lo que hay en disco hasta que el paso se
    # vuelve a correr (una corrida completa que se detiene en features no convierte en completo lo demas).
    info["pasos"] = {k: v for k, v in (info.get("pasos") or {}).items() if isinstance(v, dict)}
    info.update({"hash_codigo": codigo, "estricto": estricto, "backend": cfg.BACKEND,
                 # los archivos que leyo features: solo cambian cuando features se vuelve a correr
                 "archivos_de_datos": archivos if "features" in pasos else info.get("archivos_de_datos", archivos),
                 "versiones": vers, "fecha": time.strftime("%Y-%m-%d %H:%M"),
                 "ultima_llamada": {"rapido": rapido, "estricto": estricto, "pasos": pasos}})
    guardar_info(info)

    print(f"Modo: {'RAPIDO (prueba)' if rapido else 'COMPLETO'} | estricto: {'si' if estricto else 'no'} | "
          f"pasos: {', '.join(pasos)} | codigo {codigo[:8]}")
    pdf_antes = None
    for nombre, cmd in PASOS:
        if nombre not in pasos:
            continue
        t = time.time()
        antes = _marcas(ESCRIBE_AL_FINAL.get(nombre, ()))
        if nombre == "report":
            pdf_antes = antes
        # Desde aqui el paso reescribe sus archivos: su registro ya es el de esta corrida, sin terminar. Si se
        # queda a medias, lo que hay en disco puede seguir siendo de la corrida anterior (antes_rapido).
        info["pasos"][nombre] = {"hash_codigo": codigo, "rapido": rapido, "terminado": False,
                                 "antes_rapido": es_rapido(nombre, info["pasos"].get(nombre, {})),
                                 "fecha": time.strftime("%Y-%m-%d %H:%M")}
        guardar_info(info)
        print(f"\n=== {nombre} ===", flush=True)
        r = subprocess.run([sys.executable, "-u", *cmd], cwd=SRC, env=env)
        escribio_todo = any(m != antes.get(f) for f, m in _marcas(ESCRIBE_AL_FINAL.get(nombre, ())).items())
        if r.returncode == 0 or (estricto and escribio_todo):
            del info["pasos"][nombre]["antes_rapido"]
            info["pasos"][nombre].update(terminado=True, segundos=round(time.time() - t), epoca=round(time.time(), 1),
                                         **({"detenido_por_estricto": True} if r.returncode else {}))
            guardar_info(info)
        if r.returncode != 0:
            if estricto and escribio_todo:
                print(f"\n'{nombre}' escribio todos sus archivos y despues se detuvo por --estricto. Lee su mensaje, "
                      "que esta justo arriba: dice por que y, si se puede seguir, con que comando.")
            else:
                print(f"\nFALLO en '{nombre}'. Corrige y retoma con:  python run_all.py "
                      f"{'--rapido ' if rapido else ''}{'--estricto ' if estricto else ''}--desde {nombre}")
            sys.exit(r.returncode)
        print(f"--- {nombre} listo en {time.time() - t:.0f} s", flush=True)

    pdf, de_ahora = documento(pdf_antes)
    lista_de_control(pdf, de_ahora)
    sin_documento = pdf_antes is not None and (not de_ahora or pdf.name != "documento_tecnico.pdf")
    if sin_documento:
        print("\nAVISO: out/documento_tecnico.pdf NO es el documento de esta corrida"
              + (f" (el nuevo quedo en out/{pdf.name})" if de_ahora else "") + ". Si lo tienes abierto en un visor "
              "de PDF, cierralo y corre:  python run_all.py " + ("--rapido " if rapido else "")
              + ("--estricto " if estricto else "") + "--solo report")
    print("\nListo." + (f" Documento: out/{pdf.name}" + ("" if de_ahora else " (de una corrida anterior)")
                        if pdf is not None else "")
          + "\nDashboard:  python -m streamlit run dashboard/app.py")
    if estricto and sin_documento:
        sys.exit(2)


if __name__ == "__main__":
    main()
