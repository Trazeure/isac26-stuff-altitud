# Stuff+ calibrado por altitud — ISAC Hackathon 2026 (Diablos Rojos)

## Abstract (English)

At 2,236 m, Estadio Alfredo Harp Helú has about 20% less air density than the Mexican League's sea-level
parks, so pitch movement and run values do not transfer between environments. Using only the official ISAC
dataset (618,982 qualified pitches from 1,039 LMB pitchers, 2024–2026), this project builds Stuff+, Location+
and Pitching+ models that predict ten pitch outcomes and convert them into expected run value. Run values are
estimated from the dataset itself with a base-out Markov chain; air density comes from public sources (ERA5
reanalysis and the public LMB schedule). Altitude effects are measured by comparing each pitcher against
himself, and every pitch is evaluated in both Harp Helú and sea-level air. A four-seam fastball's induced
vertical break falls from 17.3 to 12.9 inches, and the same pitch allows +1.51 runs per 100 pitches more at
Harp Helú (95% CI +0.69 to +2.33). Relative to other pitches, curveballs, cutters, sliders and splitters gain
value at altitude, while four-seamers and sinkers lose it. Out-of-sample on unseen pitchers, Pitching+
improves log-loss by 27.3% over a count-only baseline. Interactive dashboard (password-protected; access in
the submission email): https://isac26-stuff-altitud.vercel.app

The ISAC data is not included in this repository: copy the official file to `data/` (section 2).

Ajustado al **diccionario oficial de ISAC** (`stuff_plus_data_dictionary.csv`, de ISAC; no viene en
esta carpeta): el programa acepta todas las columnas y todos los valores que el diccionario enumera.
Si encuentra otros lo dice en pantalla con la cuenta de pitcheos afectados, y con `--estricto` se
detiene cuando pasan de un límite (sección 4).

**Corrida de entrega**: archivo oficial de ISAC (`stuff_model_df`, temporadas 2024-2026; 618,982
pitcheos calificados, 1,039 pitchers), `python run_all.py --estricto`, código `bcd5e452`, en
Windows con Python 3.12 y las versiones de `requirements.txt`. Todas las tablas de `out/`, el
documento técnico (su pie imprime la misma huella) y las dos tablas de predicciones salen de esa
corrida.

**Dashboard en línea (entregable 4)**: https://isac26-stuff-altitud.vercel.app. Trae el simulador
(Stuff+ de un pitcheo en el Harp Helú y al nivel del mar, lado a lado), los mapas de movimiento y de
ubicación, el ranking, la ficha de cada pitcher y un visualizador 3D del Harp Helú. Sus cifras salen del
`out/` de la corrida de entrega y van cifradas; el acceso viene en el correo de entrega.
`dashboard/app.py` es la versión local en Streamlit (sección 3), que además acepta cualquier altitud de
medición, la temperatura y la humedad; se revisó pestaña por pestaña con un sustituto de Streamlit.

## 1. Instalar (una sola vez)

1. Python 3.12 (el de python.org, marcando *"Add Python to PATH"*, o el de Microsoft Store). Es la
   versión con la que se probó; 3.13 debería servir igual. Con 3.10 o 3.11 `pip` instala versiones
   anteriores de las librerías (con 3.10, pandas 2), un camino que no se probó: usa 3.12.
2. En una terminal **dentro de la carpeta del proyecto** (la que tiene `run_all.py`):

   ```
   python -m pip install -r requirements.txt
   ```

Si después de instalar algo falla con un error que nombra a pandas, numpy o scikit-learn, instala
exactamente las versiones con las que se probó (están también al final de `requirements.txt`; piden
Python 3.12 o posterior):

```
python -m pip install pandas==3.0.5 numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 pyarrow==25.0.1 joblib==1.6.0 matplotlib==3.11.1 reportlab==4.2.2
```

## 2. Poner los datos

Copia el archivo de ISAC (`stuff_model_df.parquet`) a la carpeta `data/`. El programa lee **todo** lo
que haya ahí (csv, csv.gz o parquet) y al empezar imprime la lista de archivos que va a leer: no dejes
otros archivos de datos en `data/`.

## 3. Correr

```
python run_all.py --rapido
```
Prueba de que todo funciona (30% de los pitchers y menos árboles: tarda mucho menos que la
completa). Debe terminar con la **lista de control** y el PDF en `out/documento_tecnico.pdf`.

```
python run_all.py
```
Corrida completa (la larga; al terminar cada paso la consola dice cuántos segundos tardó; la de
entrega se hizo con `--estricto`).

```
python src/simulator.py
```
Prueba del simulador. Debe terminar en **PASA**.
Comprueba que el simulador arma, para pitcheos reales, la misma fila que las tablas, y que el camino
que usa el dashboard está bien conectado; no dice nada de la calidad del modelo. Su sección 3 mide
qué tan bien un promedio tecleado reproduce el Stuff+ real de un arsenal: si esa medida queda por
debajo de lo esperado, la última línea es un AVISO y el Simulador y el Prospecto se leen con cautela.

```
python -m streamlit run dashboard/app.py
```
Dashboard en el navegador (abre las cinco pestañas una vez antes de la demo). La primera vez
Streamlit puede pedir un correo en la terminal: presiona Enter para saltarlo.

Si algo falla, el mensaje dice en qué paso y con qué comando retomar (cópialo tal cual: ya trae
`--rapido` si estabas en la prueba rápida). Por ejemplo, para retomar desde `score`:

```
python run_all.py --desde score
```

Para repetir solo algunos pasos (separados por espacio o por coma):

```
python run_all.py --solo insights report
```

- Pasos: `features`, `physics`, `valores`, `validar`, `final`, `score`, `insights`, `report`.
- Un argumento mal escrito (`--rapdio`) detiene todo antes de empezar: no lanza la corrida completa.
- Al retomar, el programa avisa si lo que no vuelve a calcular viene de otro código, de otro modo
  (rápido / completo) o de un paso que no terminó. Si avisa, lo seguro es retomar desde el paso que
  indica.
- Para la **corrida de entrega** usa `python run_all.py --estricto`: los avisos graves detienen el
  pipeline en lugar de dejarlo seguir (portada sin llenar, etiquetas sin clasificar, valor en carreras
  fuera de lo razonable, documento fuera de 6-10 páginas).
- Con `--estricto`, `score` puede detenerse **al final** si el contrafactual de altitud no pasa su
  prueba de aceptación. En ese caso ya escribió todos sus archivos y su mensaje dice qué revisar y
  con qué comando seguir (`python run_all.py --estricto --desde insights` si las categorías de
  altitud están bien).
- Si tienes `out/documento_tecnico.pdf` abierto en un visor, Windows no deja reescribirlo: el
  documento nuevo queda en `out/documento_tecnico_nuevo.pdf` y la última línea de la corrida lo
  dice. Cierra el visor y corre `python run_all.py --solo report` (ese `_nuevo` se borra solo en
  cuanto el documento se puede escribir con su nombre).
- Lo mismo pasa con un CSV de `out/` abierto en Excel: el paso que lo reescribe falla. Cierra Excel
  antes de correr.
- El documento se compacta solo (texto y figuras un poco más chicos) para quedar en 10 páginas o
  menos, como pide el reto; si tuvo que compactar, la consola dice cuántas páginas salieron en cada
  nivel.

## 4. El día del hackathon (dataset real de ISAC)

1. Llena `EQUIPO`, `INTEGRANTES` e `INSTITUCION` en `src/config.py`. Llenarlos después de la corrida
   no la invalida: basta `python run_all.py --solo report`.
2. Deja en `data/` solo el archivo de ISAC: `stuff_model_df.parquet` (el `.pkl` y el `.rds` son la
   misma tabla y el programa no los lee).
3. `python run_all.py --rapido`. `src/config.py` ya trae las tres categorías del archivo oficial
   (`No Altitude`, `Medium Altitude`, `Extreme Altitude`) con sus metros y su densidad; el
   comentario de `ALTITUDE_M` explica de dónde salen. Si un archivo trajera otras etiquetas, la
   corrida se detiene y dice cuáles encontró: ponlas en `ALTITUDE_M` (etiqueta → metros), en
   `ALTITUDE_RHO` (su densidad) y en `HOME_ALTITUDE` la del Harp Helú.
4. Revisa en pantalla la tabla de **chequeo de etiquetas** (swing, whiff, zona): compara lo que el
   programa deriva de `PitchCall` contra las banderas oficiales del archivo.
5. Lee los avisos de la limpieza. Los que piden una corrección dicen dónde hacerla:

   | Mensaje en pantalla | Qué hacer |
   |---|---|
   | `PitchCall sin reconocer` | Agrega la etiqueta a `PITCH_CALL_MAP` en `src/config.py` (ball / called_strike / whiff / foul / in_play / hbp) |
   | `Batazos (hit_type) sin reconocer` | Agrégala a `HIT_TYPE_MAP` en `src/config.py` (gb / ld / fb / pu / bunt) |
   | `Resultado del batazo (play_result) sin reconocer` | Agrégala a `PLAY_RESULT_MAP` en `src/config.py` (Single / Double / Triple / HomeRun / Error / Out, con esas mayúsculas) |
   | `Tipo de pitcheo sin reconocer` | Agrégala a `PT_MAP` en `src/features.py` |
   | `Bandera ... sin reconocer` | Agrega el texto a `BANDERA_SI` / `BANDERA_NO` en `src/features.py` |
   | `Lado del bateador (BatterSide) sin reconocer` | El valor debe empezar con R o L. Si no, esos pitcheos se quedan sin lado del bateador (solo afecta a Pitching+) |
   | `Batazos sin rastreo` | Nada: son batazos sin medición (sin tipo, ángulo ni velocidad de salida; 6,539, el 5.7%, en la corrida de entrega). Se descartan y el documento lo dice. Con `--estricto` solo detiene si pasan de 15% |
   | `Mas de la mitad de los batazos no trae un tipo de batazo reconocible` | La corrida se detiene siempre: revisa la columna `hit_type` (`COLUMN_MAP`) y sus etiquetas (`HIT_TYPE_MAP`) |
   | `Para calcular el valor en carreras hacen falta ...` o `Solo ... medias entradas completas` | La corrida se detiene siempre: el valor en carreras se estima con las medias entradas del archivo (sección 5) y necesita `game_anon_id`, `Inning`, `Top/Bottom`, `RunsScored` y `OutsOnPlay` (`COLUMN_MAP`) |
   | `el valor en carreras no salio razonable` | Revisa `RunsScored`, `OutsOnPlay`, `KorBB` y `play_result` en el archivo. Con `--estricto` detiene la corrida |

   La etiqueta se puede escribir tal como la imprime el aviso (`"Fly Out": "Out"`): mayúsculas,
   espacios y guiones no cuentan. Un valor que no sea de los listados detiene la corrida y lo dice.

   Sin `--estricto` esos avisos (menos los dos que detienen la corrida siempre) dejan seguir y dicen
   cuántos pitcheos afectan. Con
   `--estricto` detienen la corrida cuando pasan de un límite (0.5% de los batazos para
   `play_result`, 1% para `PitchCall`, `hit_type` y las banderas, 3% para el tipo de pitcheo, 5%
   para el lado del bateador).
6. `python run_all.py --estricto` → lista de control → `python src/simulator.py` → documento →
   dashboard.

## 5. Datos: el archivo de ISAC y, para el aire, fuentes públicas

Todos los datos de pitcheo y de resultados salen del archivo de ISAC (`stuff_model_df`). El archivo
no trae estadio ni fecha: trae `altitude_category`, y el programa trabaja con eso: cada categoría usa
su densidad de aire promedio y "casa" es toda la categoría del Harp Helú (`HOME_ALTITUDE`), no solo
ese parque. El documento y el dashboard lo dicen.

**El valor en carreras sale del mismo archivo.** ISAC no trae run value, así que el paso `features`
lo estima con el propio dataset (`features.estimar_valores_carrera`): junta las medias entradas
completas (juego, entrada y mitad, entradas 1 a 9), calibra con ellas una cadena de Markov de bases y
outs (el archivo no trae corredores, así que su avance se ajusta para reproducir las carreras de cada
tipo de jugada, el doble play y las carreras por media entrada) y de ella saca el peso de cada evento:
lo que cambia la expectativa de carreras (RE24) más lo que anota. El valor de cada cuenta sale de la
cadena de cuentas de la liga. Todo queda en `out/valores_carrera.json`: los pesos con su intervalo
por remuestreo de juegos, el valor de cada cuenta y, como contraste, una regresión de las carreras de
cada media entrada sobre sus eventos. El documento técnico explica cómo se estimó.

**Lo único externo es el aire**, como constantes de `src/config.py`, y el documento lo declara: la
densidad media del aire de cada categoría de altitud y la de referencia del Harp Helú y del nivel del
mar (`ALTITUDE_RHO`, `RHO_CDMX`, `RHO_MAR`), calculadas aparte con `src/clima_lmb.py` solo con fuentes
públicas: el calendario público de la LMB 2024-2026 (MLB Stats API) para saber en qué estadio y a
qué hora se jugó cada juego, y el reanálisis ERA5 (vía Open-Meteo) para la temperatura, la humedad y
la presión a la mitad de cada juego (90 minutos después de la hora programada). Los metros del Harp
Helú y de cada categoría (`CDMX_M`, `ALTITUDE_M`) salen de altitudes públicas de los parques (modelo
de elevación Copernicus GLO-90, vía Open-Meteo; las de cada categoría, promediadas por juego). Qué
parques caen en cada categoría no lo dice el diccionario: se dedujo del dataset, y el comentario de
`ALTITUDE_M` en `src/config.py` explica cómo.

Salidas por pitcheo para entregar: `out/predicciones_por_pitcheo.csv.gz` (columnas oficiales: los
Plus, el xRV, las diez probabilidades de resultado y `p_swing`, `p_chase` y
`p_linea_o_elevado_duro`) y `out/predicciones_extra_por_pitcheo.csv.gz` (`pitching_plus_neutral`,
con los batazos en el mismo aire para todos los parques, y `p_out_turno`, el Out+, con
`p_out_nulo_cuenta`). Traen **solo los pitcheos calificados** (618,982 en la corrida de entrega): los
que la limpieza descarta (bolas intencionales y automáticas, pitcheos sin cantar, toques, pitcheos
sin tipo reconocido, de tipo `Other` o knuckleball, pitcheos sin velocidad, movimiento, ubicación o
cuenta, y los batazos sin rastreo; la consola de `features` dice cuántos por cada motivo) no están y
al unirlos por `PitchUID` quedan sin predicción. Son archivos grandes (`score.py` imprime el tamaño): mándalos por
un enlace compartido solo con ISAC, no como adjunto, y no los publiques (los datos son de ISAC).

## 6. Qué hace cada archivo

| Archivo | Qué hace |
|---|---|
| `src/config.py` | Lo que normalmente se edita: columnas ISAC, etiquetas, categorías de altitud y su aire, equipo (los tipos de pitcheo y los textos de las banderas están en `src/features.py`, sección 4) |
| `src/features.py` | Limpieza, zurdos espejeados, arsenal, ubicación, densidad del aire, valor en carreras estimado con el propio dataset (`out/valores_carrera.json`), run value de cada pitcheo, transporte de un pitcheo a otro aire |
| `src/physics.py` | Efecto medido de la altitud: elasticidad por pitcheo (cada pitcher contra sí mismo), placebos, carry de batazos, adaptación |
| `src/model.py` | Stuff / Location / Pitching (10 resultados → xRV), validación cruzada por pitcher, temporada y categoría de altitud fuera de muestra, calibración, prueba de H4 |
| `src/score.py` | Scores fuera de muestra por pitcheo, pitcher y arsenal; Harp Helú vs nivel del mar; pruebas del contrafactual; error del modelo |
| `src/insights.py` | Formas que ganan o pierden, perfil por pitcheo, hipótesis H2 y H3, listas de pitchers |
| `src/simulator.py` · `dashboard/app.py` | Simulador y dashboard (5 pestañas); `python src/simulator.py` es su prueba |
| `src/report.py` | Documento técnico PDF: todas las cifras y todos los veredictos salen de `out/`; también escribe `out/predicciones_extra_por_pitcheo.csv.gz` (Pitching+ neutral y Out+ por pitcheo) |
| `src/clima_lmb.py` | Utilidad aparte, solo con fuentes públicas (calendario público de la LMB en MLB Stats API, reanálisis ERA5 y elevación Copernicus GLO-90 vía Open-Meteo): calcula la densidad del aire a la mitad de cada juego, de donde salen las constantes de densidad y clima de `src/config.py`. Se corre con `python src/clima_lmb.py clima_publico`. No se usa en la corrida normal |
| `run_all.py` | Corre todo en orden y al final imprime la lista de control |
| `exportar_web.py` | Después de la corrida (`python exportar_web.py`): escribe `web_datos.zip` con lo que muestra la página web: los resúmenes y rankings de `out/`, los mapas de calor y una rejilla de Stuff+ del modelo del simulador, y las trayectorias promedio (ajuste de 9 parámetros, columnas `x0`…`az0`) por tipo y por arsenal, que lee del archivo de ISAC en `data/`. No lleva pitcheos sueltos ni datos crudos |
| `clima_publico/` | Salida de `src/clima_lmb.py`: `constantes.json` (de ahí salen las densidades y el clima de `config.py`) y `clima_resumen_estadio.csv` (elevación y promedios de cada estadio). El script escribe además `clima_por_juego.csv` y guarda las respuestas de las APIs en `crudo/`, que reusa si vuelve a correr |

`out/insights_top_cdmx.csv`, `insights_afectados.csv` e `insights_consistentes.csv` traen también el
Pitching+ neutral con su intervalo (`pitching_neutral_plus_enc`, `pitching_neutral_lo`,
`pitching_neutral_hi`) y el `fold` del modelo que calificó a cada pitcher.

## 7. Decisiones que hay que saber defender

- **Stuff+ solo ve la física del pitcheo**: velocidad, quiebre, spin, punto de salida, extensión y
  los diferenciales contra la recta del propio pitcher. No ve la cuenta, la ubicación, el uso, el
  estadio ni la densidad del aire (con 3 categorías de altitud, la densidad como variable directa
  solo identificaría la categoría, no la física del pitcheo).
- **El túnel es del repertorio, no de cada pitcheo**: es la distancia entre el punto de salida
  promedio de un tipo y el de la recta primaria del pitcher, y vale 0 para esa recta. Con el punto
  de salida de cada pitcheo medía la dispersión de la mano (comando), que no es "stuff".
- **La altitud entra por dos caminos medidos**: (1) el movimiento se lleva de un aire a otro con la
  elasticidad medida en la liga, pitcher contra sí mismo; (2) los batazos valen distinto según la
  densidad del aire. Los dos se ponen a prueba en cada corrida (`out/val_contrafactual.csv`): si el
  transporte reproduce el movimiento observado en altura y si el Stuff+ neutral de un mismo pitcher
  depende de dónde lanzó. La lista de control dice si pasan; no lo des por hecho.
- **Scores fuera de muestra**: cada pitcher lo califica un modelo que nunca vio sus pitcheos.
- **Dos escalas que no se comparan**: puntos por pitcheo (10 = una desviación estándar entre
  pitcheos) y puntos entre pitchers (10 = una desviación estándar entre pitchers).
- **Los intervalos por pitcher no incluyen el error del modelo**. `score.py` lo mide aparte (el
  error típico de un modelo de la validación, comparando los que califican a los mismos pitchers) y
  el documento y el dashboard lo ponen junto al intervalo en la misma unidad (una desviación
  estándar). En la corrida de entrega (5 modelos calificando las mismas 196 pitcher-temporadas) es de
  3.5 puntos en Stuff+ y 2.5 en el efecto altitud, contra intervalos impresos que equivalen a 2.1 y
  1.9 (el semiancho medio del IC 95% de `scores_pitcher.csv` entre 1.96); además, el nivel medio de
  dos modelos llega a diferir 1.6 puntos en Stuff+ y 0.7 en el efecto altitud (`out/val_stability.json`,
  claves `error_modelo_*_pts` y `desfase_modelos_*_pts`).
- **El ruido de muestreo de un pitcher se mide por juego** (los pitcheos de un mismo juego se
  parecen). Si al archivo le falta `game_anon_id` en más de 2% de los pitcheos, se suponen pitcheos
  independientes, los intervalos salen angostos y el documento y el dashboard lo dicen.
- **Run value**: ISAC no lo trae. Se calcula con pesos lineales y valores por cuenta estimados con el
  propio dataset: una cadena de Markov de bases y outs calibrada con sus medias entradas completas y
  la cadena de cuentas de la liga (sección 5, `out/valores_carrera.json`).
- **Validación**: pitchers, temporada y categoría de altitud fuera de muestra; contra el nulo por
  cuenta del reto y contra el nulo global; calibración por deciles; intervalos por bootstrap de
  pitchers.
- **H2** (los pitcheos que viven del diferencial de velocidad ganan valor relativo) se decide con
  cambio + split contra toda la liga; los contrastes contra slider + curva y contra las rectas se
  reportan con su propio veredicto. **H3** (los pitchers de rodados valen más en altura) dice "sin
  datos" cuando menos de 30 pitchers tienen 150+ pitcheos en cada banda de altitud.
- **Nada del documento está escrito a mano**: cada frase que afirma una dirección sale del número y
  su intervalo; si el intervalo incluye el cero, el documento dice "sin diferencia medible".

### El simulador y el dashboard

- Se captura el **promedio** de un pitcheo y hay que decir **dónde se midió** (y, si se quiere, con
  qué temperatura y humedad): el mismo quiebre vale distinto según el aire.
- Todo lo horizontal se captura con **lado del brazo positivo**, igual para diestros y zurdos.
- Un promedio no se califica como si fuera un solo pitcheo: se le suma la dispersión con que los
  pitchers de la liga repiten ese tipo (300 repeticiones) y se promedia. El dashboard muestra el
  error de simulación de cada número.
- El simulador usa el modelo ajustado con todos los datos; el Ranking, los modelos de la validación
  cruzada. **No dan el mismo número.** El dashboard mide en cada corrida cuánto coinciden para los
  pitchers de la liga y solo llama "comparable" al Stuff+ de un prospecto si la correlación llega a
  0.8; si no, lo llama "orientativo". `python src/simulator.py` imprime esa correlación.
- No está modelado que en aire delgado la bola pierde menos velocidad en el camino, ni la
  ubicación, ni la dispersión propia del prospecto (se usa la típica de la liga).

## 8. Reglas del evento

Equipos de 2 a 4 **estudiantes**; la entrega final debe usar el dataset oficial de ISAC.
