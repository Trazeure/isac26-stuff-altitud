# Stuff+ calibrado por altitud — ISAC Hackathon 2026 (Diablos Rojos)

El Estadio Alfredo Harp Helú está a 2,236 m. Ahí el aire es cerca de 20% menos denso que en los parques
al nivel del mar de la Liga Mexicana de Beisbol, así que los pitcheos se mueven menos y la pelota bateada se
frena menos. El ajuste de parque de siempre corrige a todos los pitchers por igual, pero el aire no les
cobra igual a todos: depende de qué tira cada quien.

Este proyecto califica cada pitcheo dos veces, con el aire del nivel del mar y con el del Harp Helú, para
saber cuánto vale cada pitcher en casa de los Diablos antes de firmarlo y con qué conviene atacar cuando
abre ahí.

**Plataforma interactiva:** https://isac26-stuff-altitud.vercel.app

## Resultados principales

- La misma recta de cuatro costuras sube 17.3 pulgadas al nivel del mar y unas 12.9 en el Harp Helú.
- En el Harp Helú, cada 100 pitcheos cuestan en promedio 1.5 carreras más que al nivel del mar
  (intervalo de 95%: +0.69 a +2.33). La recta que más sube es la que más paga; al slider y a la curva les
  cuesta más o menos la mitad.
- Dos pitchers con el mismo Stuff+ al nivel del mar (105) pueden quedar en 87 y en 116 en el Harp Helú.
- Con pitchers que el modelo nunca vio, Pitching+ se equivoca 27.3% menos (log-loss) que un modelo que solo
  conoce la cuenta.

## Cómo funciona

1. **Datos:** solo el archivo oficial de ISAC (618,982 pitcheos de 1,039 pitchers de la LMB, 2024–2026).
   Para el aire se usan fuentes públicas: el calendario de la LMB, el clima ERA5 y la elevación
   Copernicus GLO-90.
2. **Modelos:** Stuff+ (velocidad y movimiento), Location+ (ubicación) y Pitching+ (todo junto). Cada uno
   predice la probabilidad de 10 resultados posibles del pitcheo con clasificadores HistGradientBoosting
   de scikit-learn.
3. **Valor en carreras:** cada resultado vale lo que cambia las carreras esperadas según outs y corredores
   (una cadena de Markov estimada con el propio dataset). La escala es 100 + 10·z: 100 es el promedio de la
   liga.
4. **Altitud:** el efecto del aire en el movimiento se mide comparando a cada pitcher contra sí mismo en
   parques altos y bajos (742 pitchers). Con eso cada pitcheo se lleva al aire de cada lugar y se califica
   en los dos.
5. **Validación:** fuera de muestra, por pitcher y por parque. Cada pitcher lo califica un modelo que nunca
   vio sus pitcheos.

## Cómo correrlo

```
pip install -r requirements.txt
python run_all.py                       # corrida completa
python -m streamlit run dashboard/app.py
```

Los datos de ISAC no vienen en el repositorio: el archivo oficial va en `data/` (ver `data/LEEME.txt`).

## Estructura

| Archivo | Qué hace |
| --- | --- |
| `run_all.py` | Corre todo el proceso en orden |
| `src/config.py` | Configuración: categorías de altitud, densidad del aire y etiquetas del dataset |
| `src/features.py` | Variables de cada pitcheo y valor en carreras |
| `src/physics.py` | Efecto del aire en el movimiento, cada pitcher contra sí mismo |
| `src/model.py` | Modelos de Stuff+, Location+ y Pitching+ y su validación |
| `src/score.py` | Calificaciones por pitcheo, pitcher y arsenal en los dos aires |
| `src/insights.py` | Hipótesis del reto y listas para Diablos |
| `src/report.py` | Documento técnico |
| `src/simulator.py` | Simulador de un pitcheo en los dos aires |
| `src/clima_lmb.py` | Densidad del aire con fuentes públicas |
| `dashboard/app.py` | Tablero interactivo local |
| `exportar_web.py` | Datos resumidos para la plataforma web |

## Equipo

Equipo Trez — Luis Ángel Córdova Gil y José Ángel López Mercado. Instituto Politécnico Nacional, ESCOM.

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
improves log-loss by 27.3% over a count-only baseline. Interactive dashboard:
https://isac26-stuff-altitud.vercel.app
