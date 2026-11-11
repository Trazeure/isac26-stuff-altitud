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
improves log-loss by 27.3% over a count-only baseline. Interactive dashboard:
https://isac26-stuff-altitud.vercel.app

