# Satellite Flood Mapping and Road Access

Team riwaq: Sadat Bashir and Adfar Jaan.

Satellite images acquired before and after a flood are compared to map water and deposited debris, using
Sentinel-1 backscatter change, Sentinel-2 reflectance change and water segmentation. Slope, height above drainage
and river connections restrict detections on the valley sides. The extent is compared with buildings, roads
and bridges from pre-event OpenStreetMap, and routes are calculated again after affected road segments are
removed. This identifies settlements that have lost road access even when their buildings lie outside the
flood extent.

![Trishuli flood extent and settlement access](docs/dashboard.jpg)

The case study is the Bhote Koshi–Trishuli flood of 26 August 2026. Its dashboard contains the flood extent,
settlements without road access, the breaks on their usual routes and repair candidates ranked by the number
of buildings reconnected. Walking time is estimated where a mapped trail still connects to a road. English
and Nepali situation reports contain the same counts and acquisition dates.

[Dashboard](results/case_study_trishuli_2026/dashboard.html) ·
[English report](results/case_study_trishuli_2026/sitrep.pdf) ·
[Nepali report](results/case_study_trishuli_2026/sitrep_ne.pdf) ·
[Technical report](docs/report/REPORT.pdf)

## Run

Use Linux with Python 3.11 or later. Measured CPU runs used eight vCPUs and 50 GB RAM; use 32 GB RAM or more and
about 10 GB free disk, with more disk space for large OpenStreetMap extracts. A CUDA GPU is optional.
The run downloads satellite images, terrain and pre-event OpenStreetMap over HTTPS.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y osmium-tool aria2
python -m pyproj sync --file us_nga_egm08_25
export TRI_ROOT=$PWD/runs TRI_MODELS=$PWD/models
python src/preflight.py
python src/pipeline.py --event trishuli_2026 --bbox 84.85 27.75 85.65 28.45 --date 2026-08-26
```

The four model folders `r34_scratch_v2`, `r50_scratch_v2`, `r34_s1f11rl_v1` and `s2ai_r34_v1` belong under
`models/`, each containing `best.pt` and `config.json`. The local checkpoints are excluded from Git.
Without them, the run uses change detection and records the missing water models in `summary.json`.
[Running instructions](docs/RUNNING.md), [model details](docs/MODEL_CARD.md) and
[training commands](training/README.md) describe the required inputs and model preparation.

## Evaluation

| Evaluation | Flood-extent F1 |
|---|---|
| Trishuli, four EMSR927 areas pooled | 0.906 |
| Combined map, 27 events | 0.546 |
| UN-SPIDER implementation, same 27 events | 0.258 |
| Otsu thresholding, same 27 events | 0.259 |
| Thame, Buner and Charikar mean | 0.366 |

Trishuli and the other development events were used to select mapping settings, while validation and lowland
comparisons also informed the rules for plains. Thame, Buner and Charikar were excluded from further selection
after scoring; Charikar used a different plains configuration. Mapped infrastructure damage and road access have not
been checked on the ground.

## Data and attribution

Sentinel-1 GRD, Sentinel-2 L2A and Copernicus DEM GLO-30 are obtained from Microsoft Planetary Computer.
OpenStreetMap is taken from a dated Geofabrik extract or historical query, with its timestamp checked against
the event. Copernicus EMS and UNOSAT maps are used only for evaluation.

Contains modified Copernicus Sentinel data 2026 and 2017–2025 for the evaluation events. Produced using Copernicus
WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the
European Union and ESA; all rights reserved. © OpenStreetMap contributors, with derived layers under ODbL.
Training data: Bountos et al. (2024), Kuro Siwo; Bonafilia et al. (2020), Sen1Floods11.

Claude and Codex were used to edit code and documentation.
