# Running the pipeline

Supply an area and a flood date. The run downloads Sentinel-1, Sentinel-2 and Copernicus DEM from Microsoft
Planetary Computer, and obtains OpenStreetMap from before the flood. The flood extent and infrastructure status
are written first, after which the road-access results, dashboard and situation report are produced.

## 0. Machine
- Linux, Python ≥ 3.11. Tested with Python 3.13.
- Recommended: 8 vCPU and ≥ 32 GB RAM. The measured runs used 8 vCPU / 50 GB.
- A GPU is optional. With CUDA, water predictions are averaged over rotations and reflections and run faster.
- About 10 GB of free disk, more if large Geofabrik OSM extracts are cached.
- Outbound HTTPS to:
  - `planetarycomputer.microsoft.com` and `*.blob.core.windows.net`;
  - `cdn.proj.org`;
  - `download.geofabrik.de`, `api.ohsome.org`, `overpass-api.de`.
- Windows and macOS are untested. On those systems the Sentinel-1 step runs in-process, without the fork pool, and is
  slower.

## 1. Install
```bash
cd mapping-flood-damage
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip
# GPU only: first install the torch build for your CUDA version, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
sudo apt-get install -y osmium-tool aria2
python -m pyproj sync --file us_nga_egm08_25
export TRI_ROOT=$PWD/runs TRI_MODELS=$PWD/models
```

## 2. Model weights
The pipeline expects four U-Net checkpoints, one folder per model:
```
models/r34_scratch_v2/{best.pt, config.json}     radar water model ensemble member (Kuro Siwo)
models/r50_scratch_v2/{best.pt, config.json}     radar water model ensemble member (Kuro Siwo)
models/r34_s1f11rl_v1/{best.pt, config.json}     radar water model ensemble member (Kuro Siwo + Sen1Floods11)
models/s2ai_r34_v1/{best.pt, config.json}        optical water model (Sen1Floods11 S2 hand labels)
```
- The model files are provided in the local `models/` folder.
- `python src/preflight.py` checks that each model checkpoint loads.
- Without the weights the run still completes. The map then uses the rule-based radar and optical evidence only, and
  the summary records a warning and an error for the `dl` / `ais2` steps.

## 3. Preflight
Check the runtime before starting a run:
```bash
python src/preflight.py            # or: python src/pipeline.py --preflight
```
The preflight returns exit code 1 if a requirement fails. Its PASS, WARN and FAIL rows cover:
- the Python packages (and prints their versions);
- that the output root is writable, and the free disk space;
- that `osmium` and `aria2c` are installed;
- that the EGM2008 geoid grid is on disk (it downloads the grid if missing);
- that a Planetary Computer STAC search and a signed read work for S1 GRD, S2 L2A and Copernicus DEM;
- the model weights;
- that the OSM sources are reachable;
- CPU, RAM and torch/CUDA.

A FAIL row identifies a missing requirement. WARN rows identify slower processing or unavailable data; check the
listed reason before running.

## 4. Run
```bash
# Check the area and date
python src/pipeline.py --event judge_area_1 --bbox 85.20 27.55 85.55 27.82 --date 2024-09-27 --check
# Run (--source LON LAT adds upstream drainage tracing)
python src/pipeline.py --event judge_area_1 --bbox 85.20 27.55 85.55 27.82 --date 2024-09-27
```
- `--bbox` is **W S E N** in degrees (lon/lat).
- `--event` names the output folder `<root>/<event>`. Use letters, digits, `.`, `_` and `-`.
- Steps run in this order: `dem → s1 → geometry → s1_rules → s1_joint → s1x → s2 → s2_rules → s2_cal → dl → ais2 →
  fusion → osm → impacts → flowpath → report`. `--only <steps>` runs a subset.
- Exit codes:
  - `0`: done. Warnings may remain; see `summary.json`.
  - `1`: a fatal step failed (`dem` or `fusion`).
  - `2`: bad input. Nothing ran.
  - `3`: run-key mismatch. No processing was started; the cached-input check is described below.

### Outputs (`<root>/<event>/`)
- `report/dashboard.html`: the map dashboard. The data are embedded. The map libraries load from CDNs, so the dashboard
  needs internet.
- `report/sitrep.md|html`, `report/sitrep_ne.md|html`: one-page situation reports in English and Nepali, with the same
  numbers.
- `report/*.geojson`: the vector layers.
- `hazard/hazard.tif`: 0 = no flood evidence, 1 = possible, 2 = likely, 255 = unobserved.
- `impact/*.gpkg`.
- `summary.json`: counts, acquisition dates, parameters, timings and `warnings` (plain-language list), `errors` (per step), `timing`,
  `data_availability`, `small_bbox`, `code_version`.
- `run_key.json`: the bbox, the date as given, the interpreted UTC start and the code version of the first run.

## 5. Re-runs, the run key and `--fresh`
- Steps resume: a step whose outputs exist is skipped. An interrupted run continues where it stopped.
- The first run writes `run_key.json`. A later run with the **same `--event` but another bbox or date is refused**
  (exit 3), and nothing runs. Without this check the run would reuse the cached grid, DEM, Sentinel stacks and OSM of
  the old inputs, possibly including OSM from *after* the new event.
- In that case, use a new `--event` name, or add `--fresh`. `--fresh` deletes `<root>/<event>` and starts over. It leaves
  any copy in the optional project store (`--push`) untouched.
- If the code changed since the first run, the run prints a note and reuses the cached outputs. `--fresh` recomputes
  everything.

## 6. Dates and times are UTC
- `--date` accepts:
  - `YYYY-MM-DD`: date only;
  - `YYYY-MM-DDTHH:MM[:SS]`: **taken as UTC**;
  - a time with an offset: `2026-08-26T03:15+05:45` (Nepal time) or `...Z`.
- The run prints the UTC instant it uses, e.g. `== event start 2026-08-25T21:30:00 UTC`.
- **Time given:** images acquired at or after that instant are "post"; earlier ones are "pre". If the local flood time is
  known, give it with its offset.
- **Date only:** the date is a **UTC calendar day**. Images acquired on that day are ambiguous and are not used. Images
  on earlier UTC days are "pre", later ones "post".
- **Local and UTC days differ:**
  - **East of Greenwich** (e.g. Nepal, UTC+5:45): a flood early on local day D falls on UTC day D−1. A satellite pass on
    the afternoon of UTC day D−1 is then labelled "pre", although it could show the flood.
  - **West of Greenwich** (e.g. the Americas): an evening pass on local day D falls on UTC day D+1 and is labelled
    "post", although it may precede a late-evening flood.
  - In such cases, give a time with an offset, or give the last day that is certainly before the flood.
- Pre-event OSM is taken strictly before the event. A Geofabrik extract must have a replication timestamp before 00:00
  UTC of the event's UTC date. Overpass/ohsome queries use the event date minus 1 day. With a time and offset, the run
  uses the UTC date, which can precede the local calendar date east of Greenwich.

## 7. Area limits
- The bbox must have W < E, S < N and |lat| < 84°. Each side must be > 0.01°. Areas across the antimeridian are not
  supported.
- Above ~2,500 km² the run warns about processing time and memory use.
- Above ~12,000 km² the run is refused unless you pass `--force-large`.
- Grids narrower than 256 px (≈ 2.6 km) get a `small_bbox` note and a warning. The water models use 512-px tiles.
- At high latitudes Sentinel-1 is often HH/HV only, and the pipeline processes only IW VV+VH. The run warns and continues
  with Sentinel-2 only.

## 8. When a step fails
| Failure | Result |
|---|---|
| DEM (Copernicus DEM download/processing) | **Fatal**: the run stops (exit 1); `summary.json` holds the error |
| Sentinel-1 step (e.g. geoid grid unavailable after one automatic `pyproj sync` attempt, STAC error) | Continues **without radar**: the map comes from Sentinel-2 only, and the radar water model is not used. A warning is recorded. The next run of the same event retries Sentinel-1 |
| Geometry (layover/shadow annotation fetch) | Continues. Layover/shadow count as absent for the affected tracks; a warning is recorded |
| Water model (no weights, no torch, tiny area) | Continues with the rule-based evidence. A warning is recorded |
| Fusion | **Fatal** (exit 1) |
| OSM / impacts / flowpath / report | Continue. Each failure is recorded in `errors` and `warnings` |

`summary.json` records every failure in `errors` (per step, technical) and in `warnings` (plain language). The report and
the dashboard show them. When a failed step succeeds on a later run, its entries are removed from both lists.

## 9. Run times
End to end from raw data, final code:

| Area | Machine | Time |
|---|---|---|
| 504 km² (Charikar) · ~550 km² (Thame) | 8-vCPU CPU VM | 19 min · 24 min |
| 994 km² (Buner) · 1,331 km² (Kurigram) | 8-vCPU CPU VM | 41 min · 29 min |
| 1,470 km² (Sahuka) · 1,529 km² (Larkana) | 8-vCPU CPU VM | 36 min · 46 min |
| 1,311 km² (Sikkim, unseen area) | L4 GPU VM (12 vCPU) | **36.5 min** (radar water model 3.3 min with TTA, optical water model 0.6 min) |
| 6,080 km² (Trishuli case-study box) | 8-vCPU CPU VM | 1 h 51 min (radar water model 39 min, S2 18 min, S1 15 min) |

Data download (Sentinel-1/2, OSM) takes most of the time. A GPU reduces model inference time. The first use of an OSM
region adds the Geofabrik extract download; large multi-country extracts can take tens of minutes.

## 10. Folders and environment variables
| Variable | Default | Meaning |
|---|---|---|
| `TRI_ROOT` | `/content/tri/events` if `/content/tri` exists, else `./runs` | output root (`--root` overrides) |
| `TRI_MODELS` | `/content/tri/ckpt` if `/content/tri` exists, else `<repo>/models` | model weights |
| `TRI_OSM_BUDGET` | `900` | seconds allowed for the Overpass/ohsome fallback |
| `TRI_STORE_URL` | — | URL of the optional project store used with `--push` |
| `TRI_STORE_TOKEN` | — | token for the optional project store (`--push`) |

Export `TRI_ROOT` and `TRI_MODELS` as in step 1. If `/content/tri` exists, the defaults switch to the paths under it.

## 11. Technical report

Compile the report with [Tectonic](https://tectonic-typesetting.github.io/en-US/install.html):

```bash
cd docs/report
tectonic REPORT.tex
```

Keep the two JPEG figures and four PDF plots beside `REPORT.tex`. The command produces `REPORT.pdf`.
To regenerate the plots from the recorded results, install `matplotlib` and run
`python docs/report/plot_results.py` from the repository root.

## 12. Demo video and upload figures

The captioned demo is in `video/trishuli_flood_mapping_demo_revised.mp4`; its text is in
`docs/VIDEO_SCRIPT.md`. The upload figures are in `upload_figures/`.

To regenerate the demo on Windows, install Chrome, FFmpeg, Pillow and PyMuPDF. Run these commands
from the repository root to capture the saved dashboards, encode the video and refresh the upload figures:

```powershell
python video/make_video_revised.py
python upload_figures/make_figures.py
```
