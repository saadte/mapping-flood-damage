# Running the system

Supply an area and a flood date. The program downloads Sentinel-1, Sentinel-2 and Copernicus DEM from Microsoft
Planetary Computer, and obtains OpenStreetMap from before the event. It maps the flood extent, calculates
infrastructure exposure and road access, then writes the dashboard and English and Nepali situation reports.

## Requirements

Use Linux and Python 3.11 or later, with at least 32 GB RAM and about 10 GB free disk. Large OpenStreetMap
extracts require more disk space. Measured CPU runs used eight vCPUs and 50 GB RAM. A CUDA GPU reduces water-model
inference time; GPU predictions average rotations and reflections that CPU runs omit.

The machine needs outbound HTTPS access to Microsoft Planetary Computer, Azure Blob Storage, `cdn.proj.org`,
Geofabrik, Overpass and ohsome. Windows and macOS have not been tested for the complete pipeline.

## Install

```bash
cd mapping-flood-damage
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y osmium-tool aria2
python -m pyproj sync --file us_nga_egm08_25
export TRI_ROOT=$PWD/runs TRI_MODELS=$PWD/models
```

For GPU inference, install a CUDA-compatible PyTorch build before installing the remaining requirements.
`TRI_ROOT` sets the output folder and `TRI_MODELS` sets the model folder.

## Model weights

The pipeline reads these four model folders:

```text
models/r34_scratch_v2/
models/r50_scratch_v2/
models/r34_s1f11rl_v1/
models/s2ai_r34_v1/
```

Each folder must contain `best.pt` and `config.json`. The first three models predict water from radar, while
`s2ai_r34_v1` uses Sentinel-2. Model definitions and training are described in [MODEL_CARD.md](MODEL_CARD.md)
and [training/README.md](../training/README.md). The local checkpoints are excluded from Git. If weights are
missing, the run continues with change detection and records the missing models in `summary.json`.

## Check and run

```bash
python src/preflight.py
python src/pipeline.py --event nepal_2024 --bbox 85.20 27.55 85.55 27.82 --date 2024-09-27 --check
python src/pipeline.py --event nepal_2024 --bbox 85.20 27.55 85.55 27.82 --date 2024-09-27
```

The preflight checks dependencies, available disk, model loading and access to the data services. `--check`
checks the supplied area and date without processing images. The bounding box is west, south, east and north
in longitude and latitude degrees. Use `--source LON LAT` to trace drainage from an upstream point and list
settlements along that path.

A date without a time is treated as a UTC calendar day; acquisitions on that day are excluded because their
relation to the flood is unknown. If the event time is known, supply it with a UTC offset, for example
`--date 2026-08-26T03:15+05:45`. Acquisitions before that instant are pre-event and those at or after it are
post-event. The OpenStreetMap snapshot must also precede the event.

## Outputs

Files are written under `$TRI_ROOT/<event>/`.

| File | Contents |
|---|---|
| `report/dashboard.html` | Flood extent, settlement access, routes and repair candidates |
| `report/sitrep.md`, `report/sitrep.html` | English situation report |
| `report/sitrep_ne.md`, `report/sitrep_ne.html` | Nepali situation report |
| `report/*.geojson`, `report/*.csv`, `impact/*.gpkg` | Map layers and infrastructure and settlement lists |
| `hazard/hazard.tif` | 0: no flood evidence; 1: possible; 2: likely; 255: unobserved |
| `summary.json` | Acquisition dates, counts, timings, warnings and errors |

The dashboard embeds its map data, but its display libraries load from the internet. The maps estimate
conditions at the satellite acquisition dates; infrastructure damage and passability require field checks.

## Resume or start again

Completed steps are reused when the same event is run again. Reusing an event name with a different area or
date is refused, since it would mix inputs from different runs. Use a new event name, or add `--fresh` to
remove that event's output folder and recompute it. A change to the code alone does not invalidate cached
outputs; `--fresh` is needed to recompute those outputs.

DEM or flood-fusion failure stops the run. If Sentinel-1 processing fails, the run continues with Sentinel-2;
a failed model step leaves change detection available. Check `warnings` and `errors` in `summary.json` for
missing observations and incomplete processing. A later run retries a failed step.

Areas above about 2,500 km² trigger a processing-time warning. Areas above about 12,000 km² require
`--force-large`. Bounding boxes crossing the antimeridian or reaching beyond 84° latitude are unsupported.
