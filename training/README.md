# Training

The four U-Nets are trained from Kuro Siwo GRD tiles and Sen1Floods11 hand-labelled chips. Radar training
takes about 2.5 h per model on an NVIDIA L4. Run the commands from the repository root with a CUDA GPU.

| Script | Processing |
|---|---|
| `convert_kuro_siwo.py` | Kuro Siwo labelled GRD shards → chunks of 1,000 tiles: uint8 dB codes `round((dB + 35) × 6)` for post, pre1 and pre2 VV/VH; labels 0 no water, 1 permanent water, 2 flood, 255 ignore |
| `consolidate_kuro_siwo.py` | chunks → `X.npy`, `Y.npy`, `meta.json`, one copy of each tile |
| `build_sen1floods11.py` | Sen1Floods11 hand-labelled train and valid chips → `s1f11_{train,valid}.npz` in the same format, with S1 post and pre images processed by `src/s1_grd.py` |
| `relabel_sen1floods11.py` | → `s1f11_{train,valid}_rl.npz`: flood pixels that were water on both pre dates become permanent water, on one pre date ignored |
| `train_radar.py` | radar U-Net (Kuro Siwo, optionally with Sen1Floods11) |
| `train_optical.py` | optical U-Net on the Sen1Floods11 S2 chips; scores the test and Bolivia splits once at the end |

```
hf download orion-ai-lab/Kuro-Siwo-Webdataset --repo-type dataset --include "train_GRD/*" --local-dir data/kuro_siwo_raw
hf download orion-ai-lab/Kuro-Siwo-Webdataset --repo-type dataset --include "test_GRD/*" --local-dir data/kuro_siwo_raw
python training/convert_kuro_siwo.py --out data/kuro_siwo_chunks data/kuro_siwo_raw/*_GRD/shard-*.tar
python training/consolidate_kuro_siwo.py data/kuro_siwo_chunks data/kuro_siwo
python training/build_sen1floods11.py data/s1f11
python training/relabel_sen1floods11.py data/s1f11
```

| Model | Command |
|---|---|
| `r34_scratch_v2` | `python training/train_radar.py --mm data/kuro_siwo --out models/r34_scratch_v2 --encoder resnet34 --seed 2` |
| `r50_scratch_v2` | `python training/train_radar.py --mm data/kuro_siwo --out models/r50_scratch_v2 --encoder resnet50 --seed 4` |
| `r34_s1f11rl_v1` | `python training/train_radar.py --mm data/kuro_siwo --out models/r34_s1f11rl_v1 --encoder resnet34 --seed 2 --extra data/s1f11/s1f11_train_rl.npz,data/s1f11/s1f11_valid_rl.npz` |
| `s2ai_r34_v1` | `python training/train_optical.py data/s2ai_chips models/s2ai_r34_v1` |

The Kuro Siwo shards require about 170 GB. The optical training script downloads the Sentinel-2 chips
and labels into `data/s2ai_chips`. Each model folder contains `best.pt` and `config.json`, which the mapping
pipeline reads from `models/`.
