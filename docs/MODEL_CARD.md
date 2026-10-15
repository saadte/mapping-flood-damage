# Water segmentation models

The radar and optical U-Nets predict water from Sentinel-1 and Sentinel-2 images. Kuro Siwo and Sen1Floods11
provide the water labels, while the event reference maps can also contain deposited debris after water has
receded. The water predictions are combined with measured change and terrain restrictions for these event maps.

## Radar models

| Model folder | Encoder | Training data | Kuro Siwo validation flood IoU |
|---|---|---|---|
| `r34_scratch_v2` | ResNet-34 | Kuro Siwo | 0.422 |
| `r50_scratch_v2` | ResNet-50 | Kuro Siwo | 0.442 |
| `r34_s1f11rl_v1` | ResNet-34 | Kuro Siwo and reprocessed Sen1Floods11 | 0.394 |

Each folder contains `best.pt` and `config.json`. The models use U-Net decoders from segmentation-models-pytorch and
random initial weights. Six input channels contain post-event VV and VH, the most recent pre-event VV and VH, and
the preceding VV and VH. Linear backscatter is converted to dB and standardised using training statistics.

Kuro Siwo uses the BlackBench event split: 27 training events, 7 validation events, of which 6 have labels, and
10 test events. Sen1Floods11 radar chips are reprocessed from GRD on their own chip grids. Water present in the
pre-event images is assigned to permanent water using the dataset's per-event VH threshold, while water occurring
only after the event is assigned to flood. The Sen1Floods11 validation split is drawn by chip rather than by event.
Radar chips for its test and Bolivia splits were not built.

Training uses weighted cross-entropy with class weights 0.5 for dry ground, 1.0 for permanent water and 2.0 for flood,
plus Dice loss on the water classes. AdamW uses learning rate 1e-3 and weight decay 1e-4, with warm-up followed by
cosine decay. The recorded runs use 30 epochs of 800 steps and batch size 32; 70% of sampled tiles contain water.
Rotations, reflections, gain variation, speckle, blur and changes to the pre-event input channels are applied during
training. Checkpoints are selected on validation flood IoU plus half the permanent-water IoU; the additional
Sen1Floods11 validation score is included for the model using that dataset.

At prediction time, a 7 × 7 Lee filter is applied and batch-normalisation statistics are estimated from scene tiles
with HAND < 80 m. Model probabilities are averaged, and the maximum across orbit tracks is retained. GPU predictions
also average eight rotations and reflections. CPU runs omit this augmentation, which means the two paths can give
different probabilities. Inference runs on a 10 m grid, while Kuro Siwo pixels in EPSG:3857 are approximately 7–9 m.

## Kuro Siwo test exposure

Four early variants were compared on the test split during development. Two of them remain in the radar ensemble.
The later choices used validation and development-event scores, but the test results below were already observed
and are reported with that exposure.

| Variant | Flood F1 | Mean IoU | Used in the ensemble |
|---|---|---|---|
| `r34_imnet_v1` | 0.780 | 0.745 | No |
| `r34_scratch_v1` | 0.787 | 0.742 | No |
| `r34_scratch_v2` | 0.789 | 0.752 | Yes |
| `r50_scratch_v2` | 0.788 | 0.750 | Yes |

The early comparison informed the choice between random and ImageNet initialisation. The validation scores gave the
same ordering. Neither the third radar member nor the ensemble has a later Kuro Siwo test score. The stand-alone
radar maps on the three locked events have mean F1 0.115 over the reference extent and 0.190 below HAND 40 m.

## Optical model

The optical model in `s2ai_r34_v1` is a U-Net with a ResNet-34 encoder initialised from ImageNet. Its nine channels
are B2, B3, B4, B8, B11 and B12, and MNDWI, NDWI and NDVI. Dark-object subtraction is applied to visible bands,
and the channels are standardised using training statistics. Sen1Floods11 provides 252 hand-labelled training
chips. Validation selected epoch 32, with water IoU 0.811.

Rotations, reflections, a 20 m blur and smooth illumination variation are applied during training. The blur
represents optical composites sampled at 20 m and resampled to the 10 m analysis grid. Predictions use overlapping
512-pixel tiles in the valley area. Water appearing after the flood requires probability above 0.5 in the post-event
image and below 0.3 in the pre-event composite. Both the first clear and later composites are predicted.

The test and Bolivia splits were evaluated once using the selected checkpoint. Bolivia is absent from the training
split.

| Split | Water IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| Sen1Floods11 test | 0.833 | 0.909 | 0.937 | 0.883 |
| Bolivia | 0.796 | 0.886 | 0.903 | 0.870 |

The recorded metrics are in `results/evaluation/s2ai_r34_v1_final_metrics.json`. Adding optical new water to the
combined flood map increased mean F1 on the 12 development events by 0.0066, with a paired 90% interval from
0.0001 to 0.0138. On large plains the optical new-water mask is combined with the radar water prediction.

## Event selection and overlap

Model weights were not trained on the development or locked flood events. However, the radar ensemble, adaptation
settings and flood thresholds were selected on development-event scores, including Trishuli. The reported development scores include that selection. The locked Thame run provides a Himalayan event that was not used for those choices.

Ahr 2021 shares a storm with the Kuro Siwo Belgium training event and Netherlands validation event, approximately
85–120 km away. The basins do not share labelled pixels, though a radar pass can cover both. Kuro Siwo's event
labelled Nepal is in the Bihar plain and belongs to its test split.

## Limitations

Rough debris can increase radar backscatter even where flood water has reduced it earlier. The radar models
have no debris class, and water predictions alone miss much of the deposited material in mountain reference maps.
Radar and optical change provide the debris detections, with drainage and nearby measured change restricting
false detections from radar geometry and seasonal change on slopes and in urban valleys.

The optical model requires clear imagery. Flood water and newly flooded fields can have similar spectral responses,
and dry debris has no open-water label. GPU runs average rotations and reflections that CPU runs omit, so their probabilities can differ.
Scene-adaptation settings were selected on development events.

## Data credits

Kuro Siwo: Bountos et al. (2024), NeurIPS Datasets and Benchmarks. Sen1Floods11: Bonafilia et al. (2020), CVPR
Workshops. The optical encoder uses ImageNet initialisation from segmentation-models-pytorch. The Sen1Floods11
radar relabelling uses permanent-water information already contained in the supplied dataset labels.
