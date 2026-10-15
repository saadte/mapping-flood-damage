# Method

Satellite images acquired before and after the flood are processed on a 10 m grid, where measured change and
water predictions are combined to obtain the flood extent. Slope and height above drainage restrict detections
on the valley sides, and the remaining extent is compared with buildings, roads and bridges from pre-event
OpenStreetMap. Road access is calculated using the pre-event network with affected road segments removed,
so a settlement outside the flood extent can still be identified as having lost its road connection.

## Data

| Input | Processing |
|---|---|
| Sentinel-1 GRD, IW VV+VH | Calibration and noise removal from ESA annotation tables; Range-Doppler terrain correction using Copernicus DEM |
| Sentinel-2 L2A | Cloud, cloud-shadow and snow masking; pre-event, first clear post-event and later composites |
| Copernicus DEM GLO-30 | Elevation, slope, D8 drainage, flow accumulation and height above drainage |
| Pre-event OpenStreetMap | Buildings, roads, bridges, places and health facilities from a dated Geofabrik extract or historical Overpass/ohsome query |
| Kuro Siwo and Sen1Floods11 | Training splits for model weights and validation splits for checkpoint selection |
| EMS and UNOSAT maps | Scoring only; excluded from mapping inputs |

Sentinel images and the DEM are obtained through Microsoft Planetary Computer. Each OSM snapshot time is checked
against the event time, including the wider road network used for access. Radar terrain correction uses the EGM2008
geoid grid and ESA orbit and ground-range annotations. Reprocessed Sen1Floods11 radar chips have correlation
0.95–0.98 with the official chips and a recorded bias of +0.3 dB.

## Satellite change

Smooth water often decreases radar backscatter, while rough debris can increase it. Both directions of change
are tested using VV and VH from the same orbit track, with four pre-event dates providing the before observations.
The two-polarisation T² statistic uses moderated variance. Pre-event pairs give its change distribution within
each terrain group, from which the threshold is selected at a nominal 1% false-alarm rate per track. The first
post-event date and the mean of the first two post-event dates are tested with Bonferroni correction; tracks
with fewer than three pre-event acquisitions use the simpler change rule.

Optical debris change requires dNDVI < −0.15 and either dBSI > 0 or increased brightness > 0.02, where pre-event
NDVI is at least 0.2. New water requires positive MNDWI after the flood and negative MNDWI before it. The change
is measured on the first clear image and the later composite. Deposited debris must persist between these views,
while water in the first view is retained because it can recede before the later view. Snow is masked.

The fractions used to select image dates are calculated from Sentinel-2 scene classification (SCL) classes
2, 4, 5, 6 and 7, which are also retained when reflectance is processed. Snow, class 11, is excluded from date
selection because it does not contribute reflectance to the composites. The early composite takes the first
valid observation at each pixel within twelve days of the flood, so different pixels can use different dates.

Ground with pre-event NDVI below 0.3 is also assessed with a detector calibrated on two pre-event composite halves.
It combines MNDWI, NDVI, BSI, NBR2 and spectral-angle changes at a nominal 1% false-alarm rate. This provides a
change observation where vegetation loss cannot be measured. The thresholds are estimated separately for each
scene; the false-alarm rates and remaining mapping constants were selected on the development events.

## Water models

The radar prediction is the mean of three U-Nets with ResNet-34 or ResNet-50 encoders trained from random weights.
Two models use Kuro Siwo; the third also uses reprocessed Sen1Floods11 radar chips. Its water labels are converted
to new water using pre-event VH thresholds from the dataset. Inputs are VV and VH from the post-event image and
two pre-event images, after Lee filtering. Batch-normalisation statistics are estimated from valley tiles with
HAND < 80 m. On GPU, predictions are averaged over rotations and reflections; CPU runs omit this augmentation.
The maximum prediction across orbit tracks is used.

The optical U-Net uses an ImageNet-initialised ResNet-34 encoder. Its nine inputs are B2, B3, B4, B8, B11 and B12,
plus MNDWI, NDWI and NDVI. It is trained on Sen1Floods11 hand-labelled optical chips, with visible-band dark-object
subtraction, blur and illumination variation. New water requires probability > 0.5 after the event and < 0.3 before
it; on large plains, water in either the first clear image or the later composite contributes to the mapped extent.

[Model training and evaluation](MODEL_CARD.md).

## Combining the observations

| Terrain | Likely flood extent |
|---|---|
| Mountain valleys | Radar change below HAND 40 m, optical change below 80 m, and radar water model > 0.8 below 40 m where measured change is nearby; connected detections must touch drainage > 20 km² |
| Flat ground outside large plains | Mean slope < 6° over 310 m; two of radar change, optical change and radar water model > 0.7 agree within 20 m; detections must touch drainage ≥ 5 km² |
| Large low plains | Connected flat ground ≥ 100 km² and HAND < 10 m; radar water model > 0.5 or optical new water |

Radar change and radar water predictions come from the same images and count as one sensor source when road
breaks are assessed. Detections above the height limits are retained where radar and optical change agree, or
where two masks agree and connect to the likely extent. Morphological opening with an 8-pixel radius removes
wide water bodies present both before and after the event; narrow channel water adjoining the extent remains.

Weaker detections, including radar water model > 0.3 and early optical change, form the possible extent near drainage.
Ground without a usable radar or optical view is marked unobserved. The availability mask is retained
when an extent is inferred on this ground.

## Flood level beneath cloud

Cloud-covered valley ground has no optical change measurement. Drainage is divided into 300 m river reaches,
and each reach has a flood level selected from 17 heights between 0 and 60 m above the channel. Radar and optical
detections and observed non-detections are grouped by height above drainage. Their likelihood is combined along
the river tree, with a penalty for differences between neighbouring flood levels, and dynamic programming selects
the levels. Detection and false-alarm rates are fitted by hard expectation-maximisation.

The false-alarm rate is bounded below by the rate measured on ground with HAND ≥ 60 m. A source exceeding 15% on
this ground is excluded from the fit. Confined-valley pixels below the estimated level are added only where
optical observations are absent and the likely map has no detection. These pixels are labelled inferred and do
not establish strong evidence of a road break.

## Infrastructure and road access

Buildings are assessed at their representative point, within a 10 m buffer of the mapped extent. Roads are split
at shared vertices and into segments at most 50 m long. A segment is likely impassable when at least half its
length, or a continuous 50 m length, lies within the likely extent. Possible status requires more than 20% of the
segment within either extent. Bridges are assessed from their approaches, up to 30 m beyond each end.

Road and trail graphs cover the area and a 15 km buffer. Routes to towns, hospitals and main-road exits are
calculated before and after the affected segments are removed. Roads outside the mapped area are assumed open.
Strong road-break evidence is persistent optical change, radar change from two orbits, or radar with optical
change. Separate calculations also include weaker breaks and uncertain bridge failures. Buildings within the
flood extent are counted separately from buildings outside it that have lost road access.

A remaining trail gives walking access to a connected road, with walking time estimated from the elevation model.
Where no mapped trail avoids the extent, the settlement has no mapped ground connection. The normal pre-event route and its
breaks are saved for each affected settlement. Repair candidates are ranked by the buildings reconnected when
the affected road segments reopen.

## Outputs and limits

The dashboard embeds the map data and loads its display libraries from the internet. It includes settlement
search, normal routes, road breaks, repair candidates and CSV export. The English and Nepali reports use the same
counts. A supplied upstream point is traced through D8 drainage, with nearby settlements listed by height above
the path. Water depth and arrival time are not calculated.

Water labels do not represent every debris deposit. Narrow urban flood traces, small mudflows and landslides
outside the river corridor have low scores in the evaluated events. Water already present in pre-event images
can show little additional change, as at Kurigram. Buildings and roads absent from pre-event OSM cannot be
counted, and a calculated road route has not been checked on the ground. Sentinel observations describe
conditions at their acquisition dates and cannot warn of a glacier collapse minutes ahead.

[Results](RESULTS.md) · [Evaluation](EVALUATION_PROTOCOL.md)
