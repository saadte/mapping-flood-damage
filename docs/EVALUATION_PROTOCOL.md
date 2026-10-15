# Evaluation

Copernicus EMS and UNOSAT provide the reference extents for event evaluation. Reference geometries are excluded
from the mapping inputs and used only for scoring. Development events were used to compare mapping settings,
which means their final scores include selection on those events.

## Event sets

| Set | Events | Use |
|---|---|---|
| Development, 12 | Trishuli 2026, Melamchi 2021, Kaski 2021, Kathmandu 2024, Baghlan 2024, Shovi 2023, Arstanbap 2024, Storm Alex 2020, Slovenia 2023, Jablanica 2024, Ahr 2021, Derna 2023 | Selection of change thresholds, terrain limits, model settings and road-access rules |
| Initial hold-out, 5 | Panjshir 2018, Baghlan-jadid 2024, Nepal Far-West 2020, Nepal Terai 2021, Sunamganj 2022 | First evaluation before selection on these events; three lowland events subsequently informed the mapping of plains |
| Fresh lowland, 5 | Sahuka 2025, Larkana 2022, Kurigram 2017, Sharaqpur 2025, Jamalpur 2020 | Each candidate scored once; Sharaqpur and Jamalpur were not used to design the combined water rule, but their scores were used to accept it |
| Fresh mountain, 2 | Kłodzko 2024, Cantiano 2022 | Each candidate scored once |
| Additional floods, 2 | Tzermiado 2026, San Sebastiano da Po 2025 | Comparison of the specified change in optical date selection |
| Additional landslide | Viksdalen 2024 | EMSR775 AOI11 contains a landslide polygon; reported separately from the additional flood cases |
| Locked, 3 | Thame GLOF 2024, Buner 2025, Charikar 2020 | Raw-data runs scored once; their scores were not used for later selection |
| Qualitative | Bhote Koshi 2025, Sikkim 2023 | Visual or runtime checks without a scored reference comparison |

Fresh lowland references use COSMO-SkyMed for Sahuka, Sentinel-2 for Larkana, TerraSAR-X for Kurigram and Jamalpur,
and ICEYE for Sharaqpur. Kłodzko uses EMSR756 from WorldView-3, and Cantiano uses EMSR634. The locked references
are UNOSAT 3919 for Thame, EMSR838 AOI10/11 for Buner and EMSR459 AOI01 for Charikar.

The additional references are EMSR933 AOI03 from GeoEye-1, EMSR800 AOI01 from Pléiades and EMSR775 AOI11 from
Pléiades. The snow exclusion was specified after Kaski had improved in an earlier comparison that also changed
shadow masking. It was then tested on the same twelve development events with the existing shadow treatment
retained. Bounding boxes and flood dates for the additional areas were specified before their reference
geometries were read, and their scores were used to decide whether to retain the date-selection change.
Viksdalen's observed-event layer was found to contain only a small landslide, so a second flood case was added
at San Sebastiano. The Viksdalen score is retained as a landslide result.

## Scores

Precision, recall, F1 and IoU are calculated within the analysed extent declared by each reference product.
The likely flood extent is scored separately from likely plus possible detections, and the predicted area
outside the reference extent is recorded. Each event has equal weight in an event mean. Runs with Sentinel-2
withheld use the same references and extents to measure the loss when optical observations are unavailable.

Candidate acceptance criteria were recorded before scoring. Comparisons used paired event differences, with
bootstrap intervals where recorded. The observed result for each candidate is retained, including candidates
that did not meet the criteria. A later result does not replace an earlier locked-event measurement.

## Model evaluation

Training uses the published training splits and validation selects checkpoints. Four early radar variants were
compared on the Kuro Siwo test split during development; two remain in the ensemble. These observed test scores
informed the early comparison of initialisation methods, and later selection used validation and development
events. The [model card](MODEL_CARD.md) gives the individual scores.

Sen1Floods11 radar chips were built for training and validation. Its optical test and Bolivia splits were scored
once with the selected optical checkpoint. The stand-alone model scores on locked events are also reported
separately from the combined map.

## Reference and development exposure

Trishuli was used to select settings and is a development case study. Its results are compared with both EMSR927
and the UNOSAT corridor map. The latter shares the 27 August optical acquisition with the mapping inputs.
The Nepal Far-West and Terai references use Sentinel-1 imagery. These comparisons share a sensor or acquisition with the mapping inputs.

Early preprocessing was started on the locked events and stopped before prediction or scoring; no maps were
viewed. Reference file names, sizes and bounds were inspected before the final bounding boxes were selected.
The final locked runs started again from raw satellite data. Charikar was scored with the plain-area rule used
at that time, which was subsequently changed; its first score is retained.

Earlier RTC-based development runs are excluded from the submission results because their input processing
did not meet the selected data requirements.

[Results](RESULTS.md).
