# Tested alternatives

| Change | Evaluation | Result |
|---|---|---|
| Radar models retrained with Himalayan no-event negatives | Three Himalayan events, F1 below HAND 40 m | −0.103, interval −0.155 to −0.050; not used |
| Flood level as the entire mountain extent | Twelve development events | F1 0.47–0.49 against 0.527 for the combined map; not used as a replacement |
| Flood level for cloud-covered valley ground | Twelve development events with Sentinel-2 withheld | Mean F1 increase 0.046, interval 0.023–0.073; retained |
| Optical new water added to the combined map | Twelve development events | Mean F1 increase 0.0066, interval 0.0001–0.0138; retained |
| First hold-out evaluation | Five events before selection on them | Mean F1 0.499 |
| Radar water prediction on large plains | Three lowland events and fourteen other events | Lowland gain 0.091 with optical observations and 0.191 without; retained |
| Removal of optical change on plains | Development events, then Larkana 2022 | Development gain 0.006, followed by a 0.26 loss on Larkana; removed |
| Radar or optical water on plains | Development lowland events, then Sharaqpur and Jamalpur | Development gain 0.014; fresh-event gain 0.0135 with optical observations and 0.052 without; retained |
| Promoting possible detections below the estimated flood level | Fourteen non-plain events | Negative mean change; not used |
| Removing components above the estimated flood level | Fourteen non-plain events and two fresh mountain events | Improved the fresh mountain events, but lowered Arstanbap by 0.080 and Kaski by 0.057; not used |
| River connections at 2 or 5 km², four variants | Twelve development events | Best mean F1 change +0.0022, below the required +0.01; not used |
| Optical water predictions near measured change or rivers | Five development events | Kaski lost 0.022–0.042; not used |
| Masking terrain shadow, with or without a change in date selection | Trishuli, Kaski and Storm Alex | Trishuli lost 0.0073; not used |
| Calibrated vegetation change, two variants | Twelve development events | Mean F1 change −0.0025 and −0.0006; not used |
| Remove vegetation change recurring in the previous year, two variants | Eight events with prior-year imagery; four others unchanged | Mean F1 change +0.0009 and +0.0007; not used |
| Early debris change with radar or model support, two variants | Twelve development events | Mean F1 change −0.0010 and −0.0017; not used |
| Lee Sigma filtering before radar terrain correction | Twelve development events, with and without Sentinel-2 | Mean F1 change −0.0006 and −0.0008; not used |
| First four suitable post-event optical dates | Eleven completed development events | Mean F1 change +0.0007; comparison incomplete, not used |
| Exclude snow from optical date selection | Twelve development events, followed by Tzermiado and San Sebastiano | Mean development F1 change +0.0016; additional flood scores unchanged; retained |

The Trishuli and twelve-event development scores include selection on those events. The first hold-out score was
measured before the three lowland hold-out events informed the settings for plains. Among the five fresh lowland
events, the combined water rule was specified before Sharaqpur and Jamalpur were scored. Their scores were
used to accept it. The locked-event scores were not used for later selection.
