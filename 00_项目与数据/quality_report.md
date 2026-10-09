# Data quality report

Status: **ok**

Indexed images: **36747**

Inventory SHA-256: `df49cf1086400375db1d3acd3710893038fa49b1cff120258f17f304b4fae4e3`

| Dataset | Images | Groups | Split counts | Ring | Pith | Defect boxes | Defect masks | Defect classes | Ignored instances |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| coating | 15075 | 335 | auxiliary_stress=15075 | 0 | 0 | 0 | 0 | 0 | 0 |
| indiana | 1632 | 136 | external_test=1632 | 1466 | 0 | 0 | 0 | 0 | 0 |
| mokume | 1140 | 190 | test=186, train=774, val=180 | 1140 | 0 | 258 | 0 | 258 | 1 |
| oulu | 839 | 839 | external_test=839 | 0 | 0 | 839 | 0 | 839 | 673 |
| urudendro | 64 | 14 | test=15, train=38, val=11 | 64 | 64 | 64 | 0 | 64 | 0 |
| urudendro2 | 54 | 18 | train=42, val=12 | 54 | 54 | 0 | 0 | 0 | 0 |
| urudendro4 | 102 | 21 | external_test=102 | 102 | 102 | 0 | 0 | 0 | 0 |
| vnwoodknot | 1515 | 1515 | external_test=1515 | 0 | 0 | 1015 | 0 | 1515 | 0 |
| vsb | 16000 | 10 | test=4000, train=10000, val=2000 | 0 | 0 | 16000 | 16000 | 16000 | 4574 |
| wvtec_wood | 326 | 326 | aux_train_reference=247, external_anomaly_test=79 | 0 | 0 | 0 | 0 | 0 | 0 |

## Integrity checks

- Duplicate image paths: 0
- Missing referenced files: 0
- Invalid or unreadable images: 0
- Cross-split physical-group leaks: 0
- UruDendro4 R2 fold tree counts: fold 0=5, fold 1=4, fold 2=4, fold 3=4, fold 4=4

## Warnings

- VSB has only 10 conservative acquisition groups. The fixed vsb-final split prevents frame leakage but does not create additional independent wood sources.
- VNWoodKnot numeric class mapping is centralized in configs/defect_taxonomy.json and must be confirmed against the dataset publication before final experiments.
