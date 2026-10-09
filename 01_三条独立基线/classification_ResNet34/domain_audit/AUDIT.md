# Classification domain audit

Checkpoint: `D:\教务处实习\wood_preproject\outputs\classification_baseline\runs\resnet34_balanced\best.pt`
Split: `external_test`
Samples: **3055**

## Per-dataset metrics

| Dataset | Samples | Accuracy | Balanced accuracy (supported) | Macro-F1 (supported) |
|---|---:|---:|---:|---:|
| oulu | 1534 | 0.6375 | 0.3865 | 0.3571 |
| vnwoodknot | 1521 | 0.2189 | 0.1733 | 0.2696 |

## Source-class errors

| Dataset | Source class | Samples | Accuracy | Top wrong prediction |
|---|---|---:|---:|---|
| oulu | decayed_knot | 24 | 0.7083 | crack |
| oulu | dry_knot | 304 | 0.7566 | crack |
| oulu | edge_knot | 97 | 0.7010 | crack |
| oulu | encased_knot | 70 | 0.6857 | crack |
| oulu | horn_knot | 40 | 0.3750 | crack |
| oulu | knot_hole | 32 | 0.7812 | crack |
| oulu | leaf_knot | 58 | 0.4828 | crack |
| oulu | moustache_knot | 4 | 0.7500 | crack |
| oulu | small_knot | 329 | 0.7964 | crack |
| oulu | sound_knot | 238 | 0.7647 | crack |
| oulu | split | 235 | 0.4255 | knot |
| oulu | verified_negative | 103 | 0.0000 | knot |
| vnwoodknot | folder_class | 1021 | 0.3066 | crack |
| vnwoodknot | verified_negative | 500 | 0.0400 | crack |

## Interpretation safeguards

- Metrics for a dataset that has no samples of a contract class are also reported over supported classes.
- `normal` crops are verified negatives from the external dataset, not a universal wood-normal distribution.
- Use `misclassifications.csv` to inspect image content and label correctness before changing training data.
