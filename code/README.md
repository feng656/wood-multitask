# Wood multi-task final data engineering

This project is the finalized, leakage-audited data layer for annual-ring and two-class wood-defect research. All generated files stay below `wood-multitask-final`; sibling datasets are treated as read-only inputs.

## Locked label contract

- Detection and instance segmentation: `knot`, `crack` only.
- Classification: `normal`, `knot`, `crack`.
- `knot_with_crack` maps to `knot` only and never contributes a crack target.
- Marrow, overgrown, Quartzity, Blue_stain and resin are not target classes.
- Non-target regions from datasets that expose them are retained as ignore regions, not normal background.
- Missing task labels are isolated by task masks and never interpreted as negative labels.

## Final inventory

- Unified manifest: 36,747 images (21,346 primary and 15,401 auxiliary).
- Ring targets: 2,826 images and 14,130 target PNG files.
- Extended COCO: 18,676 images, 33,895 knot/crack instances and 674 ignore regions.
- Classification crops: 36,288 indexed crops; 32,095 VSB crops are reused in place.
- Bridge set: 322 images with fixed train=194, val=59 and test=69 physical-group splits.

The fixed split seed is `20260729`. VSB inherits the immutable 10,000/2,000/4,000 split from `vsb-final`.

## Rebuild

```powershell
python scripts/build_manifest.py --config configs/data.json --refresh-image-metadata
python scripts/audit_provenance.py --baseline
python scripts/run_data_pipeline.py --stages bridge rings coco crops duplicates --vsb-segmentation-limit 0
python scripts/audit_provenance.py --verify
python scripts/validate_data_artifacts.py
python -m unittest discover -s tests -v
python scripts/build_final_report.py
```

Use the system Python environment with OpenCV, NumPy, Pillow and Shapely for the data pipeline. The bundled workspace Python can generate DOCX/XLSX reports.

## Entry points

- `configs/defect_taxonomy.json`: auditable label policy.
- `data_processed/manifests/manifest.csv`: source-centered unified catalog.
- `data_processed/integrity_report.json`: complete artifact validation.
- `data_processed/provenance/source_verification.json`: before/after read-only proof.
- `docs/paper_grade_data_engineering_report.md`: final engineering report.
- `docs/bidirectional_validation_protocol.md`: preregistered experiment design.

Data readiness does not prove bidirectional positive transfer. That claim requires the fixed model matrix and group-level statistics defined in the validation protocol.

The semantic stage-3 path applies `ignore_index=255`. Exporters and Mask R-CNN paths that cannot express ignore boxes conservatively exclude affected images; use the extended COCO loader and an ignore-aware evaluator for full OULU reporting.
