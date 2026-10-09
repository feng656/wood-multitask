from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class DataRecord:
    image_path: str
    dataset_id: str
    sample_id: str
    group_id: str
    image_width: int = 0
    image_height: int = 0
    image_bytes: int = 0
    tree_id: str = ""
    cube_id: str = ""
    face_id: str = ""
    surface_type: str = ""
    split: str = ""
    has_ring: bool = False
    has_pith: bool = False
    has_defect_box: bool = False
    has_defect_mask: bool = False
    has_defect_class: bool = False
    ring_annotation_path: str = ""
    defect_annotation_path: str = ""
    dataset_role: str = ""
    source_split: str = ""
    ring_label_state: str = "missing"
    defect_label_state: str = "missing"
    defect_instance_count: int = 0
    ignored_defect_instance_count: int = 0
    defect_mask_path: str = ""
    pith_annotation_path: str = ""
    ignore_mask_path: str = ""
    raw_defect_classes: str = ""
    unified_category: str = ""
    defect_subtype: str = ""
    attributes: str = ""
    annotation_geometry: str = ""
    supervision_scope: str = "primary"
    auxiliary_task: str = ""
    auxiliary_label: str = ""
    auxiliary_annotation_path: str = ""
    source_variant: str = ""
    base_sample_id: str = ""
    license_id: str = ""
    ring_r2_fold: str = ""
    notes: str = ""

    @classmethod
    def field_names(cls) -> list[str]:
        return [item.name for item in fields(cls)]

    def to_csv_row(self) -> dict[str, object]:
        row = asdict(self)
        for key in (
            "has_ring",
            "has_pith",
            "has_defect_box",
            "has_defect_mask",
            "has_defect_class",
        ):
            row[key] = int(bool(row[key]))
        return row
