"""
Maps a detector's native output classes to the Level 1 required taxonomy:
car, LGV, HGV, bus, motorcycle, pedestrian (per the brief).

Detector chosen: `dronefreak/visdrone-yolov8x` (HuggingFace) - VisDrone-pretrained YOLOv8x.
Verified on a real crop from videos/Multi_Road_Merged.MP4: 93 detections across all required
classes (incl. pedestrian and motor, which a generic COCO-pretrained YOLO missed entirely on
the same crop - see PLAN.md). Native classes: pedestrian, people, bicycle, car, van, truck,
tricycle, awning-tricycle, bus, motor, others.

No public pretrained detector natively distinguishes LGV (light goods vehicle, e.g. a
van/transit) from HGV (heavy goods vehicle, rigid/articulated lorry) - that's a UK DfT
traffic-survey distinction, not a VisDrone class. VisDrone's "van"/"truck" split is kept as
LGV/HGV *candidates* below rather than treated as final - refine the boundary using
real-world vehicle length once geo-referencing gives a size reference; a bounding-box-only
heuristic can't reliably separate them without one.

VisDrone also natively detects tricycle/awning-tricycle - relevant here since auto-rickshaws
are common in this footage's setting (Pune, India) but aren't in Level 1's named class list.
Mapped to "other" for now rather than dropped silently - revisit if a level requires them.
"""

VISDRONE_TO_TAXONOMY = {
    "car": "car",
    "van": "LGV_candidate",       # refine via real-world size once geo-referencing exists
    "truck": "HGV_candidate",     # refine via real-world size once geo-referencing exists
    "bus": "bus",
    "motor": "motorcycle",
    "pedestrian": "pedestrian",
    "people": "pedestrian",       # VisDrone's "people" = pedestrians in a dense/seated group
    "bicycle": "other",           # not in the brief's Level 1 list
    "tricycle": "other",          # auto-rickshaw - common here, not in Level 1's list
    "awning-tricycle": "other",
    "others": "other",
}

REQUIRED_CLASSES = ["car", "LGV", "HGV", "bus", "motorcycle", "pedestrian"]


def map_class(native_label: str, mapping: dict[str, str | None] = VISDRONE_TO_TAXONOMY) -> str | None:
    """Maps one detector-native label to a taxonomy class, or None if unmapped/dropped."""
    return mapping.get(native_label.lower())
