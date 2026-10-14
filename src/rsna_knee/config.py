"""Project-wide constants."""

from pathlib import Path

TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

ID_COLUMN = "StudyInstanceUID"
DEFAULT_MAX_SLICES = 32
DEFAULT_IMAGE_FEATURES = 10


def default_path(name: str) -> Path:
    """Return a path relative to the project root."""
    return Path("data") / name
