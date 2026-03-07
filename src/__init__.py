"""RingAug package public API."""

from src.helper import build_parser, build_runtime_config, print_run_summary

__all__ = [
    "IndexPreservingPolygonAugmentor",
    "build_parser",
    "build_runtime_config",
    "print_run_summary",
]


def __getattr__(name: str):
    # Delay heavy CV imports until augmentation engine is actually needed.
    if name == "IndexPreservingPolygonAugmentor":
        from src.augmentor import IndexPreservingPolygonAugmentor

        return IndexPreservingPolygonAugmentor
    raise AttributeError(f"module 'src' has no attribute {name!r}")
