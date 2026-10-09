"""Data indexing, split, and loading utilities for the wood multi-task project."""

from .loaders import WoodCatalog, WoodDataset

try:
    from .multitask import SharedMultitaskBaseline, build_shared_multitask_model
except ImportError:  # Data engineering hosts do not need the training stack.
    SharedMultitaskBaseline = None
    build_shared_multitask_model = None

__version__ = "0.2.0"
