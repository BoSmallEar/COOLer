"""Vis4D Backbone module."""
from .base import BaseBackbone
from .mmdet import MMDetBackbone

__all__ = [
    "BaseBackbone",
    "MMDetBackbone",
]
