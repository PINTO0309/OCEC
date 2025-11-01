"""OCEC: Open and closed eye classification pipeline."""

from .model import VSDLM
from .pipeline import main

__all__ = ["VSDLM", "main"]
