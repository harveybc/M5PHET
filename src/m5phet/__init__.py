"""Minimal typed-ML contracts. No execution or calibration authority."""

from .classification import ContractError, make_classification_result

__version__ = "0.1.0"
__all__ = ["ContractError", "make_classification_result"]
