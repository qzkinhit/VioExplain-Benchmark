"""Public interface for constraint-violation explanation."""
from .api import ExplainResult, explain
from .publicexp.features import Violation
from .publicexp.knowledge import Representation

__version__ = "0.1.0"
__all__ = ["ExplainResult", "Representation", "Violation", "explain"]
