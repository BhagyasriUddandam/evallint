"""evalcheck — audits LLM eval datasets for flaws that make evaluations lie."""

from .io import LoadError, load
from .schema import EvalCase, EvalSet, SchemaError

__all__ = ["EvalCase", "EvalSet", "LoadError", "SchemaError", "load"]
