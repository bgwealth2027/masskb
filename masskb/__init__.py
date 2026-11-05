"""MassKB：面向海量知识库的智能检索中间件。"""

__version__ = "0.1.0"

from masskb.client import (
    MassKB,
    MassKBConnectionError,
    MassKBError,
    MassKBParseError,
    MassKBResponseError,
    MassKBServiceError,
    MassKBValidationError,
)

__all__ = [
    "MassKB",
    "MassKBConnectionError",
    "MassKBError",
    "MassKBParseError",
    "MassKBResponseError",
    "MassKBServiceError",
    "MassKBValidationError",
    "__version__",
]
