# This file initializes the mdcrow package and defines its public interface.
from .agent import MDCrow

# __all__ controls what is exported when a user writes "from mdcrow import *".
# Here, it explicitly exposes the MDCrow class as the primary entry point.
__all__ = ["MDCrow"]
