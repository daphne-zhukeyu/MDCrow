# This file initializes the mdcrow package and exposes its primary components.
from .agent import MDCrow

# The __all__ list defines the public API for the package.
# It ensures that 'from mdcrow import *' only exports the MDCrow class.
__all__ = ["MDCrow"]
