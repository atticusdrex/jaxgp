"""Top-level package for jaxgp.

Expose a simple top-level API and package version. The version will be
read from installed metadata when available, otherwise this module falls
back to a default development version.
"""

from importlib.metadata import version, PackageNotFoundError

__all__ = [
	"__version__",
	"gp",
	"kernel",
	"likelihood",
	"mean",
	"mf",
	"optim",
	"util",
]

try:
	# If the package is installed, use the installed distribution version
	__version__ = version("jaxgp")
except PackageNotFoundError:
	# Package not installed (e.g. running from source). Use a sensible default.
	__version__ = "0.0.1"

# Convenience imports for users who want to access submodules from the top-level
from . import gp, kernel, likelihood, mean, mf, optim, util

