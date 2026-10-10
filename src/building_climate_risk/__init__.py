"""Building-level climate risk (risk = social vulnerability x exposure x hazard).

A hazard-agnostic core (``core``) and hazard modules (``hazards``). The first hazard is
pluvial flooding, the framework of Vogelbacher et al. (2026, NHESS); the top-level names
below keep the v0.1 API.
"""

from .core.assess import assess
from .core.exposure import allocate, residents_on_floor
from .core.response import exceedance, lognormal_cdf
from .core.risk import CLASS_NAMES, iterated_means, risk, risk_class
from .core.vulnerability import flood_sensitivity, sensitivity_transform, social_vulnerability, topsis
from .core.zonal import annulus, clip_to, covered_fraction, rings_without_buildings
from .hazards.pluvial import flooded_fraction, hazard_mobility, hazard_wellbeing, mobility_fractions

__version__ = "0.2.0"
