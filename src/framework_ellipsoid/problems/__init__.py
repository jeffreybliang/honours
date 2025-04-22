from .axisaligned import AxisAlignedProblem
from .nonaxisaligned import NonAxisAlignedProblem
from .chamfersampled import ChamferSampledProblem
from .chamferboundary import ChamferBoundaryProblem
from .volumeconstrained import VolumeConstrainedProblem

PROBLEM_REGISTRY = {
    "axisaligned": AxisAlignedProblem,
    "nonaxisaligned": NonAxisAlignedProblem,
    "chamfersampled": ChamferSampledProblem,
    "chamferboundary": ChamferBoundaryProblem,
    "volumeconstrained": VolumeConstrainedProblem,
}
