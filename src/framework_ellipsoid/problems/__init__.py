from .axisaligned import AxisAlignedProblem
from .nonaxisaligned import NonAxisAlignedProblem
from .chamfersample import ChamferSampledProblem
from .chamferboundary import ChamferBoundaryProblem
from .volumeconstrained import VolumeConstrainedProblem

PROBLEM_REGISTRY = {
    "axisaligned": AxisAlignedProblem,
    "nonaxisaligned": NonAxisAlignedProblem,
    "chamfer_sampled": ChamferSampledProblem,
    "chamfer_boundary": ChamferBoundaryProblem,
    "volume": VolumeConstrainedProblem,
}
