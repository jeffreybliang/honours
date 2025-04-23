from .axisaligned import AxisAlignedProblem
from .nonaxisaligned import NonAxisAlignedProblem
<<<<<<< HEAD
from .chamfersample import ChamferSampledProblem
=======
from .chamfersampled import ChamferSampledProblem
>>>>>>> main
from .chamferboundary import ChamferBoundaryProblem
from .volumeconstrained import VolumeConstrainedProblem

PROBLEM_REGISTRY = {
    "axisaligned": AxisAlignedProblem,
    "nonaxisaligned": NonAxisAlignedProblem,
<<<<<<< HEAD
    "chamfer_sampled": ChamferSampledProblem,
    "chamfer_boundary": ChamferBoundaryProblem,
    "volume": VolumeConstrainedProblem,
=======
    "chamfersampled": ChamferSampledProblem,
    "chamferboundary": ChamferBoundaryProblem,
    "volumeconstrained": VolumeConstrainedProblem,
>>>>>>> main
}
