import inspect
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

print("clip_samples" in inspect.signature(DDIMScheduler.__init__).parameters)
    