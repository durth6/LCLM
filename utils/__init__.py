from .checkpointing import *
from .nan_checks import *
from .seed import *


def __getattr__(name):
    """Avoid importing training-only PEFT/FSDP code for inference utilities."""
    if name == "convert_fsdp_to_hf":
        from .convert_fsdp_to_hf import convert_fsdp_to_hf
        return convert_fsdp_to_hf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
