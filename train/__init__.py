"""Training package.

The original distributed LCLM trainer has optional Liger/Accelerate
dependencies.  Keep it lazy so the standalone Stage-A selector trainer can
run in an inference-capable BF16 environment without importing that stack.
"""

__all__ = ["LCLMTrainer"]


def __getattr__(name):
    if name == "LCLMTrainer":
        from .trainer import LCLMTrainer
        return LCLMTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
