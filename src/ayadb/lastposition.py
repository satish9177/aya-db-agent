"""Explicit last-position inference alongside the unchanged historical HF engine."""
from __future__ import annotations

from .engine import HFEngine


class _LastPositionModel:
    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return self.model.device

    def generate(self, *args, **kwargs):
        if kwargs.get("logits_to_keep", 1) != 1:
            raise ValueError("registered inference must project only the last position")
        kwargs["logits_to_keep"] = 1
        return self.model.generate(*args, **kwargs)


class LastPositionHFEngine(HFEngine):
    def __init__(self, model, *args, **kwargs):
        super().__init__(_LastPositionModel(model), *args, **kwargs)

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        self.calls[-1]["logits_to_keep"] = 1
        return result
