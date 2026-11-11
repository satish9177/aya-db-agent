"""Opt-in Cohere2 supervised-position projection, retaining its native scaled loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F

FULL = "full-logits"
SELECTED = "trained-positions-v1"


def selected_inputs(inputs: dict, *, num_items_in_batch=None):
    """Keep hidden position i iff label[i+1] is trained, with no cross-row shift."""
    if any(k in inputs for k in ("shift_labels", "logits_to_keep")):
        raise ValueError("position selection must start from original unshifted labels")
    labels = inputs["labels"]
    if labels.ndim != 2 or labels.dtype != torch.long:
        raise ValueError("expected batched int64 causal labels")
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
    keep = shifted.ne(-100)  # Must stay bool: integer tensors mean positional indices.
    if not bool(keep.any()) and (num_items_in_batch is None or int(num_items_in_batch) <= 0):
        raise ValueError("no trained next-token targets in this loss window")
    if num_items_in_batch is not None and int(num_items_in_batch) < int(keep.sum()):
        raise ValueError("accumulation denominator is smaller than this microbatch's target count")
    return {**inputs, "logits_to_keep": keep, "shift_labels": shifted[keep]}


class TrainedPositionLossMixin:
    """Use ahead of HF Trainer; all loss reduction/GA scaling remains in native Trainer."""
    loss_policy = FULL

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.loss_policy == SELECTED:
            if not self.model_accepts_loss_kwargs:
                raise ValueError("selected loss requires native num_items_in_batch support")
            if self.label_smoother is not None or self.compute_loss_func is not None:
                raise ValueError("selected loss requires the unmodified native causal loss")
            inputs = selected_inputs(inputs, num_items_in_batch=num_items_in_batch)
        elif self.loss_policy != FULL:
            raise ValueError("unknown registered loss policy")
        # Original labels remain intact for Trainer's full-window count. It forwards the
        # unchanged num_items_in_batch; Cohere2 applies logit_scale and ForCausalLMLoss.
        return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
