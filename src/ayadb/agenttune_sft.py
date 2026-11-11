"""Supported AgentTune custom SFT factory preserving audited episode targets.

Construction is CPU-only and does not train or download a model. The caller must supply an
already prepared model (including QLoRA, if desired) when an individually approved run begins.
The legacy sft.py entry point and its results remain unchanged.
"""
from __future__ import annotations

from agenttune.agentic.distill_trainer import create_distill_trainer
from agenttune.agentic.events import Event, EventKind, EventLog

from .sft import Collator
from .sqlsftdata import encode_episode


class AuditedTrajectory(EventLog):
    """AgentTune EventLog extension retaining per-turn masks and the exact call format.

    The shipped generic EventLog export reformats calls and loses injected-error flags.
    This explicit format adapter feeds the registered custom factory the audited rows intact.
    """
    def __init__(self, episode: dict):
        super().__init__(tier="full", id=episode["id"])
        self.episode = episode
        self.append(Event(EventKind.OBSERVATION, {"text": episode["prompt"]}))
        for message in episode["messages"]:
            kind = EventKind.TEXT if message["role"] == "assistant" else EventKind.OBSERVATION
            self.append(Event(kind, {"text": message["content"], "train": bool(message.get("train"))}))

    def as_dataset_rows(self, fmt: str) -> list[dict]:
        if fmt != "sft":
            raise ValueError("audited episodes support only SFT")
        return [{**self.episode, "messages": [dict(m) for m in self.episode["messages"]]}]


class AuditedSFTFactory:
    def __init__(self, prepared_model, tokenizer, training_args, *, max_len=4096,
                 trainer_class=None, eval_dataset=None):
        self.prepared_model, self.tokenizer, self.training_args = prepared_model, tokenizer, training_args
        self.max_len, self.trainer_class, self.eval_dataset = max_len, trainer_class, eval_dataset
        self.trainer = None

    def __call__(self, *, model, train_dataset, **kwargs):
        # AgentTune's `model` is the student identifier; use the caller's exact prepared
        # instance so neither tokenizer substitution nor a second PEFT wrapping can occur.
        examples = []
        for episode in train_dataset:
            example = encode_episode(episode, self.tokenizer, self.max_len)
            if example is None:
                raise ValueError("oversized audited episode; refusing truncation")
            if "labels" in episode:
                # Continuation matching may mask a final subset of targets. Validate
                # against a fresh encoding so it can never unmask an observation/error.
                supplied = episode["labels"]
                if episode.get("input_ids") != example["input_ids"] or len(supplied) != len(example["labels"]):
                    raise ValueError("matched-token encoding differs from audited input")
                if any(label != -100 and label != base for label, base in zip(supplied, example["labels"])):
                    raise ValueError("matched-token mask attempts to add an unaudited target")
                example["labels"] = list(supplied)
            examples.append(example)
        cls = self.trainer_class
        if cls is None:
            from transformers import Trainer
            cls = Trainer
        self.trainer = cls(model=self.prepared_model, args=self.training_args,
                           train_dataset=examples, eval_dataset=self.eval_dataset,
                           data_collator=Collator(self.tokenizer.pad_token_id))
        return self.trainer


def prepare_trainer(student: str, episodes: list[dict], trainer_factory, output_dir: str):
    """No model load, teacher call, optimizer step, upload or implicit backend selection."""
    if not episodes:
        raise ValueError("no audited episodes")
    return create_distill_trainer(student, trajectories=[AuditedTrajectory(ep) for ep in episodes],
                                  trainer_factory=trainer_factory, output_dir=output_dir)
