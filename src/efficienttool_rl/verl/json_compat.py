"""JSON compatibility helpers for verl's Tensor-carrying metadata dumps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def to_jsonable(value: Any) -> Any:
    """Convert common tensor/array containers into JSON-serializable values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def patch_ray_trainer_json_dump() -> None:
    """Make verl generation dumps tolerate Tensor-valued reward metadata.

    The patch is intentionally installed by the local runner's Ray task, so
    the shared upstream verl checkout remains untouched.
    """
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer, "_efficienttool_json_dump_patched", False):
        return

    original_dump = RayPPOTrainer._dump_generations

    def safe_dump(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        return original_dump(
            self,
            to_jsonable(inputs),
            to_jsonable(outputs),
            to_jsonable(gts),
            to_jsonable(scores),
            to_jsonable(reward_extra_infos_dict),
            dump_path,
        )

    RayPPOTrainer._dump_generations = safe_dump
    RayPPOTrainer._efficienttool_json_dump_patched = True


def patch_tool_agent_chat_template_defaults() -> None:
    """Propagate configured chat-template kwargs to incremental tool turns.

    verl's native ``ToolAgentLoop`` passes ``apply_chat_template_kwargs`` when
    constructing the initial prompt, but omits them when appending a tool
    response.  Qwen3 therefore silently switches back to thinking mode after
    the first tool call.  Patch only the class used by this project's Ray
    task, leaving the editable upstream checkout unchanged.
    """
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    if getattr(ToolAgentLoop, "_efficienttool_chat_template_patched", False):
        return

    original_init_class = ToolAgentLoop.init_class

    @classmethod
    def patched_init_class(cls, config, tokenizer, processor, **kwargs):
        original_init_class(config=config, tokenizer=tokenizer, processor=processor, **kwargs)
        apply_kwargs = dict(getattr(cls, "apply_chat_template_kwargs", {}))
        if not apply_kwargs or getattr(cls, "_efficienttool_template_bound", False):
            return

        original_apply_chat_template = cls.tokenizer.apply_chat_template

        def apply_chat_template(*args, **call_kwargs):
            merged_kwargs = dict(call_kwargs)
            for key, value in apply_kwargs.items():
                merged_kwargs.setdefault(key, value)
            return original_apply_chat_template(*args, **merged_kwargs)

        cls.tokenizer.apply_chat_template = apply_chat_template
        cls._efficienttool_template_bound = True

    ToolAgentLoop.init_class = patched_init_class
    ToolAgentLoop._efficienttool_chat_template_patched = True
