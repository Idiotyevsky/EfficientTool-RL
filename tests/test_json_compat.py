import json

import numpy as np
import torch

from efficienttool_rl.verl.json_compat import (
    patch_tool_agent_chat_template_defaults,
    to_jsonable,
)


def test_to_jsonable_handles_nested_tensor_and_numpy_values():
    value = {
        "tensor": torch.tensor([1.0, 2.0]),
        "array": np.array([[3, 4]], dtype=np.int64),
        "scalar": np.float32(0.5),
    }

    converted = to_jsonable(value)
    assert json.loads(json.dumps(converted)) == {
        "tensor": [1.0, 2.0],
        "array": [[3, 4]],
        "scalar": 0.5,
    }


def test_chat_template_patch_is_exposed_for_the_local_runner():
    assert callable(patch_tool_agent_chat_template_defaults)
