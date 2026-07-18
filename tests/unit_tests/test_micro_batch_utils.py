# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from rlinf.utils.nested_dict_process import (
    get_num_micro_batches,
    iter_dict_micro_batches,
)


def _make_batch(batch_size: int):
    return {
        "x": torch.arange(batch_size),
        "nested": {"y": torch.arange(batch_size * 2).reshape(batch_size, 2)},
        "optional": None,
        "metadata": "kept",
    }


def _micro_batch_sizes(batch_size: int, micro_batch_size: int):
    batch = _make_batch(batch_size)
    return [
        micro_batch["x"].shape[0]
        for micro_batch in iter_dict_micro_batches(
            batch, batch_size=batch_size, micro_batch_size=micro_batch_size
        )
    ]


def test_iter_dict_micro_batches_exact_division():
    assert get_num_micro_batches(96, 32) == 3
    assert _micro_batch_sizes(96, 32) == [32, 32, 32]


def test_iter_dict_micro_batches_keeps_remainder_batch():
    batch = _make_batch(100)
    micro_batches = list(
        iter_dict_micro_batches(batch, batch_size=100, micro_batch_size=32)
    )

    assert [micro_batch["x"].shape[0] for micro_batch in micro_batches] == [
        32,
        32,
        32,
        4,
    ]
    assert torch.equal(micro_batches[-1]["x"], torch.tensor([96, 97, 98, 99]))
    assert torch.equal(
        micro_batches[-1]["nested"]["y"], torch.arange(192, 200).reshape(4, 2)
    )
    assert micro_batches[-1]["optional"] is None
    assert micro_batches[-1]["metadata"] == "kept"


def test_iter_dict_micro_batches_small_batch():
    assert get_num_micro_batches(17, 32) == 1
    assert _micro_batch_sizes(17, 32) == [17]


def test_iter_dict_micro_batches_loss_weights_sum_to_one():
    sizes = _micro_batch_sizes(100, 32)
    weights = [size / 100 for size in sizes]

    assert weights == [0.32, 0.32, 0.32, 0.04]
    assert sum(weights) == pytest.approx(1.0)


def test_get_num_micro_batches_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        get_num_micro_batches(-1, 32)
    with pytest.raises(ValueError):
        get_num_micro_batches(32, 0)
