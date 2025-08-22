#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/test_offline_inference.py`.
"""
import os
from unittest.mock import patch

import pytest
from modelscope import snapshot_download  # type: ignore
from vllm import SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

from tests.e2e.conftest import VllmRunner

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"
DEEPSEEK_W4A8_MODELS = [
    "vllm-ascend/DeepSeek-V3-W4A8-Pruing",
    "vllm-ascend/DeepSeek-R1-w4a8-pruning"
]


def test_models_distributed_QwQ():
    example_prompts = [
        "Hello, my name is",
    ]
    dtype = "half"
    max_tokens = 5
    with VllmRunner(
            "Qwen/QwQ-32B",
            dtype=dtype,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_models_distributed_DeepSeek_multistream_moe():
    example_prompts = [
        "Hello, my name is",
    ]
    dtype = "half"
    max_tokens = 5
    with VllmRunner(
            "vllm-ascend/DeepSeek-V3-Pruning",
            dtype=dtype,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            additional_config={
                "torchair_graph_config": {
                    "enabled": True,
                    "enable_multistream_moe": True,
                },
                "ascend_scheduler_config": {
                    "enabled": True,
                },
                "refresh": True,
            },
            enforce_eager=False,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.skip(
    reason=
    "deepseek dbo dose not consider the support on half precision float, will enable this ut after we actually support it"
)
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_DBO": "1"})
def test_models_distributed_DeepSeekV3_dbo():
    example_prompts = ["The president of the United States is"] * 41
    dtype = "half"
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
    with VllmRunner(
            "vllm-ascend/DeepSeek-V3-Pruning",
            dtype=dtype,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
    ) as vllm_model:
        model_arch = 'DeepseekV3ForCausalLM'
        registed_models = ModelRegistry.models
        assert registed_models[
            model_arch].module_name == "vllm_ascend.models.deepseek_dbo"
        assert registed_models[
            model_arch].class_name == "CustomDeepseekDBOForCausalLM"
        vllm_model.generate(example_prompts, sampling_params)


def test_models_distributed_pangu():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
            snapshot_download("vllm-ascend/pangu-pro-moe-pruing"),
            max_model_len=8192,
            enforce_eager=True,
            dtype="auto",
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION": "1"})
def test_models_distributed_topk() -> None:
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
        "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
        "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    ]
    dtype = "half"
    sampling_params = SamplingParams(max_tokens=5,
                                     temperature=0.0,
                                     top_k=50,
                                     top_p=0.9)

    with VllmRunner(
            "deepseek-ai/DeepSeek-V2-Lite",
            dtype=dtype,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ": "1"})
def test_models_distributed_alltoallv() -> None:
    example_prompts = [
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
        "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
        "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    ]
    dtype = "half"
    sampling_params = SamplingParams(max_tokens=5,
                                     temperature=0.0,
                                     top_k=50,
                                     top_p=0.9)

    with VllmRunner(
            "deepseek-ai/DeepSeek-V2-Lite",
            dtype=dtype,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


def test_models_distributed_Qwen3_W8A8():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
            snapshot_download("vllm-ascend/Qwen3-8B-W8A8"),
            max_model_len=8192,
            dtype="auto",
            tensor_parallel_size=2,
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_models_distributed_Qwen3_W4A8DYNAMIC():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
            snapshot_download("vllm-ascend/Qwen3-8B-W4A8"),
            max_model_len=8192,
            dtype="auto",
            tensor_parallel_size=2,
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.parametrize("model", DEEPSEEK_W4A8_MODELS)
@patch.dict(os.environ, {"VLLM_ASCEND_MLA_PA": "1"})
def test_models_distributed_DeepSeek_W4A8DYNAMIC(model):
    prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            snapshot_download(model),
            dtype="auto",
            tensor_parallel_size=2,
            quantization="ascend",
            enforce_eager=True,
            enable_expert_parallel=True,
            additional_config={
                "torchair_graph_config": {
                    "enabled": False,
                },
                "ascend_scheduler_config": {
                    "enabled": True,
                }
            },
    ) as vllm_model:
        vllm_model.generate_greedy(prompts, max_tokens)


def test_sp_for_qwen3_moe() -> None:
    example_prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(max_tokens=5,
                                     temperature=0.0,
                                     top_k=50,
                                     top_p=0.9)

    with VllmRunner(
            snapshot_download("Qwen/Qwen3-30B-A3B"),
            dtype="auto",
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            compilation_config={
                "pass_config": {
                    "enable_sequence_parallelism": True
                }
            },
            enable_expert_parallel=True,
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)
