import fcntl
import os
import shutil
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
import torch_npu

try:
    # Recent release of torchair has moved these ops to `.scope`.
    from torchair.scope import npu_stream_switch as _npu_stream_switch
    from torchair.scope import npu_wait_tensor as _npu_wait_tensor
except ImportError:
    from torchair.ops import NpuStreamSwitch as _npu_stream_switch
    from torchair.ops import npu_wait_tensor as _npu_wait_tensor

KV_CACHE_BYTES_CACHE_PATH_NAME = ".kv_cache_bytes"
KV_CACHE_BYTES_CACHE_FILE_NAME = "kv_cache_bytes"
TORCHAIR_CACHE_PATH_NAME = ".torchair_cache"
TORCHAIR_CACHE_DIR = os.getenv(
    'TORCHAIR_CACHE_HOME', os.path.join(os.getcwd(), TORCHAIR_CACHE_PATH_NAME))


@dataclass
class TorchairCommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.
    
    For many of the tensors we keep both GPU and CPU versions.
    """

    num_reqs: int
    """Number of requests"""

    num_actual_tokens: int
    """Total number of tokens in batch"""

    decode_token_per_req: int

    actual_seq_lengths_q: list[int]

    attn_mask: torch.Tensor = None

    spec_attn_mask: torch.Tensor = None

    graph_pad_size: int = -1


@contextmanager
def _file_lock(file_descriptor, lock_type):
    fcntl.flock(file_descriptor, lock_type)
    try:
        yield
    finally:
        fcntl.flock(file_descriptor, fcntl.LOCK_UN)


def _get_torchair_current_work_dir(file_name=None):
    if file_name is None:
        return TORCHAIR_CACHE_DIR
    return os.path.join(TORCHAIR_CACHE_DIR, file_name)


def check_torchair_cache_exist():
    res = False
    torch_air_abs_path = _get_torchair_current_work_dir()
    if os.path.exists(torch_air_abs_path):
        file_list = os.listdir(torch_air_abs_path)
        if len(file_list) != 0:
            res = True
    return res


def check_kv_cache_bytes_cache_exist():
    res = False
    kv_cache_bytes_cache_abs_path = _get_torchair_current_work_dir(
        KV_CACHE_BYTES_CACHE_PATH_NAME)
    if os.path.exists(kv_cache_bytes_cache_abs_path):
        file_list = os.listdir(kv_cache_bytes_cache_abs_path)
        if len(file_list) != 0:
            res = True
    return res


def read_kv_cache_bytes_from_file(rank) -> int:
    kv_cache_bytes = -1
    kv_cache_bytes_cache_abs_path = _get_torchair_current_work_dir(
        KV_CACHE_BYTES_CACHE_PATH_NAME)
    kv_cache_bytes_file = os.path.join(
        kv_cache_bytes_cache_abs_path,
        f"{rank}_{KV_CACHE_BYTES_CACHE_FILE_NAME}")
    with open(kv_cache_bytes_file, "r", encoding="utf-8") as f:
        with _file_lock(f, fcntl.LOCK_SH):
            kv_cache_bytes = int(f.readline())
    return kv_cache_bytes


def write_kv_cache_bytes_to_file(rank, kv_cache_bytes):
    kv_cache_bytes_cache_abs_path = _get_torchair_current_work_dir(
        KV_CACHE_BYTES_CACHE_PATH_NAME)
    os.makedirs(kv_cache_bytes_cache_abs_path, exist_ok=True)
    kv_cache_bytes_file = os.path.join(
        kv_cache_bytes_cache_abs_path,
        f"{rank}_{KV_CACHE_BYTES_CACHE_FILE_NAME}")
    with open(kv_cache_bytes_file, "w", encoding="utf-8") as f:
        with _file_lock(f, fcntl.LOCK_EX):
            f.write(f"{kv_cache_bytes}")


def delete_torchair_cache_file():
    torch_air_abs_path = _get_torchair_current_work_dir()
    if os.path.exists(torch_air_abs_path):
        shutil.rmtree(torch_air_abs_path)


def npu_stream_switch(tag: str, priority: int, *, enabled: bool = True):
    return _npu_stream_switch(tag, priority) if enabled else nullcontext()


def npu_wait_tensor(self: torch.Tensor,
                    dependency: torch.Tensor,
                    *,
                    enabled: bool = True):
    return _npu_wait_tensor(self, dependency) if enabled else self


def converting_weight_acl_format(model, format):
    # currently, there are some operations which do not support ACL_FORMAT_FRACTAL_NZ
    # in eager mode but support it in torchair graph mode. since ACL_FORMAT_FRACTAL_NZ
    # is much more preferred than ACL_FORMAT_FRACTAL_ND on 300I Duo, we add this
    # conversion when using torchair graph mode on 300I Duo platform.
    # TODO: we will remove this conversion if npu_quant_grouped_matmul_dequant
    # accepts weight format of ACL_FORMAT_FRACTAL_NZ in eager mode.
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    for module in model.modules():
        if isinstance(module, FusedMoE):
            if torch_npu.get_npu_format(module.w13_weight.data) == format:
                return
            module.w13_weight.data = torch_npu.npu_format_cast(
                module.w13_weight.data, format)
            module.w2_weight.data = torch_npu.npu_format_cast(
                module.w2_weight.data, format)


def register_torchair_model():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "vllm_ascend.torchair.models.torchair_deepseek_mtp:TorchairDeepSeekMTP"
    )

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "vllm_ascend.torchair.models.torchair_deepseek_v2:TorchairDeepseekV2ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "vllm_ascend.torchair.models.torchair_deepseek_v3:TorchairDeepseekV3ForCausalLM"
    )

    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "vllm_ascend.torchair.models.qwen2:CustomQwen2ForCausalLM")

    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "vllm_ascend.torchair.models.qwen3_moe:CustomQwen3MoeForCausalLM")
