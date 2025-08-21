# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch.nn as nn
import torchair
from torchair import patch_for_hcom
import vllm.envs as envs_vllm
from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, VllmConfig, get_layers_from_vllm_config,
                         set_current_vllm_config)
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import (
    process_weights_after_loading, set_default_torch_dtype)

from vllm.forward_context import get_forward_context
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.models.deepseek_mtp import CustomDeepSeekMTP
from vllm_ascend.torchair.utils import TorchairCommonAttentionMetadata
from vllm_ascend.utils import ProfileExecuteDuration

PADDING_SLOT_ID = -1


class EagleProposer:

    def __init__(self,
                 vllm_config: VllmConfig,
                 device: torch.device,
                 runner=None):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.runner = runner
        self.model_config = vllm_config.model_config
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size
        self.num_speculative_tokens = (
            self.speculative_config.num_speculative_tokens)
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)
        self.device = device
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()

        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE and
                               not self.vllm_config.model_config.enforce_eager)
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=device)
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=device)
        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        self.arange = torch.arange(vllm_config.scheduler_config.max_num_seqs +
                                   1,
                                   device=device,
                                   dtype=torch.int32)
        mask_len = os.getenv("PAGED_ATTENTION_MASK_LEN", 10000)
        self.attn_mask_len = min(self.model_config.max_model_len,
                                 int(mask_len))
        self.attn_mask_builder = AttentionMaskBuilder(self.attn_mask_len,
                                                      self.dtype)
        self.torchair_compiled_model = None  # type: ignore
        self.torchair_compiled_models = {}  # type: ignore
        self.torchair_graph_enabled = get_ascend_config(
        ).torchair_graph_config.enabled

    def _make_attention_mask(
        self,
        seq_lens,
        query_lens,
        position,
    ) -> torch.Tensor:
        return self.attn_mask_builder.get_splitfuse_attn_mask(
            seq_lens, query_lens, position, self.dtype, self.device)

    def propose(
            self,
            # [num_tokens]
            target_token_ids: torch.Tensor,
            # [num_tokens]
            target_positions: torch.Tensor,
            # [num_tokens, hidden_size]
            target_hidden_states: torch.Tensor,
            # [num_tokens]
            target_slot_mapping: torch.Tensor,
            # [batch_size]
            next_token_ids: torch.Tensor,
            # [batch_size + 1] starting with 0
            cu_num_tokens: torch.Tensor,
            # [batch_size, max_num_blocks_per_req]
            block_table: torch.Tensor,
            sampling_metadata: SamplingMetadata,
            token_indices=None  # MTP-specific parameter
    ) -> torch.Tensor:
        device = cu_num_tokens.device
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if self.method == "eagle3":
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size
            cu_num_tokens = cu_num_tokens.cpu()
            block_table = block_table.cpu()
            target_positions = target_positions.cpu()

        last_token_indices = cu_num_tokens[1:] - 1

        if token_indices is not None and self.torchair_graph_enabled:
            last_token_indices = token_indices

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        query_lens = cu_num_tokens[1:] - cu_num_tokens[:-1]
        max_query_len = query_lens.max().item()

        is_running_torchair = self.torchair_graph_enabled and \
                              not self.runner.with_prefill
        if self.method == "eagle3" and self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        elif is_running_torchair:
            num_input_tokens = self.runner.graph_pad_size  # MTP Torchair
        else:
            num_input_tokens = num_tokens

        if self.method == "eagle3":
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.runner.query_start_loc[:batch_size + 1],
                query_start_loc_cpu=self.runner.query_start_loc_cpu[:batch_size + 1],
                seq_lens_cpu=self.runner.seq_lens_cpu,
                max_query_len=max_query_len,
                num_reqs=batch_size,
                num_actual_tokens=num_tokens,
                actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                block_table_tensor=self.runner.input_batch.block_table[0].get_device_tensor(),
                slot_mapping_cpu=target_slot_mapping,
                positions=target_positions,
                attn_mask=self.runner.attn_mask,
                spec_attn_mask=self.runner.spec_attn_mask,
                attn_state=self.runner.attn_state,
                decode_token_per_req=self.runner.decode_token_per_req,
            )
            attn_metadata = self.runner.attn_metadata_builder.build(
                common_attn_metadata, self.runner.model)
        else:
            seq_lens = target_positions[last_token_indices].add(1).int()
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=cu_num_tokens[:batch_size + 1],
                query_start_loc_cpu=cu_num_tokens[:batch_size + 1].cpu(),
                seq_lens_cpu=seq_lens.cpu(),
                num_reqs=batch_size,
                num_actual_tokens=num_tokens,
                max_query_len=max_query_len,
                actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                block_table_tensor=self.runner.input_batch.block_table[0].get_device_tensor(),
                slot_mapping_cpu=target_slot_mapping,
                positions=target_positions,
                attn_mask=self.runner.attn_mask,
                spec_attn_mask=self.runner.spec_attn_mask,
                attn_state=self.runner.attn_state,
                graph_pad_size=self.runner.graph_pad_size,
                decode_token_per_req=self.runner.decode_token_per_req,
            )

            attn_metadata = self.runner.attn_metadata_builder.build(common_attn_metadata, self.runner.get_model())

            if attn_metadata.prefill is not None:
                attn_metadata.prefill.query_lens = query_lens.cpu()
                attn_metadata.prefill.input_positions = target_positions
                attn_metadata.prefill.seq_lens = seq_lens

            if not self.torchair_graph_enabled:
                # torch mode need to update num_tokens_across_dp
                (num_input_tokens, num_tokens_across_dp, with_prefill,
                 _) = self.runner._get_forward_metadata_across_dp_and_pad(
                    num_tokens, self.runner.with_prefill, False)
                attn_metadata.slot_mapping = target_slot_mapping
            else:
                # torchair mode can reuse self.runner.num_tokens_across_dp
                num_tokens_across_dp = self.runner.num_tokens_across_dp
                with_prefill = self.runner.with_prefill

        self.positions[:num_tokens] = target_positions.to(device)
        self.hidden_states[:num_tokens] = target_hidden_states

        if self.method == "eagle3":
            attn_metadata.block_tables = block_table.to(device)
            with set_ascend_forward_context(attn_metadata, self.vllm_config, num_tokens=num_input_tokens):
                last_hidden_states, hidden_states = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    hidden_states=self.hidden_states[:num_input_tokens],
                )
        else:
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    with_prefill=with_prefill,
                    num_tokens_across_dp=num_tokens_across_dp,
                    in_profile_run=self.runner.in_profile_run,
                    num_actual_tokens=num_tokens):
                with ProfileExecuteDuration().capture_async('mtp_forward'):
                    model_kwargs = {}
                    model_kwargs = {"attn_metadata": attn_metadata}
                    if self.torchair_graph_enabled:
                        model_kwargs["kv_caches"] = self.runner.kv_caches[-1:]

                    if is_running_torchair:
                        torchair_compiled_model = self._get_torchair_lazy_compiled_model(num_input_tokens)
                        hidden_states = torchair_compiled_model(
                            input_ids=self.input_ids[:num_input_tokens],
                            positions=self.positions[:num_input_tokens],
                            previous_hidden_states=self.hidden_states[:num_input_tokens],
                            inputs_embeds=None,
                            intermediate_tensors=None,
                            spec_step_idx=0,
                            **model_kwargs
                        )
                    else:
                        hidden_states = self.model(
                            input_ids=self.input_ids[:num_input_tokens],
                            positions=self.positions[:num_input_tokens],
                            previous_hidden_states=self.hidden_states[:num_input_tokens],
                            kv_caches=self.runner.kv_caches[-1:] if self.method != "eagle3" else None,
                        )
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)
        draft_token_ids = logits.argmax(dim=-1)
        # [batch_size, 1]
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        if self.method == "eagle3":
            draft_token_ids_tensor = torch.zeros(
                (self.num_speculative_tokens, *draft_token_ids.shape),
                dtype=draft_token_ids.dtype)
            draft_token_ids_tensor[0] = draft_token_ids

            positions_cpu = target_positions[last_token_indices].cpu().to(torch.int64)
            hidden_states = hidden_states[last_token_indices]

            if self.use_cuda_graph and batch_size <= self.cudagraph_batch_sizes[-1]:
                input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size)
            else:
                input_batch_size = batch_size

            attn_metadata.num_actual_tokens = batch_size
            attn_metadata.max_query_len = 1
            attn_metadata.query_start_loc = self.arange[:batch_size + 1]

            if self.num_speculative_tokens > 2:
                raise ValueError("Speculative tokens > 2 are not supported yet.")

            attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
            for now_speculative in range(self.num_speculative_tokens - 1):
                # Update the inputs.
                # cast to int32 is crucial when eagle model is compiled.
                # tensor.argmax() returns int64 by default.
                input_ids = draft_token_ids_tensor[now_speculative].to(device)
                positions_cpu += 1

                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length. Since it is complex
                # to remove such requests from the batch, we keep them in the batch
                # but adjust the position ids and slot mappings to avoid the
                # out-of-range access during the model execution. The draft tokens
                # generated with this adjustment should be ignored.
                exceeds_max_model_len = positions_cpu >= self.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions_cpu = torch.where(exceeds_max_model_len, 0,
                                                    positions_cpu)
                clamped_positions = clamped_positions_cpu.to(device)

                # TODO: Increment the sequence lengths.

                attn_metadata.seq_lens += 1
                # TODO: Consider max model length.
                # attn_metadata.max_seq_len = min(attn_metadata.max_seq_len,
                #                                 self.max_model_len)
                # For the requests that exceed the max model length, we set the
                # TODO: sequence length to 1 to minimize their overheads in attention.

                block_numbers = (clamped_positions_cpu // self.block_size)
                block_ids = block_table.gather(dim=1, index=block_numbers.view(-1, 1))
                block_ids = block_ids.view(-1)
                slot_mapping_cpu = (block_ids * self.block_size + clamped_positions_cpu % self.block_size)

                slot_mapping_cpu.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
                attn_metadata.slot_mapping = slot_mapping_cpu.to(torch.int32).to(device)

                self.input_ids[:batch_size] = input_ids
                self.positions[:batch_size] = clamped_positions
                self.hidden_states[:batch_size] = hidden_states
                positions = positions_cpu.to(device)

                attn_mask = self._make_attention_mask(
                    seq_lens=attn_metadata.seq_lens,
                    query_lens=attn_metadata.max_query_len,
                    position=positions,
                )
                attn_metadata.attn_mask = attn_mask
                attn_metadata.block_tables = block_table.to(device)

                with set_ascend_forward_context(attn_metadata, self.vllm_config, num_tokens=input_batch_size):
                    last_hidden_states, hidden_states = self.model(
                        input_ids=self.input_ids[:input_batch_size],
                        positions=self.positions[:input_batch_size],
                        hidden_states=self.hidden_states[:input_batch_size],
                    )

                hidden_states = hidden_states[:batch_size]
                logits = self.model.compute_logits(last_hidden_states[:batch_size], None)
                # TODO(wenlong): get more than one token for tree attention
                draft_token_ids = logits.argmax(dim=-1)
                draft_token_ids_tensor[now_speculative + 1] = draft_token_ids.cpu()

            # [batch_size, num_speculative_tokens]
            draft_token_ids = draft_token_ids_tensor.swapaxes(0, 1)
            return draft_token_ids
    @staticmethod
    def prepare_inputs(
            # [batch_size + 1]
            cu_target_query_lens: torch.Tensor,
            # [batch_size]
            num_rejected_tokens: torch.Tensor,
            token_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            slot_mapping: torch.Tensor,
            is_torchair_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor]:

        # [0, a, a + b, a + b + c] -> [a, b, c]
        query_len_per_req = (cu_target_query_lens[1:] - cu_target_query_lens[:-1])
        # [a, b, c] -> [a - n1, b - n2, c - n3]
        num_tokens_per_req = query_len_per_req - num_rejected_tokens

        if is_torchair_graph:

            cu_num_tokens = cu_target_query_lens
            relative_index = query_len_per_req - num_rejected_tokens - 1
            token_indices = cu_num_tokens[:-1] + relative_index

            target_token_ids = token_ids
            target_positions = positions
            target_hidden_states = hidden_states
            target_slot_mapping = slot_mapping
        else:
            cu_num_tokens = torch.zeros_like(cu_target_query_lens)
            torch.cumsum(num_tokens_per_req, dim=0, out=cu_num_tokens[1:])
            cu_num_tokens[0] = 0

            num_tokens = cu_num_tokens[-1].item()
            token_indices = torch.zeros(
                num_tokens,
                dtype=torch.int32,
                device=cu_num_tokens.device,
            )

            BLOCK_SIZE = 1024
            prepare_input_sequential(
                token_indices,
                cu_target_query_lens,
                cu_num_tokens,
                block_size=BLOCK_SIZE,
            )

            target_token_ids = token_ids[token_indices]
            target_positions = positions[token_indices]
            target_hidden_states = hidden_states[token_indices]
            target_slot_mapping = slot_mapping[token_indices]

        return cu_num_tokens, token_indices, target_token_ids, target_positions, target_hidden_states, target_slot_mapping

    def load_model(self, target_model: nn.Module = None) -> None:

        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())

        is_eagle = self.method in ["eagle", "eagle3"]
        if is_eagle:
            draft_model_config = self.vllm_config.speculative_config.draft_model_config
            self.model = get_model(vllm_config=self.vllm_config,
                                   model_config=draft_model_config)

            draft_attn_layer_names = (
                    get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
                    target_attn_layer_names)
            self.attn_layer_names = list(draft_attn_layer_names)
            self.attn_layer_name = next(iter(draft_attn_layer_names))

            if get_pp_group().world_size == 1:
                logger.info("The EAGLE head shares the same vocab embedding with the target model.")
                self.model.model.embed_tokens = target_model.model.embed_tokens
            else:
                logger.info("Since PP > 1, the EAGLE head loaded its own vocab embedding weights.")

            if self.vllm_config.speculative_config.method != "eagle3" and hasattr(target_model, "lm_head"):
                logger.info("Loading EAGLE LM head weights from the target model.")
                if supports_multimodal(target_model):
                    self.model.lm_head = target_model.get_language_model().lm_head
                else:
                    self.model.lm_head = target_model.lm_head
        else:
            loader = get_model_loader(self.vllm_config.load_config)
            draft_model_config = self.vllm_config.speculative_config.draft_model_config
            target_device = self.vllm_config.device_config.device

            with set_default_torch_dtype(draft_model_config.dtype), \
                    set_current_vllm_config(self.vllm_config):
                self.model = CustomDeepSeekMTP(vllm_config=self.vllm_config).to(target_device)
            draft_attn_layer_names = (
                    get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
                    target_attn_layer_names)
            assert len(draft_attn_layer_names) == 1
            self.attn_layer_name = next(iter(draft_attn_layer_names))

            self.model.load_weights(
                loader.get_all_weights(
                    self.vllm_config.speculative_config.draft_model_config,
                    self.model))
            process_weights_after_loading(self.model, draft_model_config,
                                          self.vllm_config.device_config.device)

    def dummy_run(
            self,
            num_tokens: int,
            with_prefill: bool = False,
            skip_attn: bool = False,
            num_reqs: int = 0,
            num_tokens_across_dp=None
    ) -> None:

        if self.method == "eagle3":
            with set_ascend_forward_context(None, self.vllm_config, num_tokens=num_tokens):
                self.model(
                    input_ids=self.input_ids[:num_tokens],
                    positions=self.positions[:num_tokens],
                    hidden_states=self.hidden_states[:num_tokens],
                )
            return

        if not self.torchair_graph_enabled:
            (num_tokens, num_tokens_across_dp, with_prefill, _) = \
                self.runner._get_forward_metadata_across_dp_and_pad(
                    num_tokens, with_prefill, False)

        is_running_torchair = self.torchair_graph_enabled and not with_prefill

        if is_running_torchair:
            skip_attn = False

        if skip_attn:
            attn_metadata = None
        else:
            common_attn_metadata = TorchairCommonAttentionMetadata(
                num_reqs=num_reqs,
                num_actual_tokens=1,
                actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                attn_mask=self.runner.attn_mask,
                spec_attn_mask=self.runner.spec_attn_mask,
                decode_token_per_req=self.runner.decode_token_per_req,
            )
            attn_metadata = self.runner.attn_metadata_builder.build_torchair_graph_dummy(common_attn_metadata)

        input_ids = self.input_ids[:num_tokens]
        positions = self.positions[:num_tokens]
        previous_hidden_states = self.hidden_states[:num_tokens]

        with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                with_prefill=with_prefill,
                num_tokens_across_dp=num_tokens_across_dp,
                in_profile_run=self.runner.in_profile_run,
                num_actual_tokens=0):
            if is_running_torchair:
                assert attn_metadata is not None
                torch._dynamo.mark_static(input_ids)
                torch._dynamo.mark_static(positions)
                torch._dynamo.mark_static(previous_hidden_states)
                torch._dynamo.mark_static(attn_metadata.decode.block_table)
                torch._dynamo.mark_static(attn_metadata.decode.input_positions)
                if hasattr(attn_metadata.decode, "sin"):
                    torch._dynamo.mark_static(attn_metadata.decode.sin)
                    torch._dynamo.mark_static(attn_metadata.decode.cos)
                torch._dynamo.mark_static(get_forward_context().mc2_mask)
                torch._dynamo.mark_static(attn_metadata.slot_mapping)
                torch._dynamo.mark_static(attn_metadata.decode.attn_mask)
                torchair_compiled_model = self._get_torchair_lazy_compiled_model(num_tokens)
                torchair_compiled_model(
                    input_ids=input_ids,
                    positions=positions,
                    previous_hidden_states=previous_hidden_states,
                    inputs_embeds=None,
                    intermediate_tensors=None,
                    attn_metadata=attn_metadata,
                    kv_caches=self.runner.kv_caches[-1:],
                    spec_step_idx=0)
            else:
                self.model(
                    input_ids=input_ids,
                    positions=positions,
                    previous_hidden_states=previous_hidden_states
                )

    def _get_torchair_lazy_compiled_model(self, batch_size: int):
        if batch_size < 0 or batch_size > self.runner.torchair_graph_batch_sizes[
                -1]:
            raise ValueError(
                f"Bad graph batch size:{batch_size}! max_graph_batch_sizes:{self.runner.torchair_graph_batch_sizes[-1]}"
            )

        compiled_model = self.torchair_compiled_models.get(
            batch_size
        ) if self.runner.use_cached_npu_graph else self.torchair_compiled_model

        if compiled_model:
            return compiled_model

        patch_for_hcom()
        config = torchair.CompilerConfig()
        config.experimental_config.frozen_parameter = True
        config.experimental_config.tiling_schedule_optimize = True
        config.experimental_config.enable_view_optimize = \
        get_ascend_config().torchair_graph_config.enable_view_optimize
        torch.npu.set_compile_mode(jit_compile=False)
        if not self.runner.use_cached_npu_graph:
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            self.torchair_compiled_model = torch.compile(
                self.model,
                dynamic=True,
                fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=npu_backend)
            return self.torchair_compiled_model
        else:
            # Generate a new forward proxy code object to prevent the invalidation of
            # compilation cache caused by dynamo retracing
            forward_proxy_name = f"{self.model.__class__.__name__}_forward_with_batch_size_{batch_size}"
            forward_fn = self.model.forward
            code = forward_fn.__code__
            # Mark code object with a new proxy name
            modified_code = code.replace(co_name=forward_proxy_name, )

            modified_func = types.FunctionType(modified_code,
                                               forward_fn.__globals__,
                                               name=forward_proxy_name,
                                               argdefs=forward_fn.__defaults__)

            self.model.__dict__[forward_proxy_name] = modified_func.__get__(
                self.model, nn.Module)
            self.torchair_compiled_models[
                batch_size] = torchair.inference.cache_compile(
                    self.model.__dict__[forward_proxy_name],
                    dynamic=True,
                    fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                    config=config,
                    ge_cache=False)
            return self.torchair_compiled_models[batch_size]


def prepare_input_sequential(out_tensor: torch.Tensor,
                                   cu_query_lens: torch.Tensor,
                                   cu_num_tokens: torch.Tensor,
                                   block_size: int):
    device = cu_query_lens.device
    dtype = out_tensor.dtype

    offsets = torch.arange(block_size, device=device, dtype=dtype)
    start_pos = cu_num_tokens[:-1]
    end_pos = cu_num_tokens[1:]
    num_tokens = end_pos - start_pos

    global_indices = (start_pos.view(-1, 1) + offsets.view(1, -1))
    values = (cu_query_lens[:-1].view(-1, 1) + offsets.view(1, -1))

    mask = (offsets.view(1, -1) < num_tokens.view(-1, 1))

    global_indices_flat = global_indices[mask]
    values_flat = values[mask]
    out_tensor[global_indices_flat] = values_flat
