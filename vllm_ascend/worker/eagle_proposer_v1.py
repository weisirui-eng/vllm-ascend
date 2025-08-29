# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch.nn as nn
from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

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
    ) -> torch.Tensor:
        device = cu_num_tokens.device
        cu_num_tokens = cu_num_tokens.cpu()
        block_table = block_table.cpu()
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = cu_num_tokens[1:] - 1
        target_positions = target_positions.cpu()
        if self.method == "eagle3":
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids[0]

        query_lens = cu_num_tokens[1:] - cu_num_tokens[:-1]
        max_query_len = query_lens.max().item()

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=self.runner.query_start_loc[:batch_size + 1],
            query_start_loc_cpu=self.runner.query_start_loc_cpu[:batch_size +
                                                                1],
            seq_lens_cpu=self.runner.seq_lens_cpu,
            max_query_len=max_query_len,
            num_reqs=batch_size,
            num_actual_tokens=num_tokens,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            block_table_tensor=self.runner.input_batch.block_table[0].
            get_device_tensor(),
            slot_mapping_cpu=target_slot_mapping,
            positions=target_positions,
            attn_mask=self.runner.attn_mask,
            spec_attn_mask=self.runner.spec_attn_mask,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
        )
        # FIXME(woosuk): The below two ops cause synchronization. Optimize.
        attn_metadata = self.runner.attn_metadata_builder.build(
            common_attn_metadata, self.runner.model)
        if self.use_cuda_graph and \
            num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens
        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions.to(device)
        self.hidden_states[:num_tokens] = target_hidden_states
        attn_metadata.block_tables = block_table.to(device)
        with set_ascend_forward_context(attn_metadata,
                                        self.vllm_config,
                                        num_tokens=num_input_tokens):
            last_hidden_states, hidden_states = self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self.positions[:num_input_tokens],
                hidden_states=self.hidden_states[:num_input_tokens],
            )
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)
        draft_token_ids = logits.argmax(dim=-1)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        # Generate the remaining draft tokens.
        draft_token_ids_tensor = torch.zeros(
            (self.num_speculative_tokens, *draft_token_ids.shape),
            dtype=draft_token_ids.dtype)
        draft_token_ids_tensor[0] = draft_token_ids

        positions_cpu = target_positions[last_token_indices].cpu().to(
            torch.int64)
        hidden_states = hidden_states[last_token_indices]
        if self.use_cuda_graph and \
            batch_size <= self.cudagraph_batch_sizes[-1]:
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

            # Compute the slot mapping.
            block_numbers = (clamped_positions_cpu // self.block_size)
            block_ids = block_table.gather(dim=1,
                                           index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            slot_mapping_cpu = (block_ids * self.block_size +
                                clamped_positions_cpu % self.block_size)

            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping_cpu.masked_fill_(exceeds_max_model_len,
                                          PADDING_SLOT_ID)
            # NOTE: ASCEND slot_mapping must on cpu
            attn_metadata.slot_mapping = slot_mapping_cpu.to(
                torch.int32).to(device)
            # copy inputs to buffer for cudagraph
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
            # Run the model.
            with set_ascend_forward_context(attn_metadata,
                                            self.vllm_config,
                                            num_tokens=input_batch_size):

                last_hidden_states, hidden_states = self.model(
                    input_ids=self.input_ids[:input_batch_size],
                    positions=self.positions[:input_batch_size],
                    hidden_states=self.hidden_states[:input_batch_size],
                )
            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size],
                                               None)

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
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cu_target_query_lens: [0, a, a + b, a + b + c]
        # num_rejected_tokens: [n1, n2, n3]
        # num_tokens_per_req: [a - n1, b - n2, c - n3]
        # cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
        # token_indices: [0, 1, ..., a - n1 - 1,
        #                 a, a + 1, ..., a + b - n2 - 1,
        #                 a + b, a + b + 1, ..., a + b + c - n3 - 1]

        # [0, a, a + b, a + b + c] -> [a, b, c]
        query_len_per_req = (cu_target_query_lens[1:] -
                             cu_target_query_lens[:-1])
        # [a, b, c] -> [a - n1, b - n2, c - n3]
        num_tokens_per_req = query_len_per_req - num_rejected_tokens

        # [a - n1, b - n2, c - n3] ->
        # [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
        cu_num_tokens = torch.zeros_like(cu_target_query_lens)
        torch.cumsum(num_tokens_per_req, dim=0, out=cu_num_tokens[1:])
        token_indices = torch.empty(
            num_tokens,
            dtype=torch.int32,
            device=cu_target_query_lens.device,
        )
        BLOCK_SIZE = 1024
        prepare_eagle_input_sequential(
            token_indices,
            cu_target_query_lens,
            cu_num_tokens,
            block_size=BLOCK_SIZE,
        )
        return cu_num_tokens, token_indices

    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())

        self.model = get_model(vllm_config=self.vllm_config,
                               model_config=draft_model_config)

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
            target_attn_layer_names)

        self.attn_layer_names = list(draft_attn_layer_names)
        self.attn_layer_name = next(iter(draft_attn_layer_names))
        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1:
            logger.info(
                "The EAGLE head shares the same vocab embedding" \
                " with the target model."
            )
            self.model.model.embed_tokens = target_model.model.embed_tokens
        else:
            logger.info(
                "Since PP > 1, the EAGLE head loaded its own vocab embedding" \
                " weights instead of sharing them with the target model."
            )

        # share lm_head with the target model if needed
        # some model definition do not define lm_head explicitly
        # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
        if self.vllm_config.speculative_config.method != "eagle3" and \
                hasattr(target_model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            if supports_multimodal(target_model):
                self.model.lm_head = target_model.get_language_model().lm_head
            else:
                self.model.lm_head = target_model.lm_head

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
    ) -> None:
        with set_ascend_forward_context(None,
                                        self.vllm_config,
                                        num_tokens=num_tokens):
            self.model(
                input_ids=self.input_ids[:num_tokens],
                positions=self.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
            )


def prepare_eagle_input_sequential(out_tensor: torch.Tensor,
                                   cu_query_lens: torch.Tensor,
                                   cu_num_tokens: torch.Tensor,
                                   block_size: int):
    num_programs = len(cu_num_tokens) - 1
    for pid in range(num_programs):
        start_pos = cu_num_tokens[pid].item()
        end_pos = cu_num_tokens[pid + 1].item()
        num_tokens = end_pos - start_pos
        index_start = cu_query_lens[pid].item()
        num_blocks = int(
            torch.ceil(torch.tensor(num_tokens / block_size)).item())

        for i in range(num_blocks):
            offset_tensor = torch.arange(0,
                                         block_size,
                                         dtype=torch.int32,
                                         device=out_tensor.device)
            global_start_offset = i * block_size
            target_indices = torch.tensor(
                start_pos + global_start_offset,
                dtype=torch.int32,
                device=out_tensor.device) + offset_tensor
            values_to_store = torch.tensor(
                index_start, dtype=torch.int32,
                device=out_tensor.device) + offset_tensor
            mask = (target_indices >= start_pos) & \
                   (target_indices < end_pos) & \
                   (offset_tensor < num_tokens)
            out_tensor[target_indices[mask]] = values_to_store[mask]
