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

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from pytest_mock import MockerFixture

from tests.ut.base import PytestBase, TestBase
from vllm_ascend.ops.moe_dispatcher.token_dispatcher import (
    AscendSocVersion, MoEAlltoAllSeqOverLapDispatcher, MoEDispatcherConfig,
    TokenDispatcherWithAll2AllV, TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2)
from vllm_ascend.utils import adapt_patch  # noqa E402


class TestMoEAlltoAllSeqOverLapDispatcher(PytestBase):

    @pytest.fixture
    def config(self):
        config = MoEDispatcherConfig()
        config.set_num_local_experts(2)
        config.set_num_moe_experts(4)
        config.set_moe_pad_expert_input_to_capacity(False)
        config.set_moe_expert_capacity_factor(None)
        config.set_moe_router_topk(2)
        config.set_moe_grouped_gemm(False)
        config.set_group_topk(0)
        config.set_num_groups(1)
        config.set_is_fused(False)
        return config.build()

    def mock_ep_group(self, mocker):
        mock_group = mocker.MagicMock()
        mock_group.rank_in_group = 0
        mock_group.world_size = 2
        mock_group.device_group = "mock_group"
        return mock_group

    @pytest.fixture
    def dispatcher(self, config, mocker: MockerFixture):
        mocker.patch(
            "vllm_ascend.ops.moe_dispatcher.token_dispatcher.get_ep_group",
            return_value=self.mock_ep_group(mocker))
        mocker.patch("torch.npu.current_device", return_value="cpu")
        mocker.patch("torch.npu.Stream", return_value=mocker.MagicMock)
        return MoEAlltoAllSeqOverLapDispatcher(config)

    def test_initialization(self, dispatcher, config):
        assert dispatcher.num_local_experts == config.num_local_experts
        assert dispatcher.num_experts == config.num_moe_experts
        assert dispatcher.local_expert_indices == [0, 1]
        assert dispatcher.ep_rank == 0
        assert dispatcher.ep_size == 2
        assert dispatcher.overlap_stream is not None


class TestTokenDispatcherWithMC2(TestBase):

    def setUp(self):
        self.mc2_group = MagicMock()
        self.mc2_group.device_group.return_value._get_backend.return_value.get_hccl_comm_name.return_value = "hccl_123"
        self.mc2_group.rank_in_group = 0
        self.mc2_group.world_size = 8
        self.mc2_group_patch = patch(
            "vllm_ascend.ops.moe_dispatcher.token_dispatcher.get_mc2_group",
            return_value=self.mc2_group)
        self.mc2_group_patch.start()

        self.rank_group_patch = patch("torch.distributed.get_rank",
                                      return_value=0)
        self.rank_group_patch.start()

        # Mock get_forward_context().mc2_mask
        self.forward_context = MagicMock()
        self.forward_context.mc2_mask = torch.tensor([1, 0, 1])
        self.forward_context_patch = patch(
            "vllm_ascend.ops.moe_dispatcher.token_dispatcher.get_forward_context",
            return_value=self.forward_context)
        self.forward_context_patch.start()

        # Mock get_ascend_soc_version()
        self.ascend_soc_version_patch = patch(
            "vllm_ascend.ops.moe_dispatcher.token_dispatcher.get_ascend_soc_version",
            return_value=AscendSocVersion.A3)
        self.ascend_soc_version_patch.start()

        # Mock get_ascend_config()
        self.ascend_config = MagicMock()
        self.ascend_config.torchair_graph_config.enabled = False
        self.ascend_config_patch = patch(
            "vllm_ascend.ops.moe_dispatcher.token_dispatcher.get_ascend_config",
            return_value=self.ascend_config)
        self.ascend_config_patch.start()

        kwargs = {"with_quant": False, "top_k": 8, "num_experts": 128}
        self.dispatcher = TokenDispatcherWithMC2(**kwargs)

    def tearDown(self):
        self.mc2_group_patch.stop()
        self.forward_context_patch.stop()
        self.ascend_soc_version_patch.stop()
        self.ascend_config_patch.stop()

    def test_init(self):
        # self.assertEqual(self.dispatcher.moe_all_to_all_group_name, "hccl_123")
        self.assertEqual(self.dispatcher.ep_rank_id, 0)
        self.assertEqual(self.dispatcher.ep_world_size, 8)
        self.assertFalse(self.dispatcher.torchair_graph_enabled)
        self.assertFalse(self.dispatcher.with_quant)
        self.assertTrue(self.dispatcher.enable_dispatch_v2)
        self.assertTrue(self.dispatcher.need_extra_args)
        self.assertTrue(self.dispatcher.a3_need_extra_args)

    def test_get_dispatch_mc2_kwargs_without_quant(self):
        hidden_states = torch.randn(10, 128)
        topk_ids = torch.randint(0, 8, (10, 1))
        topk_weights = torch.randn(10, 1)
        expert_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        kwargs = self.dispatcher.get_dispatch_mc2_kwargs(
            hidden_states, topk_weights, topk_ids, expert_map)
        self.assertIn("x", kwargs)
        self.assertIn("expert_ids", kwargs)
        self.assertEqual(kwargs["moe_expert_num"], 8)

    def test_token_permutation_dispatch(self):
        hidden_states = torch.randn(10, 128)
        topk_weights = torch.randn(10, 1)
        topk_ids = torch.randint(0, 8, (10, 1))
        expert_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        with patch("torch_npu.npu_moe_distribute_dispatch_v2",
                   return_value=(torch.randn(10, 128), ) * 5) as mock_dispatch:
            output = self.dispatcher.token_dispatch(hidden_states,
                                                    topk_weights, topk_ids,
                                                    expert_map)
            mock_dispatch.assert_called_once()
            self.assertEqual(output[0], 1)  # group_list_type == 1

    def test_token_dispatch_with_shared_experts_and_quant(self):
        self.shared_experts = MagicMock()
        self.shared_experts.gate_up_proj.return_value = (torch.randn(10, 128),
                                                         torch.tensor(1.0))
        self.shared_experts.act_fn.return_value = torch.randn(10, 128)
        self.dispatcher.with_quant = False
        self.dispatcher.shared_act = torch.randn(10, 128)
        self.dispatcher.swiglu_out_scale = torch.tensor(1.0)
        self.hidden_states = torch.randn(10, 128)
        self.topk_weights = torch.randn(10, 1)

        with patch("torch_npu.npu_moe_distribute_dispatch_v2",
                   return_value=(torch.randn(10, 128), ) * 5):
            with patch(
                    "vllm_ascend.ops.moe_dispatcher.token_dispatcher.npu_stream_switch",
                    autospec=True):
                with patch(
                        "vllm_ascend.ops.moe_dispatcher.token_dispatcher.npu_wait_tensor",
                        autospec=True) as mock_wait:
                    self.dispatcher.token_dispatch(
                        self.hidden_states,
                        self.topk_weights,
                        torch.randint(0, 8, (10, 1)),
                        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                        shared_experts=self.shared_experts)
                    mock_wait.assert_any_call(self.hidden_states,
                                              self.topk_weights)

    def test_get_combine_mc_kwargs_with_quant(self):
        self.dispatcher.with_quant = True
        hidden_states = torch.randn(10, 128)
        self.dispatcher.topk_ids = torch.randint(0, 8, (10, 1))
        self.dispatcher.topk_weights = torch.randint(0, 8, (10, 1))
        self.dispatcher.expert_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        self.dispatcher.ep_recv_counts = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        self.dispatcher.need_extra_args = True
        self.dispatcher.enable_dispatch_v2 = True
        self.dispatcher.output = torch.randint(0, 8, (10, 1))

        kwargs = self.dispatcher.get_combine_mc_kwargs(hidden_states)
        self.assertIn("tp_send_counts", kwargs)

    def test_token_combine_with_shared_experts(self):
        self.dispatcher.shared_experts = MagicMock()
        self.dispatcher.shared_experts.down_proj.return_value = (torch.randn(
            10, 128), torch.tensor(1.0))
        self.dispatcher.shared_act = torch.randn(10, 128)
        self.dispatcher.with_quant = True
        self.dispatcher.topk_ids = torch.randint(0, 8, (10, 1))
        self.dispatcher.topk_weights = torch.randint(0, 8, (10, 1))
        self.dispatcher.expert_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        self.dispatcher.ep_recv_counts = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        self.dispatcher.need_extra_args = True
        self.dispatcher.enable_dispatch_v2 = True
        self.dispatcher.swiglu_out_scale = torch.randint(0, 8, (10, 1))
        self.dispatcher.output = torch.randint(0, 8, (10, 1))
        self.hidden_states = torch.randn(10, 128)

        with patch("torch_npu.npu_moe_distribute_combine_v2",
                   return_value=torch.randn(10, 128)):
            with patch(
                    "vllm_ascend.ops.moe_dispatcher.token_dispatcher.npu_stream_switch",
                    autospec=True):
                with patch(
                        "vllm_ascend.ops.moe_dispatcher.token_dispatcher.npu_wait_tensor",
                        autospec=True):
                    self.dispatcher.token_combine(self.hidden_states)


class TestTokenDispatcherWithAllGather(TestBase):

    def setUp(self):
        # Mock dependencies
        kwargs = {
            "apply_router_weight_on_input": False,
            "top_k": 2,
            "max_num_tokens": 100,
            "ep_size": 2,
            "num_experts": 128,
            "with_quant": False,
        }
        self.dispatcher = TokenDispatcherWithAllGather(**kwargs)

        # Mock NPU functions
        self.patcher_moe_init_routing = patch('torch_npu.npu_moe_init_routing')
        self.mock_moe_init_routing = self.patcher_moe_init_routing.start()
        self.mock_moe_init_routing.return_value = (
            torch.randn(6, 128),  # sorted_hidden_states
            torch.tensor([0, 1, 2, 3, 4, 5]),  # expanded_row_idx
            torch.tensor([0, 1, 0, 1, 0, 1])  # expanded_expert_idx
        )

        self.patcher_moe_compute_expert_tokens = patch(
            'torch_npu.npu_moe_compute_expert_tokens')
        self.mock_moe_compute_expert_tokens = self.patcher_moe_compute_expert_tokens.start(
        )
        self.mock_moe_compute_expert_tokens.return_value = torch.tensor(
            [3, 3])  # expert_tokens

        self.patcher_moe_finalize_routing = patch(
            'torch_npu.npu_moe_finalize_routing')
        self.mock_moe_finalize_routing = self.patcher_moe_finalize_routing.start(
        )
        self.mock_moe_finalize_routing.return_value = torch.randn(3, 128)

    def tearDown(self):
        self.patcher_moe_init_routing.stop()
        self.patcher_moe_compute_expert_tokens.stop()
        self.patcher_moe_finalize_routing.stop()

    def test_token_dispatch_without_expert_map(self):
        hidden_states = torch.randn(3, 128)
        topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]])

        group_list_type, sorted_hidden_states, expert_tokens = self.dispatcher.token_dispatch(
            hidden_states, topk_weights, topk_ids, None)

        # Verify npu_moe_init_routing is called
        self.mock_moe_init_routing.assert_called_once()
        args, kwargs = self.mock_moe_init_routing.call_args

        self.assertEqual(group_list_type, 0)

    def test_token_dispatch_with_quant(self):
        kwargs = {
            "apply_router_weight_on_input": False,
            "top_k": 2,
            "max_num_tokens": 100,
            "ep_size": 2,
            "num_experts": 128,
            "with_quant": True,
        }
        self.dispatcher_quant = TokenDispatcherWithAllGather(**kwargs)

        hidden_states = torch.randn(3, 128)
        topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]])

        group_list_type, sorted_hidden_states, expert_tokens = self.dispatcher_quant.token_dispatch(
            hidden_states, topk_weights, topk_ids, None)

        # Verify quant mode returns group_list_type=1
        self.assertEqual(group_list_type, 0)

    def test_token_combine_with_expert_map(self):
        self.dispatcher.expert_map = torch.tensor([0, 1, 2, 3])
        self.dispatcher.sorted_token_indices = torch.tensor([0, 1, 1, 1, 1, 1])
        self.dispatcher.sorted_weights = torch.tensor(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.dispatcher.original_shape = (3, 128)
        self.dispatcher.mask = torch.tensor([0, 1, 1, 0])
        hidden_states = torch.randn(6, 128)

        final_hidden_states = self.dispatcher.token_combine(hidden_states)

        # Verify index_add_ is applied correctly
        self.assertEqual(final_hidden_states.shape, (3, 128))

    def test_token_combine_without_expert_map(self):
        self.dispatcher.with_quant = False
        self.dispatcher.expanded_row_idx = torch.tensor([0, 1, 1, 1, 1, 1])
        self.dispatcher.topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]])
        self.dispatcher.sorted_token_indices = torch.tensor([0, 1, 1, 1, 1, 1])
        self.dispatcher.sorted_weights = torch.tensor(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.dispatcher.original_shape = (3, 128)
        self.dispatcher.mask = torch.tensor([0, 1, 1, 0])
        hidden_states = torch.randn(6, 128)

        final_hidden_states = self.dispatcher.token_combine(hidden_states)

        # Verify npu_moe_finalize_routing is called
        self.mock_moe_finalize_routing.assert_called_once()
        args, kwargs = self.mock_moe_finalize_routing.call_args

        self.assertEqual(final_hidden_states.shape, (3, 128))

    def test_token_dispatch_with_router_weight(self):
        self.dispatcher.apply_router_weight_on_input = True
        hidden_states = torch.randn(3, 128)
        topk_weights = torch.tensor([[0.7], [0.6], [0.5]])  # topk=1
        topk_ids = torch.tensor([[0], [1], [2]])

        group_list_type, sorted_hidden_states, expert_tokens = self.dispatcher.token_dispatch(
            hidden_states, topk_weights, topk_ids, None)
        self.assertEqual(sorted_hidden_states.shape, (6, 128))

    def test_token_dispatch_invalid_topk_when_router_weight(self):
        self.dispatcher.apply_router_weight_on_input = True
        hidden_states = torch.randn(3, 128)
        topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])

        with self.assertRaises(AssertionError):
            self.dispatcher.token_dispatch(
                hidden_states, topk_weights,
                torch.tensor([[0, 1], [1, 2], [2, 3]]), None)


class TestTokenDispatcherWithAll2AllV(TestBase):

    def setUp(self):
        # Patch properties
        patcher1 = patch.object(TokenDispatcherWithAll2AllV,
                                'ep_group',
                                new_callable=PropertyMock,
                                return_value=MagicMock())
        patcher2 = patch.object(TokenDispatcherWithAll2AllV,
                                'ep_rank',
                                new_callable=PropertyMock,
                                return_value=0)
        patcher3 = patch.object(TokenDispatcherWithAll2AllV,
                                'ep_size',
                                new_callable=PropertyMock,
                                return_value=2)

        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.addCleanup(patcher3.stop)

        self.mock_ep_group_prop = patcher1.start()
        self.mock_ep_rank_prop = patcher2.start()
        self.mock_ep_size_prop = patcher3.start()

        # Mock torch_npu.npu_moe_token_permute
        patcher4 = patch('torch_npu.npu_moe_token_permute')
        self.mock_npu_moe_token_permute = patcher4.start()
        self.addCleanup(patcher4.stop)
        self.mock_npu_moe_token_permute.return_value = (torch.randn(16, 16),
                                                        torch.arange(16))

        # Mock torch_npu.npu_moe_token_unpermute
        patcher5 = patch('torch_npu.npu_moe_token_unpermute')
        self.mock_npu_moe_token_unpermute = patcher5.start()
        self.addCleanup(patcher5.stop)
        self.mock_npu_moe_token_unpermute.return_value = torch.randn(8, 16)

        # Mock async_all_to_all
        patcher6 = patch('vllm_ascend.ops.comm_utils.async_all_to_all')
        self.mock_async_all_to_all = patcher6.start()
        self.addCleanup(patcher6.stop)
        self.mock_async_all_to_all.return_value = (None, torch.randn(16, 16),
                                                   MagicMock())

        # Mock gather_from_sequence_parallel_region
        patcher7 = patch(
            'vllm_ascend.ops.moe_dispatcher.token_dispatcher.gather_from_sequence_parallel_region'
        )
        self.mock_gather_from_sequence_parallel_region = patcher7.start()
        self.addCleanup(patcher7.stop)
        self.mock_gather_from_sequence_parallel_region.return_value = torch.tensor(
            [[2, 2, 2, 2], [2, 2, 2, 2]], dtype=torch.int64)

        # Mock torch.histc
        patcher8 = patch('torch.histc')
        self.mock_histc = patcher8.start()
        self.addCleanup(patcher8.stop)
        self.mock_histc.return_value = torch.tensor([2, 2, 2, 2],
                                                    dtype=torch.int64)

        # Mock torch.npu.current_device
        patcher9 = patch('torch.npu.current_device')
        self.mock_current_device = patcher9.start()
        self.addCleanup(patcher9.stop)
        self.mock_current_device.return_value = 'cpu'

        # Mock torch_npu.npu_dynamic_quant
        patcher10 = patch('torch_npu.npu_dynamic_quant')
        self.mock_npu_dynamic_quant = patcher10.start()
        self.addCleanup(patcher10.stop)
        self.mock_npu_dynamic_quant.return_value = (torch.randn(16, 16),
                                                    torch.randn(16))

        # Mock torch_npu.npu_moe_init_routing_v2
        patcher11 = patch('torch_npu.npu_moe_init_routing_v2')
        self.mock_npu_moe_init_routing_v2 = patcher11.start()
        self.addCleanup(patcher11.stop)
        self.mock_npu_moe_init_routing_v2.return_value = (torch.randn(
            16, 16), torch.arange(16), None, torch.randn(16))

        # Mock torch.repeat_interleave
        patcher12 = patch('torch.repeat_interleave')
        self.mock_repeat_interleave = patcher12.start()
        self.addCleanup(patcher12.stop)
        self.mock_repeat_interleave.return_value = torch.arange(16)

        self.dispatcher = TokenDispatcherWithAll2AllV(top_k=2,
                                                      num_experts=4,
                                                      num_local_experts=2,
                                                      with_quant=False)

    def test_token_dispatch(self):
        hidden_states = torch.randn(8, 16)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2)).long()
        expert_map = torch.tensor([0, 1, 2, 3])

        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [0, 1], dtype=torch.int32)
        self.dispatcher.local_expert_indices = [0, 1]

        result = self.dispatcher.token_dispatch(hidden_states=hidden_states,
                                                topk_weights=topk_weights,
                                                topk_ids=topk_ids,
                                                expert_map=expert_map)

        self.assertIsNotNone(result["hidden_states"])
        self.assertIsNotNone(result["group_list"])
        self.assertEqual(result["group_list_type"], 1)

    def test_token_combine(self):
        self.dispatcher.hidden_shape = (8, 16)
        self.dispatcher.hidden_shape_before_permute = (8, 16)
        self.dispatcher.reversed_local_input_permutation_mapping = torch.arange(
            8)
        self.dispatcher.topk_weights = torch.rand(8, 4)
        self.dispatcher.input_splits = [4, 4]
        self.dispatcher.output_splits = [4, 4]
        self.dispatcher.reversed_global_input_permutation_mapping = torch.arange(
            16)

        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [0, 1], dtype=torch.int32)
        self.dispatcher.local_expert_indices = [0, 1]
        self.dispatcher.num_global_tokens_per_local_expert = torch.tensor(
            [[2, 2], [2, 2]], dtype=torch.int64)

        expert_output = torch.randn(16, 16)
        output = self.dispatcher.token_combine(expert_output)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (8, 16))

    def test_token_dispatch_with_quant(self):
        self.dispatcher = TokenDispatcherWithAll2AllV(top_k=2,
                                                      num_experts=4,
                                                      num_local_experts=2,
                                                      with_quant=True)

        hidden_states = torch.randn(8, 16)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2)).long()
        expert_map = torch.tensor([0, 1, 2, 3])

        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [0, 1], dtype=torch.int32)
        self.dispatcher.local_expert_indices = [0, 1]

        result = self.dispatcher.token_dispatch(hidden_states=hidden_states,
                                                topk_weights=topk_weights,
                                                topk_ids=topk_ids,
                                                expert_map=expert_map)

        self.assertIsNotNone(result["hidden_states"])
        self.assertIsNotNone(result["group_list"])
        self.assertIsNotNone(result["dynamic_scale"])
        self.assertEqual(result["group_list_type"], 1)

    def test_token_dispatch_with_quant_no_active_tokens(self):
        self.dispatcher = TokenDispatcherWithAll2AllV(top_k=2,
                                                      num_experts=4,
                                                      num_local_experts=2,
                                                      with_quant=True)

        self.mock_repeat_interleave.return_value = torch.tensor(
            [], dtype=torch.long)

        hidden_states = torch.randn(8, 16)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2)).long()
        expert_map = torch.tensor([0, 1, 2, 3])

        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [0, 1], dtype=torch.int32)
        self.dispatcher.local_expert_indices = [0, 1]

        result = self.dispatcher.token_dispatch(hidden_states=hidden_states,
                                                topk_weights=topk_weights,
                                                topk_ids=topk_ids,
                                                expert_map=expert_map)

        self.assertIsNotNone(result["hidden_states"])
        self.assertIsNotNone(result["group_list"])
        self.assertIsNotNone(result["dynamic_scale"])
        self.assertEqual(result["group_list_type"], 1)

    def test_token_dispatch_with_log2phy(self):
        hidden_states = torch.randn(8, 16)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2)).long()
        expert_map = torch.tensor([0, 1, 2, 3])
        log2phy = torch.tensor([1, 0, 3, 2])

        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [0, 1], dtype=torch.int32)
        self.dispatcher.local_expert_indices = [0, 1]

        result = self.dispatcher.token_dispatch(hidden_states=hidden_states,
                                                topk_weights=topk_weights,
                                                topk_ids=topk_ids,
                                                expert_map=expert_map,
                                                log2phy=log2phy)

        self.assertIsNotNone(result["hidden_states"])
        self.assertIsNotNone(result["group_list"])
        self.assertEqual(result["group_list_type"], 1)
