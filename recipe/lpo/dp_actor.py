# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from .core_algos import compute_lpo_loss, compute_lpo_weights_inline, compute_group_pi_weights
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, get_seqlen_balanced_partitions, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.actor import DataParallelPPOActor
__all__ = ["LPODataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class LPODataParallelPPOActor(DataParallelPPOActor):
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()
        lpo_metric = self.config.get("lpo_metric", "token_level_scores")  # the metric name in the data, default to "token_level_scores"
        
        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)

        lpo_listwise_contrastive = self.config.get("lpo_listwise_contrastive", False)
        lpo_projection = self.config.get("lpo_projection", "forward_kl")  # "forward_kl" or "reverse_kl"
        if lpo_projection == "reverse_kl":
            lpo_listwise_contrastive = True

        select_keys = [
            "responses", "input_ids", "attention_mask", "position_ids",
            "old_log_probs",
            "behavior_log_prob",
            "uid_int",
            "token_level_scores",
        ]
        if self.config.get("lpo_ref_lmd", 0.0) > 0:
            select_keys.append("ref_log_prob")
        select_keys.append(lpo_metric)
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss and "ref_log_prob" not in select_keys:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()


        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        _need_collection = True  # always need group integrity for inline weight computation
        _group_K = 0
        if _need_collection and "uid_int" in batch.keys():
            uid_int_all = batch["uid_int"]
            sort_idx = uid_int_all.argsort(stable=True)
            batch = batch[sort_idx]

            _, _group_counts = uid_int_all.unique(return_counts=True)
            _group_K = _group_counts[0].item()
            assert (_group_counts == _group_K).all(), (
                f"Incomplete groups on this DP rank: expected uniform size K={_group_K}, "
                f"got sizes {_group_counts.unique().tolist()}. This indicates the "
                f"batch-balance step split groups across DP ranks; group-level "
                f"operations require intact groups on each rank."
            )
            _mini_bs = self.config.ppo_mini_batch_size
            assert _mini_bs % _group_K == 0, (
                f"ppo_mini_batch_size ({_mini_bs}) must be a multiple of group size "
                f"K={_group_K} so every mini-batch contains intact groups."
            )
            if not self.config.use_dynamic_bsz:
                _micro_bs = self.config.ppo_micro_batch_size_per_gpu
                assert _micro_bs % _group_K == 0, (
                    f"ppo_micro_batch_size_per_gpu ({_micro_bs}) must be a multiple of "
                    f"group size K={_group_K} when group-level operations are active"
                )

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    if _need_collection and _group_K > 0:
                        import torch.distributed as dist
                        _seq_lens = mini_batch["attention_mask"].sum(dim=1)
                        _group_seq_lens = _seq_lens.view(-1, _group_K).sum(dim=1)
                        _n_groups = _group_seq_lens.shape[0]
                        _total = _group_seq_lens.sum().item()
                        _num_micro = min(_n_groups, -(-_total // max_token_len))
                        if dist.is_initialized():
                            _nmt = torch.tensor([_num_micro], device="cuda")
                            dist.all_reduce(_nmt, op=dist.ReduceOp.MAX)
                            _num_micro = _nmt.cpu().item()
                        _group_parts = get_seqlen_balanced_partitions(
                            _group_seq_lens.tolist(), _num_micro, equal_size=False,
                        )
                        micro_batches = []
                        for _gpart in _group_parts:
                            _idxs = []
                            for _g in _gpart:
                                _idxs.extend(range(_g * _group_K, (_g + 1) * _group_K))
                            micro_batches.append(torch.cat([mini_batch[i:i+1] for i in _idxs]))
                    else:
                        micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_idx, data in enumerate(micro_batches):
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    if _need_collection and "uid_int" in data:
                        _mb_uids, _mb_counts = data["uid_int"].unique(return_counts=True)
                        if not (_mb_counts == _group_K).all():
                            logger.warning(
                                "micro-batch %d: %d groups, sizes %s (expected K=%d)",
                                micro_idx, _mb_uids.numel(),
                                _mb_counts.tolist(), _group_K,
                            )

                    old_lp = data["old_log_probs"]
                    behavior_lp = data["behavior_log_prob"]
                    
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    clip_ratio_high = self.config.get("clip_ratio_high", self.config.get("clip_ratio", 0.2))
                    clip_ratio_low  = self.config.get("clip_ratio_low",  clip_ratio_high)

                    with torch.no_grad():
                        _weight_lp = old_lp
                        _sd_rewards_sum = data[lpo_metric].sum(-1)
                        if lpo_metric == "advantages":
                            _sd_rewards_sum = _sd_rewards_sum / response_mask.sum(-1).clamp(min=1)
                        current_sd_weights, _inline_metrics, _target_logits = compute_lpo_weights_inline(
                            current_log_prob=_weight_lp,
                            behavior_log_prob=behavior_lp,
                            ref_log_prob=data.get("ref_log_prob", old_lp),
                            uid_int=data["uid_int"],
                            outcome_rewards=_sd_rewards_sum,
                            raw_rewards=data["token_level_scores"].sum(-1),
                            response_mask=response_mask,
                            trust_lmd=self.config.get("lpo_trust_lmd", 1),
                            ref_lmd=self.config.get("lpo_ref_lmd", 0.0),
                            filter_all_failed=self.config.get("filter_all_failed", True),
                            filter_all_succeeded=self.config.get("filter_all_succeeded", False),
                        )
                        append_to_dict(metrics, _inline_metrics)

                    if lpo_listwise_contrastive:
                        with torch.no_grad():
                            log_is_t = torch.clamp(log_prob.detach() - behavior_lp, -5.0, 5.0)

                            seq_log_is = (log_is_t * response_mask).sum(-1) / (response_mask.sum(-1) + 1e-8)

                            pi_weights = compute_group_pi_weights(seq_log_is, data["uid_int"])
                            _sample_active = (current_sd_weights.abs().sum(-1) > 0)
                            group_active = torch.zeros_like(pi_weights)
                            for _gid in data["uid_int"].unique():
                                _gmask = data["uid_int"] == _gid
                                if _sample_active[_gmask].any():
                                    group_active[_gmask] = 1.0
                            pi_weights = pi_weights * group_active

                        if lpo_projection == "reverse_kl":
                            d = seq_log_is - _target_logits  # (B,) logit gap
                            rev_weight = torch.zeros_like(pi_weights)
                            for _gid in data["uid_int"].unique():
                                _gm = data["uid_int"] == _gid
                                if not group_active[_gm].any():
                                    continue
                                g_pw = pi_weights[_gm]
                                g_d = d[_gm]
                                K_g = _gm.sum().item()
                                g_P = g_pw / K_g
                                d_bar = (g_P * g_d).sum()
                                rev_weight[_gm] = -g_pw * (g_d - d_bar)
                            effective_sd_weights = rev_weight.unsqueeze(-1).expand_as(current_sd_weights)
                        else:
                            effective_sd_weights = (
                                current_sd_weights
                                - pi_weights.unsqueeze(-1).expand_as(current_sd_weights)
                            )
                    else:
                        effective_sd_weights = current_sd_weights

                    pg_loss, mirror_weights, sd_clip_frac = compute_lpo_loss(
                        log_prob=log_prob,
                        weight_mat=effective_sd_weights,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        old_log_prob=old_lp,
                        clip_ratio_low=clip_ratio_low,
                        clip_ratio_high=clip_ratio_high,
                    )
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    with torch.no_grad():
                        ratio = torch.exp(log_prob - old_lp)
                        ppo_kl = verl_F.masked_mean(old_lp - log_prob, response_mask)

                        over_upper  = ratio > 1 + clip_ratio_high
                        under_lower = ratio < 1 - clip_ratio_low
                        w = mirror_weights  # (B, L)
                        # PPO-comparable: clip binding (overshot in good direction, gradient zeroed)
                        pg_clipfrac = verl_F.masked_mean(
                            ((over_upper & (w > 0)) | (under_lower & (w < 0))).float(), response_mask
                        )
                        # Outside trust region but not binding (correction direction, gradient kept)
                        pg_clipfrac_lower = verl_F.masked_mean(
                            ((over_upper & (w < 0)) | (under_lower & (w > 0))).float(), response_mask
                        )

                    data_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/sd_loss": policy_loss.detach().item(),
                        "actor/mirror_weight_max":  mirror_weights.max().detach().item(),
                        "actor/mirror_weight_min":  mirror_weights.min().detach().item(),
                        "actor/mirror_weight_std":  mirror_weights.std().detach().item(),
                        "actor/mirror_weight_mean": mirror_weights.mean().detach().item(),
                        "actor/active_seq_frac": (mirror_weights.abs().sum(-1) > 0).float().mean().detach().item(),
                        "actor/response_lp_mean": verl_F.masked_mean(log_prob, response_mask).detach().item(),
                        "actor/approx_kl": verl_F.masked_mean(log_prob - behavior_lp, response_mask).detach().item(),
                        "actor/is_ratio_abs_mean": verl_F.masked_mean(
                            (log_prob.detach() - behavior_lp).abs(), response_mask
                        ).detach().item(),
                        "actor/ref_kl_mean": verl_F.masked_mean(
                            log_prob.detach() - data["ref_log_prob"], response_mask
                        ).detach().item() if "ref_log_prob" in data else 0.0,
                        "actor/ppo_kl":            ppo_kl.item(),
                        "actor/pg_clipfrac":        pg_clipfrac.item(),
                        "actor/pg_clipfrac_lower":  pg_clipfrac_lower.item(),
                        "actor/sd_clip_frac":       sd_clip_frac.item(),
                        "actor/pi_weight_max": pi_weights.max().detach().item() if lpo_listwise_contrastive else 0.0,
                        "actor/pi_weight_min": pi_weights.min().detach().item() if lpo_listwise_contrastive else 0.0,
                        "actor/pi_weight_std": pi_weights.std().detach().item() if lpo_listwise_contrastive else 0.0,
                    }
                    append_to_dict(metrics, data_metrics)

                grad_norm = self._optimizer_step()

                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)

        self.actor_optimizer.zero_grad()

        return metrics