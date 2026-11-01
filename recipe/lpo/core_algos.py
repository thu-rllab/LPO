# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import math
from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
import wandb
from verl.trainer.ppo.core_algos import agg_loss


def compute_maxrl_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
):
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            
        N = len(id2score[index[0]]) # number of responses per prompt
            
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)

            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))

            else:
                raise ValueError(f"no score in prompt index: {idx}")
            
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2mean[index[i]] + epsilon)

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_lpo_loss(
    log_prob,           # (B, L)
    weight_mat,         # (B, L)
    response_mask,      # (B, L)
    loss_agg_mode="token-mean",
    old_log_prob=None,  # (B, L) — start-of-step log-probs for M-step trust region (None = no clip)
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
):
    """Weighted MLE loss with optional M-step trust-region enforcement.
    """
    w = weight_mat.detach()
    loss_unclipped = -log_prob * w

    clip_frac = torch.zeros(1, device=log_prob.device)

    if old_log_prob is not None:
        lb = math.log(max(1.0 - clip_ratio_low, 1e-8))  # negative; clip_ratio_low>=1 → -∞ (no lower clip)
        ub = math.log(1.0 + clip_ratio_high)              # positive
        delta = log_prob - old_log_prob.detach()

        clipped_delta = torch.clamp(delta, lb, ub)
        clipped_log_prob = old_log_prob.detach() + clipped_delta
        loss_clipped = -clipped_log_prob * w
        loss_mat = torch.max(loss_unclipped, loss_clipped)

        with torch.no_grad():
            is_outside = ((delta > ub) | (delta < lb)) & response_mask.bool()
            clip_frac = is_outside.float().sum() / (response_mask.sum() + 1e-8)
    else:
        loss_mat = loss_unclipped

    pg_loss = agg_loss(loss_mat=loss_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return pg_loss, weight_mat, clip_frac


def compute_lpo_weights_inline(
    current_log_prob: torch.Tensor,   # (B, L) detached current policy log-probs
    behavior_log_prob: torch.Tensor,  # (B, L) frozen rollout policy log-probs
    ref_log_prob: torch.Tensor,       # (B, L) frozen reference policy log-probs
    uid_int: torch.Tensor,            # (B,) integer group IDs
    outcome_rewards: torch.Tensor,    # (B,) scalar rewards per sample
    raw_rewards: torch.Tensor,        # (B,) raw scalar rewards for filter checks
    response_mask: torch.Tensor,      # (B, L)
    trust_lmd: float = 0.1,
    ref_lmd: float = 0.01,
    filter_all_failed: bool = True,
    filter_all_succeeded: bool = False,
) -> tuple:
    B, L = current_log_prob.shape
    device = current_log_prob.device
    dtype = current_log_prob.dtype
    metrics = {}

    with torch.no_grad():
        log_is_t = torch.clamp(current_log_prob - behavior_log_prob, -5.0, 5.0)

        seq_log_is = (log_is_t * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8)

        metrics["lpo/seq_log_is_mean"] = seq_log_is.mean().item()
        metrics["lpo/seq_log_is_abs_mean"] = seq_log_is.abs().mean().item()

        # ref KL ratio: log(π_ref / π_t) — used by both legacy and hybrid adaptive
        seq_log_ratio = None
        if ref_lmd > 0:
            log_ratio_t = torch.clamp(ref_log_prob - current_log_prob, -5.0, 5.0)
            seq_log_ratio = (log_ratio_t * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8)

        # ── Legacy fixed-temperature path ──────────────────────────────
        temperature = trust_lmd + ref_lmd
        aug_rewards = outcome_rewards + temperature * seq_log_is

        if seq_log_ratio is None:
            seq_log_ratio = torch.zeros_like(seq_log_is)

        potential = (aug_rewards / max(temperature, 1e-6)) + (ref_lmd / temperature) * seq_log_ratio
        metrics["lpo/potential_mean"] = potential.mean().item()
        metrics["lpo/potential_std"] = potential.std().item() if B > 1 else 0.0

        # ── Group loop ─────────────────────────────────────────────────────
        mirror_weights = torch.zeros(B, device=device, dtype=dtype)
        target_logits = torch.zeros(B, device=device, dtype=dtype)
        group_entropies = []
        group_size = 1
        group_etas = []
        group_kls = []
        n_flat = 0

        for gid in uid_int.unique():
            group_mask = uid_int == gid
            group_size = group_mask.sum().item()
            g_raw = raw_rewards[group_mask]

            if filter_all_failed and (g_raw <= 1e-2).all():
                continue
            if filter_all_succeeded and (g_raw >= 1 - 1e-2).all():
                continue

            g_pot = potential[group_mask]
            g_target = g_pot
            g_w = torch.softmax(g_target, dim=0) * group_size
            target_logits[group_mask] = g_target

            mirror_weights[group_mask] = g_w
            group_entropies.append(
                (-(g_w / group_size).clamp(min=1e-10)
                 * torch.log((g_w / group_size).clamp(min=1e-10))).sum().item()
            )

        metrics["lpo/weight_entropy_norm"] = (
            np.mean(group_entropies) / np.log(group_size)
            if group_size > 1 and group_entropies else 0.0
        )
        active = mirror_weights[mirror_weights > 0]
        if active.numel() > 0:
            metrics["lpo/weight_max"] = active.max().item()
            metrics["lpo/weight_std"] = active.std().item()
        metrics["lpo/weight_avg_nonzero"] = active.mean().item() if active.numel() > 0 else 0.0

        weight_mat = mirror_weights.view(B, 1).expand(B, L).contiguous()

    return weight_mat, metrics, target_logits


def compute_group_pi_weights(seq_log_is: torch.Tensor, uid_int: torch.Tensor) -> torch.Tensor:
    """Compute per-sequence pi_weights via group softmax.

    Called inside dp_actor per micro-batch using the *current* log_prob (exact).

    Args:
        seq_log_is: (B,) sequence-level log IS ratio (log π_θ_current / π_b).
        uid_int:    (B,) integer group identifiers (same value = same group).

    Returns:
        pi_weights: (B,) K·softmax(seq_log_is within group).
    """
    B = seq_log_is.shape[0]
    pi_weights = torch.zeros(B, device=seq_log_is.device, dtype=seq_log_is.dtype)
    for gid in uid_int.unique():
        mask = uid_int == gid
        group_size = mask.sum().item()
        pi_weights[mask] = torch.softmax(seq_log_is[mask], dim=0) * group_size
    return pi_weights
