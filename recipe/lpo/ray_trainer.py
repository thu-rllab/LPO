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

import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           process_validation_metrics,
                                           reduce_metrics)

WorkerType = Type[Worker]

from verl.utils.tracking import ValidationGenerationsLogger
from verl import DataProto
from verl.trainer.ppo.ray_trainer import (RayPPOTrainer,
                                          ResourcePoolManager, Role, _timer,
                                          apply_kl_penalty)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

from verl.trainer.ppo.core_algos import agg_loss
from .utils import *
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions

import torch
import torch.nn.functional as F


class LPORayPPOTrainer(RayPPOTrainer):

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
    ):
        actor_cfg = config.actor_rollout_ref.actor

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.MAXRL,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        timing_raw = defaultdict(float)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics, timing_raw = self._validate(timing_raw=timing_raw)
            test_score = None
            for key in val_metrics.keys():
                if test_score is None and 'reward/mean@' in key:
                    test_score = val_metrics[key]
            acc_keys = [key for key in val_metrics.keys() if 'acc/mean@' in key]
            reward_keys = [key for key in val_metrics.keys() if 'reward/mean@' in key]
            test_score = np.mean([val_metrics[acc_keys[i]] for i in range(len(acc_keys))]) if len(acc_keys) > 0 else np.mean([val_metrics[reward_keys[i]] for i in range(len(reward_keys))])
            val_metrics['val/test_score/'] = test_score
            assert val_metrics
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps,
                            desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None

        num_gen_batches = 0
        n_rollouts = 0
        generate_tokens = 0
        metrics = {}

        # ---- filtering & buffer config (resolved once) ----
        actor_cfg = self.config.actor_rollout_ref.actor
        n_rollout: int = self.config.actor_rollout_ref.rollout.n

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # Pop keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_inputs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # ---- Generation ----
                    start_time = time.time()
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        n_rollouts += n_rollout
                        generate_tokens += torch.sum(compute_response_mask(gen_batch_output)).item()
                        metrics.update({'train/n_rollouts': n_rollouts,
                                        'train/generate_tokens': generate_tokens})
                    print('INFERENCE TIME:', time.time() - start_time)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    batch = batch.repeat(repeat_times=n_rollout, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # ---- Rewards ----
                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(batch)
                            reward_extra_infos_dict = {}

                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # ---- Response mask ----
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Compose training batch BEFORE expensive GPU passes (filter
                    # uninformative fresh groups, fill freed slots from buffer).
                    train_batch, fresh_uids, metrics = \
                        self._compose_train_batch_with_buffer(
                            batch, n_rollout, actor_cfg, metrics
                        )

                    if actor_cfg.get("loss_mode", "lpo") == "lpo":
                        _uid = train_batch.non_tensor_batch["uid"]
                        _uid_map = {u: i for i, u in enumerate(dict.fromkeys(_uid))}
                        train_batch.batch["uid_int"] = torch.tensor(
                            [_uid_map[u] for u in _uid], dtype=torch.long
                        )

                    # Balance train_batch for even token distribution across DP ranks.
                    # Done here so all downstream GPU passes (old_log_prob, ref, update)
                    # all benefit; uid-based CPU computations are order-independent.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(train_batch, metrics=metrics)

                    train_batch.meta_info["global_token_num"] = torch.sum(
                        train_batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # ---- Compute old_log_probs on train_batch only ----
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(train_batch)
                        entropys = old_log_prob.batch["entropys"]
                        train_response_masks_for_entropy = train_batch.batch["response_mask"]
                        loss_agg_mode = actor_cfg.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=train_response_masks_for_entropy,
                            loss_agg_mode=loss_agg_mode
                        )
                        metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                        train_batch = train_batch.union(old_log_prob)

                    # ---- Freeze behavior_log_prob for fresh groups ----
                    # Fresh groups: behavior = current policy log probs just computed (IS ratio = 1).
                    # Buffer groups: behavior was frozen at insertion time, already present.
                    train_uid_arr = train_batch.non_tensor_batch["uid"]
                    fresh_row_mask = np.array([uid in fresh_uids for uid in train_uid_arr])
                    fresh_idx = np.where(fresh_row_mask)[0]
                    if "behavior_log_prob" not in train_batch.batch.keys():
                        # Pure-fresh batch (no buffer data at all).
                        train_batch.batch["behavior_log_prob"] = \
                            train_batch.batch["old_log_probs"].clone()
                    elif len(fresh_idx) > 0:
                        # Mixed batch: only write fresh rows; buffer rows keep their frozen values.
                        train_batch.batch["behavior_log_prob"][fresh_idx] = \
                            train_batch.batch["old_log_probs"][fresh_idx].clone()

                    # ---- Reference log probs ----
                    _need_ref = self.use_reference_policy and (
                        actor_cfg.get("lpo_ref_lmd", 0.01) > 0
                        or actor_cfg.get("use_kl_loss", False)
                    )
                    if _need_ref:
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(train_batch)
                            train_batch = train_batch.union(ref_log_prob)

                    # ---- Values (GAE critic) ----
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(train_batch)
                            train_batch = train_batch.union(values)

                    # ---- Advantage computation ----
                    with _timer("adv", timing_raw):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        train_batch = compute_advantage(
                            train_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=n_rollout,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            batch_std_in_grpo=self.config.algorithm.get("batch_std_in_grpo", False),
                        )

                    # ---- Critic update ----
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(train_batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # ---- Actor update ----
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(train_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # ---- Validation ----
                    if (self.val_reward_fn is not None
                            and self.config.trainer.test_freq > 0
                            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics, timing_raw = self._validate(timing_raw=timing_raw)
                            test_score = None
                            for key in val_metrics.keys():
                                if test_score is None and 'reward/mean@' in key:
                                    test_score = val_metrics[key]
                            acc_keys = [key for key in val_metrics.keys() if 'acc/mean@' in key]
                            reward_keys = [key for key in val_metrics.keys() if 'reward/mean@' in key]
                            test_score = np.mean([val_metrics[acc_keys[i]] for i in range(len(acc_keys))]) if len(acc_keys) > 0 else np.mean([val_metrics[reward_keys[i]] for i in range(len(reward_keys))])
                            val_metrics['val/test_score/'] = test_score
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if (self.config.trainer.save_freq > 0
                            and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0)):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # ---- Logging ----
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=train_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=train_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=train_batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)

                metrics["train/num_gen_batches"] = num_gen_batches
                num_prompt_in_batch = 0
                num_gen_batches = 0
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

    # ------------------------------------------------------------------
    # Train-batch assembly helper
    # ------------------------------------------------------------------

    def _balance_batch(self, batch: DataProto, metrics: dict = None):
        actor_cfg = self.config.actor_rollout_ref.actor
        if actor_cfg.get("loss_mode", "lpo") != "lpo":
            return super()._balance_batch(batch, metrics=metrics)
        if "uid" not in batch.non_tensor_batch:
            return super()._balance_batch(batch, metrics=metrics)

        uid_arr = batch.non_tensor_batch["uid"]  # (B,) string UIDs
        seqlens = batch.batch["attention_mask"].sum(dim=-1).tolist()
        world_size = self.actor_rollout_wg.world_size

        # Build group → [seq_indices] mapping preserving original order
        uid_to_indices: dict = {}
        for i, uid in enumerate(uid_arr):
            uid_to_indices.setdefault(uid, []).append(i)

        group_ids = list(uid_to_indices.keys())
        group_seqlens = [sum(seqlens[i] for i in uid_to_indices[g]) for g in group_ids]
        n_groups = len(group_ids)

        assert n_groups % world_size == 0, (
            f"n_groups ({n_groups}) not divisible by world_size ({world_size}). "
            f"LPO requires whole groups to live on a single DP rank; adjust "
            f"train_batch_size / n_resp_per_prompt / buffer composition so "
            f"n_groups % world_size == 0."
        )

        group_partitions = get_seqlen_balanced_partitions(
            group_seqlens, k_partitions=world_size, equal_size=True
        )

        # Flatten: sequences within a group stay adjacent, groups ordered by partition
        new_order = []
        for partition in group_partitions:
            for gid_local_idx in partition:
                gid = group_ids[gid_local_idx]
                new_order.extend(uid_to_indices[gid])

        batch.reorder(torch.tensor(new_order, dtype=torch.long))

        if metrics is not None:
            per_gpu = len(new_order) // world_size
            gpu_totals = [
                sum(seqlens[new_order[j]] for j in range(r * per_gpu, (r + 1) * per_gpu))
                for r in range(world_size)
            ]
            metrics["seqlen_balance/group_aware_min"] = min(gpu_totals)
            metrics["seqlen_balance/group_aware_max"] = max(gpu_totals)

    def _compose_train_batch_with_buffer(
        self,
        batch: "DataProto",
        n_rollout: int,
        actor_cfg,
        metrics: dict,
    ) -> tuple:
        """Compose the training batch by filtering uninformative fresh groups and
        filling freed slots from the replay buffer.

        Returns (train_batch, fresh_uids, metrics).

        When self.buffer is None the fast path is taken: no filtering is done and
        every uid in the fresh batch is treated as a fresh uid.
        """
        total_train_rows = batch.batch.batch_size[0]
        total_groups = total_train_rows // n_rollout

        metrics["buffer/n_filled"] = 0
        metrics["buffer/adaptive_offp_ratio"] = 0.0
        fresh_uids = set(batch.non_tensor_batch["uid"])
        train_batch = self._round_groups_to_world_size(batch, n_rollout)
        if train_batch is not batch:
            fresh_uids = set(train_batch.non_tensor_batch["uid"])
        return train_batch, fresh_uids, metrics

    def _round_groups_to_world_size(self, batch: "DataProto", n_rollout: int) -> "DataProto":
        """Trim trailing whole groups so the batch's group count is a multiple
        of world_size.  Required because `_balance_batch` (LPO mode) asserts
        groups divide evenly across DP ranks.

        Assumes rows are stored group-contiguously (each `n_rollout` consecutive
        rows share the same uid), which holds for both
        `select_informative_groups` (np.concatenate over per-group index lists)
        and `buffer.sample` (DataProto.concat of per-group DataProtos).
        """
        ws = self.actor_rollout_wg.world_size
        cur_rows = batch.batch.batch_size[0]
        cur_groups = cur_rows // n_rollout
        if cur_groups == 0 or cur_groups % ws == 0:
            return batch
        keep_groups = (cur_groups // ws) * ws
        keep_rows = keep_groups * n_rollout
        print(
            f"[lpo] _round_groups_to_world_size: trimming "
            f"{cur_groups} -> {keep_groups} groups (world_size={ws}, "
            f"n_rollout={n_rollout})"
        )
        return batch.select_idxs(np.arange(keep_rows))

    def _validate(self, timing_raw):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        with _timer("val_generate", timing_raw):
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)
                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                    interleave=True
                )

                if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                    return {}

                input_ids = test_batch.batch["input_ids"]
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_inputs" in test_batch.non_tensor_batch:
                    input_texts_img = []
                    for id, text in enumerate(input_texts):
                        input_texts_img.append(text + str(test_batch.non_tensor_batch["multi_modal_data"][id]['image'][0]))
                    input_texts = input_texts_img
                    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
                if "raw_prompt" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                test_gen_batch = test_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                sample_inputs.extend(input_texts)

                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                }

                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                    test_gen_batch, self.actor_rollout_wg.world_size
                )
                if not self.async_rollout_mode:
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                else:
                    self.async_rollout_manager.wake_up()
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                    self.async_rollout_manager.sleep()

                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)

                result = self.val_reward_fn(test_batch, return_dict=True)
                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                reward_extra_infos_dict["reward"].extend(scores)
                if "reward_extra_info" in result:
                    for key, lst in result["reward_extra_info"].items():
                        reward_extra_infos_dict[key].extend(lst)

                data_source_lst.append(
                    test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
                )

        with _timer("val_log_dump", timing_raw):
            self._maybe_log_val_generations(
                inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
            )
            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir:
                self._dump_generations(
                    inputs=sample_inputs,
                    outputs=sample_outputs,
                    scores=sample_scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=val_data_dir,
                )

            for key_info, lst in reward_extra_infos_dict.items():
                assert len(lst) == 0 or len(lst) == len(sample_scores)

            data_sources = np.concatenate(data_source_lst, axis=0)
            data_src2var2metric2val = process_validation_metrics(
                data_sources, sample_inputs, reward_extra_infos_dict
            )
            metric_dict = {}
            for data_source, var2metric2val in data_src2var2metric2val.items():
                core_var = "acc" if "acc" in var2metric2val else "reward"
                for var_name, metric2val in var2metric2val.items():
                    n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                    for metric_name, metric_val in metric2val.items():
                        if (var_name == core_var and
                                any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and
                                (f"@{n_max}" in metric_name)):
                            metric_sec = "val-core"
                        else:
                            metric_sec = "val-aux"
                        pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                        metric_dict[pfx] = metric_val

        return metric_dict, timing_raw
