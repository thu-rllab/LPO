# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Optional

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=120.0):
    loop = asyncio.get_running_loop()
    try:
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference, task_extra_info),
                ),
                timeout=timeout,
            )
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        print(f"Error processing completion: {str(completion)[:50]}, Error: {e}")
        return None


async def parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    scores = []
    executor = ProcessPoolExecutor(max_workers=num_processes)
    try:
        if extra_info is None:
            extra_info = [None] * len(tasks)
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=120.0)
            for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info)
        ]
        results = await asyncio.gather(*tasks_async, return_exceptions=False)
    finally:
        for pid, proc in executor._processes.items():
            try:
                proc.kill()
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)

    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


class NewPrimeRewardManager:
    """Optimized PRIME reward manager that uses newprime_code for faster code evaluation.

    Compared to PrimeRewardManager:
    - Tighter per-sample async timeout (120s vs 300s)
    - Dynamic num_processes based on cpu_count
    - Proper executor shutdown to prevent deadlocks
    - Returns reward_extra_info to prevent double computation
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def verify(self, data):
        prompt_ids = data.batch["prompts"]

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = asyncio.run(
                parallel_compute_score_async(
                    self.compute_score,
                    sequences_str,
                    ground_truth,
                    data_sources,
                    extra_info=extra_info,
                    num_processes=min(256, (os.cpu_count() or 64)),
                )
            )
        except asyncio.TimeoutError:
            print("Global timeout in reward computing! Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)

        scores = self.verify(data)

        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        else:
            return reward_tensor
