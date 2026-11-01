# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score import math
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def extract_boxed_answer(example):
    example["answer"] = remove_boxed(example["answer"])
    return example


def get_first_answer(example):
    example["answer"] = example["answer"][0]
    return example


def make_test_map_fn(split, data_source):
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example.pop("problem")
        question = question + " " + instruction_following

        solution = example.pop("answer")
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": split, "index": idx},
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/math")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    # Build the evaluation mixture with the original task weights.
    test_data_sources = (
        ['math-ai/math500', 'math-ai/olympiadbench']
        + ['math-ai/minervamath'] * 4
        + ['math-ai/amc23'] * 32
        + ['math-ai/aime24'] * 32
        + ['math-ai/aime25'] * 32
    )
    test_dataset = []
    test_dataset.append(datasets.load_dataset('math-ai/math500', split='test').select_columns(['problem', 'answer']))
    test_dataset.append(datasets.load_dataset('math-ai/olympiadbench', split='test').rename_column("question", "problem").rename_column("final_answer", "answer").select_columns(['problem', 'answer']).map(get_first_answer))
    test_dataset.extend([datasets.load_dataset('math-ai/minervamath', split='test').rename_column("question", "problem").select_columns(['problem', 'answer'])] * 4)
    test_dataset.extend([datasets.load_dataset('math-ai/amc23', split='test').rename_column("question", "problem").select_columns(['problem', 'answer'])] * 32)
    test_dataset.extend([datasets.load_dataset('math-ai/aime24', split='test').rename_column("solution", "answer").select_columns(['problem', 'answer']).map(extract_boxed_answer)] * 32)
    test_dataset.extend([datasets.load_dataset('math-ai/aime25', split='test').select_columns(['problem', 'answer'])] * 32)

    for i in range(len(test_dataset)):
        test_dataset[i] = test_dataset[i].map(function=make_test_map_fn("test", test_data_sources[i]), with_indices=True)
    test_dataset = datasets.concatenate_datasets(test_dataset)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
