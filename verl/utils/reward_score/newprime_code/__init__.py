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

import json
import traceback

from verl.utils.reward_score.prime_code.utils import check_correctness as apps_check_correctness
from .utils import check_correctness_continuous


def compute_score(completion, test_cases, continuous=False):
    solution = completion.split("```python")[-1].split("```")[0]
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error:{e}")

        # Fast path: run all test cases in one subprocess (same as original)
        try:
            res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x is True, res))
            if success:
                return success, metadata
        except Exception:
            pass

        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]

        if continuous:
            # Build a subset with at most 10 test cases.
            # Strip fn_name to match original per-case behavior (bug-compatible).
            num_cases = min(10, len(inputs))
            subset = {"inputs": inputs[:num_cases], "outputs": outputs[:num_cases]}

            res, metadata_list = check_correctness_continuous(
                in_outs=subset, generation=solution, timeout=10, debug=False,
            )
            res_count = len(res) if len(res) > 0 else 1
            success = sum(map(lambda x: x is True, res)) / res_count
        else:
            success = False
            metadata_list = None
    except Exception:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    return success, metadata_list
