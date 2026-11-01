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

import multiprocessing
import os
import sys
import tempfile
import traceback

from .testing_util import run_test_continuous


def _temp_run_continuous_q(sample, generation, debug, queue, timeout):
    """Run run_test_continuous in an isolated subprocess, sending results via Queue."""
    tmpdir = tempfile.mkdtemp(prefix="code_eval_")
    os.chdir(tmpdir)
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata = run_test_continuous(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            queue.put((res, metadata))
        except Exception:
            traceback.print_exc(10)
            queue.put(([-1 for i in range(len(sample["inputs"]))], {}))


def check_correctness_continuous(in_outs, generation, timeout=10, debug=False):
    """Run ALL test cases in a single subprocess without early return.

    Uses run_test_continuous which compiles the code once and tests every case,
    eliminating the per-case subprocess spawn overhead of the original approach.

    The total subprocess timeout scales with the number of test cases.
    """
    num_cases = len(in_outs["inputs"])
    total_timeout = timeout * num_cases + 5

    q = multiprocessing.Queue(maxsize=1)
    p = multiprocessing.Process(
        target=_temp_run_continuous_q,
        args=(in_outs, generation, debug, q, timeout),
    )
    p.start()
    p.join(timeout=total_timeout)
    if p.is_alive():
        p.kill()
        p.join(timeout=1)

    result = None
    metadata_list = []
    try:
        if not q.empty():
            res, metadata = q.get_nowait()
            result = res
            metadata_list = [metadata]
    except Exception:
        pass
    finally:
        q.close()
        q.join_thread()

    if result is None:
        result = [-1 for i in range(num_cases)]
        if debug:
            print("global timeout in check_correctness_continuous")
    return result, metadata_list
