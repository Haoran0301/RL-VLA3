#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ManiSkill 环境下的 Pi0 inference 耗时测试（兼容旧用法）。

实际实现已集成到 rlinf.envs.bench_pi0_inference，此处为便捷入口。
"""

if __name__ == "__main__":
    import sys

    # 注入 --env maniskill 以保持向后兼容
    if "--env" not in " ".join(sys.argv):
        sys.argv.extend(["--env", "maniskill"])

    from rlinf.envs.bench_pi0_inference import main

    main()
