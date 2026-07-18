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

import os
import subprocess
import threading
import time
from typing import Any

GIB = 1024**3


class ResourceMonitor:
    """Periodically log host and GPU resource metrics through MetricLogger."""

    def __init__(self, cfg: Any, metric_logger: Any):
        monitor_cfg = cfg.runner.get("resource_monitor", {})
        self.enabled = bool(monitor_cfg.get("enabled", False))
        self.interval_sec = max(float(monitor_cfg.get("interval_sec", 5.0)), 1.0)
        self.include_per_gpu = bool(monitor_cfg.get("include_per_gpu", True))

        self.metric_logger = metric_logger
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._sample_index = 0

        self._psutil = None
        self._process = None
        self._prev_proc_cpu_times: tuple[int, int] | None = None

        self._nvml = None
        self._nvml_handles: list[Any] = []
        self._warned_messages: set[str] = set()

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._start_time = time.time()
        self._sample_index = 0
        self._stop_event.clear()
        self._init_cpu_sampler()
        self._init_gpu_sampler()

        self._thread = threading.Thread(
            target=self._run,
            name="ResourceMonitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 1.0)
            self._thread = None
        self._shutdown_gpu_sampler()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            metrics = self.sample()
            if metrics:
                try:
                    self.metric_logger.log(metrics, step=self._sample_index)
                except Exception as exc:
                    self._warn_once(f"failed to log resource metrics: {exc}")
            self._sample_index += 1
            self._stop_event.wait(self.interval_sec)

    def sample(self) -> dict[str, float]:
        metrics = {
            "system/elapsed_sec": time.time() - self._start_time,
        }
        metrics.update(self._sample_cpu_memory())
        metrics.update(self._sample_gpu())
        return metrics

    def _init_cpu_sampler(self) -> None:
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            psutil.cpu_percent(interval=None)
            self._process.cpu_percent(interval=None)
        except Exception as exc:
            self._warn_once(f"psutil unavailable, falling back to /proc: {exc}")
            self._psutil = None
            self._process = None
            self._prev_proc_cpu_times = self._read_proc_cpu_times()

    def _sample_cpu_memory(self) -> dict[str, float]:
        if self._psutil is not None:
            return self._sample_cpu_memory_with_psutil()
        return self._sample_cpu_memory_from_proc()

    def _sample_cpu_memory_with_psutil(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            virtual_memory = self._psutil.virtual_memory()
            metrics["system/cpu_percent"] = float(self._psutil.cpu_percent(interval=None))
            metrics["system/memory_used_gb"] = virtual_memory.used / GIB
            metrics["system/memory_total_gb"] = virtual_memory.total / GIB
            metrics["system/memory_percent"] = float(virtual_memory.percent)
            if self._process is not None:
                process_memory = self._process.memory_info()
                metrics["system/process_cpu_percent"] = float(
                    self._process.cpu_percent(interval=None)
                )
                metrics["system/process_memory_rss_gb"] = process_memory.rss / GIB
        except Exception as exc:
            self._warn_once(f"failed to sample psutil metrics: {exc}")
        return metrics

    def _sample_cpu_memory_from_proc(self) -> dict[str, float]:
        metrics: dict[str, float] = {}

        current_cpu_times = self._read_proc_cpu_times()
        if current_cpu_times is not None and self._prev_proc_cpu_times is not None:
            total_delta = current_cpu_times[0] - self._prev_proc_cpu_times[0]
            idle_delta = current_cpu_times[1] - self._prev_proc_cpu_times[1]
            if total_delta > 0:
                metrics["system/cpu_percent"] = (
                    (total_delta - idle_delta) / total_delta * 100.0
                )
        self._prev_proc_cpu_times = current_cpu_times

        memory_info = self._read_proc_memory_info()
        if memory_info is not None:
            total_kb, available_kb = memory_info
            used_kb = total_kb - available_kb
            metrics["system/memory_used_gb"] = used_kb * 1024 / GIB
            metrics["system/memory_total_gb"] = total_kb * 1024 / GIB
            metrics["system/memory_percent"] = used_kb / total_kb * 100.0

        return metrics

    def _read_proc_cpu_times(self) -> tuple[int, int] | None:
        try:
            with open("/proc/stat") as proc_stat:
                fields = proc_stat.readline().split()
        except OSError as exc:
            self._warn_once(f"failed to read /proc/stat: {exc}")
            return None

        if not fields or fields[0] != "cpu":
            return None

        values = [int(field) for field in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    def _read_proc_memory_info(self) -> tuple[int, int] | None:
        values: dict[str, int] = {}
        try:
            with open("/proc/meminfo") as meminfo:
                for line in meminfo:
                    key, value = line.split(":", 1)
                    values[key] = int(value.strip().split()[0])
        except OSError as exc:
            self._warn_once(f"failed to read /proc/meminfo: {exc}")
            return None

        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if total is None or available is None or total <= 0:
            return None
        return total, available

    def _init_gpu_sampler(self) -> None:
        if not self.include_per_gpu:
            return

        nvml = self._import_nvml()
        if nvml is None:
            return

        try:
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            self._nvml = nvml
            self._nvml_handles = [
                nvml.nvmlDeviceGetHandleByIndex(index) for index in range(device_count)
            ]
        except Exception as exc:
            self._warn_once(f"NVML unavailable, falling back to nvidia-smi: {exc}")
            self._nvml = None
            self._nvml_handles = []

    def _import_nvml(self) -> Any | None:
        try:
            import pynvml

            return pynvml
        except Exception:
            try:
                import ray._private.thirdparty.pynvml as pynvml

                return pynvml
            except Exception as exc:
                self._warn_once(f"pynvml unavailable, falling back to nvidia-smi: {exc}")
                return None

    def _sample_gpu(self) -> dict[str, float]:
        if not self.include_per_gpu:
            return {}
        if self._nvml is not None and self._nvml_handles:
            return self._sample_gpu_with_nvml()
        return self._sample_gpu_with_nvidia_smi()

    def _sample_gpu_with_nvml(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        gpu_utils: list[float] = []
        memory_used_gb: list[float] = []
        memory_used_percent: list[float] = []

        for index, handle in enumerate(self._nvml_handles):
            try:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            except Exception as exc:
                self._warn_once(f"failed to sample NVML GPU {index}: {exc}")
                continue

            gpu_util = float(utilization.gpu)
            memory_controller_util = float(utilization.memory)
            used_gb = memory.used / GIB
            total_gb = memory.total / GIB
            used_percent = used_gb / total_gb * 100.0 if total_gb > 0 else 0.0

            metrics[f"system/gpu_{index}_util_percent"] = gpu_util
            metrics[f"system/gpu_{index}_memory_controller_util_percent"] = (
                memory_controller_util
            )
            metrics[f"system/gpu_{index}_memory_used_gb"] = used_gb
            metrics[f"system/gpu_{index}_memory_total_gb"] = total_gb
            metrics[f"system/gpu_{index}_memory_used_percent"] = used_percent

            gpu_utils.append(gpu_util)
            memory_used_gb.append(used_gb)
            memory_used_percent.append(used_percent)

        self._add_gpu_aggregate_metrics(metrics, gpu_utils, memory_used_gb, memory_used_percent)
        return metrics

    def _sample_gpu_with_nvidia_smi(self) -> dict[str, float]:
        query_fields = [
            "index",
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "memory.total",
        ]
        command = [
            "nvidia-smi",
            f"--query-gpu={','.join(query_fields)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            )
        except Exception as exc:
            self._warn_once(f"failed to sample nvidia-smi metrics: {exc}")
            return {}

        metrics: dict[str, float] = {}
        gpu_utils: list[float] = []
        memory_used_gb: list[float] = []
        memory_used_percent: list[float] = []

        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != len(query_fields):
                continue
            try:
                index = int(parts[0])
                gpu_util = float(parts[1])
                memory_controller_util = float(parts[2])
                used_gb = float(parts[3]) / 1024.0
                total_gb = float(parts[4]) / 1024.0
            except ValueError:
                continue

            used_percent = used_gb / total_gb * 100.0 if total_gb > 0 else 0.0
            metrics[f"system/gpu_{index}_util_percent"] = gpu_util
            metrics[f"system/gpu_{index}_memory_controller_util_percent"] = (
                memory_controller_util
            )
            metrics[f"system/gpu_{index}_memory_used_gb"] = used_gb
            metrics[f"system/gpu_{index}_memory_total_gb"] = total_gb
            metrics[f"system/gpu_{index}_memory_used_percent"] = used_percent

            gpu_utils.append(gpu_util)
            memory_used_gb.append(used_gb)
            memory_used_percent.append(used_percent)

        self._add_gpu_aggregate_metrics(metrics, gpu_utils, memory_used_gb, memory_used_percent)
        return metrics

    def _add_gpu_aggregate_metrics(
        self,
        metrics: dict[str, float],
        gpu_utils: list[float],
        memory_used_gb: list[float],
        memory_used_percent: list[float],
    ) -> None:
        if not gpu_utils:
            return

        metrics["system/gpu_count"] = float(len(gpu_utils))
        metrics["system/gpu_util_percent_mean"] = sum(gpu_utils) / len(gpu_utils)
        metrics["system/gpu_util_percent_max"] = max(gpu_utils)
        metrics["system/gpu_memory_used_gb_total"] = sum(memory_used_gb)
        metrics["system/gpu_memory_used_gb_max"] = max(memory_used_gb)
        metrics["system/gpu_memory_used_percent_max"] = max(memory_used_percent)

    def _shutdown_gpu_sampler(self) -> None:
        if self._nvml is None:
            return
        try:
            self._nvml.nvmlShutdown()
        except Exception as exc:
            self._warn_once(f"failed to shutdown NVML: {exc}")
        finally:
            self._nvml = None
            self._nvml_handles = []

    def _warn_once(self, message: str) -> None:
        if message in self._warned_messages:
            return
        self._warned_messages.add(message)
        print(f"[RESOURCE_MONITOR] {message}")
