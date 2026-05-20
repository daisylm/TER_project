from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import shlex
import statistics
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except ModuleNotFoundError:  
    psutil = None


DEFAULT_PROMPTS = Path("12_prompt_dataset.json")
FALLBACK_PROMPTS = Path("generated_prompts/12_prompt_dataset.json")
OUTPUT_DIR = Path("benchmark_artifacts")
DEFAULT_OLLAMA_COMMAND = "ollama serve"
DEFAULT_WARMUP_PROMPT = "Reply with exactly: ready."

PERF_EVENTS = [
    "task-clock",
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "branches",
    "branch-misses",
    "dTLB-loads",
    "dTLB-load-misses",
    "iTLB-loads",
    "iTLB-load-misses",
    "context-switches",
    "cpu-migrations",
    "page-faults",
    "minor-faults",
    "major-faults",
]


@dataclass
class MemoryTrace:
    samples: list[dict[str, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self, pid: int, interval: float) -> None:
        self._thread = threading.Thread(
            target=self._sample_loop,
            args=(pid, interval),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return summarize_memory(self.samples)

    def _sample_loop(self, pid: int, interval: float) -> None:
        if psutil is None:
            return
        start = time.perf_counter()
        try:
            root = psutil.Process(pid)
        except psutil.Error:
            return

        while not self._stop.is_set():
            try:
                processes = [root] + root.children(recursive=True)
                rss = sum(proc.memory_info().rss for proc in processes if proc.is_running())
                self.samples.append(
                    {
                        "t_seconds": round(time.perf_counter() - start, 6),
                        "rss_bytes": float(rss),
                    }
                )
            except psutil.Error:
                pass
            self._stop.wait(interval)


def load_prompts(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    prompt_path = path
    if not prompt_path.exists() and path == DEFAULT_PROMPTS and FALLBACK_PROMPTS.exists():
        prompt_path = FALLBACK_PROMPTS

    raw = json.loads(prompt_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        prompts = raw.get("prompts")
        metadata = raw.get("metadata", {})
    elif isinstance(raw, list):
        prompts = raw
        metadata = {}
    else:
        raise ValueError(f"Unsupported prompt file shape in {prompt_path}")

    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")

    normalized = []
    for index, prompt in enumerate(prompts, start=1):
        if not isinstance(prompt, dict):
            raise ValueError(f"Prompt #{index} is not an object")
        text = prompt.get("prompt") or prompt.get("text")
        if not text and prompt.get("file"):
            file_path = prompt_path.parent / prompt["file"]
            text = file_path.read_text(encoding="utf-8")
        if not text:
            raise ValueError(f"Prompt #{index} has no prompt/text/file content")
        normalized.append(
            {
                **prompt,
                "id": str(prompt.get("id", f"prompt_{index:03d}")),
                "prompt": text,
                "category": prompt.get("category", "unknown"),
                "title": prompt.get("title", ""),
                "metrics": prompt.get("metrics", {}),
            }
        )
    return normalized, metadata, prompt_path


def is_ollama_process(proc_info: dict[str, Any]) -> bool:
    name = (proc_info.get("name") or "").lower()
    cmdline = proc_info.get("cmdline") or []
    executable = Path(cmdline[0]).name.lower() if cmdline else ""
    return name == "ollama" or executable == "ollama"


def find_ollama_pid() -> int:
    if psutil is None:
        raise SystemExit("psutil is required for PID auto-detection. Use llm_env/bin/python or install psutil.")
    candidates = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if is_ollama_process(proc.info):
                candidates.append(proc.info["pid"])
        except psutil.Error:
            continue
    if not candidates:
        raise SystemExit("Could not find an Ollama process. Pass --pid explicitly.")
    return int(candidates[0])


def is_ollama_reachable(server: str, timeout: float = 2.0) -> bool:
    request = urllib.request.Request(f"{server.rstrip('/')}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except Exception:
        return False


def wait_for_ollama(server: str, timeout_seconds: float, poll_interval: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_ollama_reachable(server, timeout=min(2.0, poll_interval)):
            return
        time.sleep(poll_interval)
    raise SystemExit(f"Ollama did not become reachable at {server} within {timeout_seconds:.1f}s")


def wait_until_ollama_stops(server: str, timeout_seconds: float, poll_interval: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not is_ollama_reachable(server, timeout=min(2.0, poll_interval)):
            return
        time.sleep(poll_interval)
    raise SystemExit(f"Ollama stayed reachable at {server} after restart shutdown request")


def find_ollama_processes() -> list[Any]:
    if psutil is None:
        return []
    processes = []
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info.get("pid") == current_pid:
                continue
            if is_ollama_process(proc.info):
                processes.append(proc)
        except psutil.Error:
            continue
    return processes


def restart_ollama(args: argparse.Namespace) -> None:
    if args.pid or not args.restart_ollama:
        return
    processes = find_ollama_processes()
    if not processes and not is_ollama_reachable(args.server):
        return

    print("Restarting Ollama before benchmark", flush=True)
    for proc in processes:
        try:
            proc.terminate()
        except psutil.Error:
            continue

    gone, alive = psutil.wait_procs(processes, timeout=args.ollama_restart_timeout) if processes else ([], [])
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            continue
    if alive:
        psutil.wait_procs(alive, timeout=5)
    wait_until_ollama_stops(args.server, args.ollama_restart_timeout, args.ollama_poll_interval)


def start_ollama(args: argparse.Namespace) -> subprocess.Popen[bytes] | None:
    if args.pid or not args.start_ollama:
        return None
    if is_ollama_reachable(args.server):
        print(f"Ollama is already reachable at {args.server}", flush=True)
        return None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "ollama_serve.log"
    log_file = log_path.open("ab")
    command = shlex.split(args.ollama_command)
    print(f"Starting Ollama with: {' '.join(command)}", flush=True)
    proc = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._benchmark_log_file = log_file  # type: ignore[attr-defined]
    try:
        wait_for_ollama(args.server, args.ollama_ready_timeout, args.ollama_poll_interval)
    except Exception:
        stop_ollama(proc)
        raise
    print(f"Ollama is reachable at {args.server}", flush=True)
    return proc


def stop_ollama(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    stop_process_group(proc, signal.SIGTERM)
    log_file = getattr(proc, "_benchmark_log_file", None)
    if log_file is not None:
        log_file.close()


def write_warmup_result(output_dir: Path, result: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "warmup_result.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def run_warmup(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    print("Sending unmeasured warmup prompt to load the model", flush=True)
    try:
        response = post_ollama_stream(
            args.server,
            args.model,
            args.warmup_prompt,
            args.warmup_num_predict,
            args.timeout,
        )
        completed = bool(response.get("response") or response.get("ollama_metrics"))
        result = {
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "completed": completed,
            "excluded_from_metrics": True,
            "model": args.model,
            "server": args.server,
            "num_predict": args.warmup_num_predict,
            "prompt": args.warmup_prompt,
            "response_content": response.get("response", ""),
            "timing": {
                "time_to_first_token_seconds": response.get("time_to_first_token_seconds"),
                "tokens_per_second": response.get("tokens_per_second"),
                "total_execution_seconds": response.get("total_execution_seconds"),
            },
            "ollama_metrics": response.get("ollama_metrics", {}),
            "error": None,
        }
    except Exception as exc:
        result = {
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "completed": False,
            "excluded_from_metrics": True,
            "model": args.model,
            "server": args.server,
            "num_predict": args.warmup_num_predict,
            "prompt": args.warmup_prompt,
            "response_content": "",
            "timing": {},
            "ollama_metrics": {},
            "error": repr(exc),
        }

    artifact = write_warmup_result(args.output_dir, result)
    if not result["completed"]:
        raise SystemExit(f"Warmup failed; details written to {artifact}: {result['error']}")
    print(f"Warmup completed and saved to {artifact}", flush=True)
    return result


def post_ollama_stream(
    server: str,
    model: str,
    prompt: str,
    num_predict: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": num_predict},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{server.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_token = None
    response_parts: list[str] = []
    final_metrics: dict[str, Any] = {}

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line.decode("utf-8"))
            chunk = event.get("response", "")
            if chunk and first_token is None:
                first_token = time.perf_counter()
            response_parts.append(chunk)
            if event.get("done"):
                final_metrics = event

    finished = time.perf_counter()
    eval_count = final_metrics.get("eval_count") or 0
    eval_duration_ns = final_metrics.get("eval_duration") or 0
    tokens_per_second = None
    if eval_count and eval_duration_ns:
        tokens_per_second = eval_count / (eval_duration_ns / 1_000_000_000)

    return {
        "response": "".join(response_parts),
        "time_to_first_token_seconds": None
        if first_token is None
        else first_token - started,
        "total_execution_seconds": finished - started,
        "tokens_per_second": tokens_per_second,
        "ollama_metrics": {
            key: final_metrics.get(key)
            for key in [
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            ]
        },
    }


def perf_command(args: argparse.Namespace, *parts: str) -> list[str]:
    command = ["perf", *parts]
    if args.sudo_perf and os.geteuid() != 0:
        return ["sudo", "-n", *command]
    return command


def validate_sudo_for_perf(args: argparse.Namespace) -> None:
    if not args.sudo_perf or os.geteuid() == 0:
        return
    try:
        subprocess.run(["sudo", "-v"], check=True)
    except FileNotFoundError:
        raise SystemExit("--sudo-perf was requested, but sudo was not found.")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"sudo authentication failed for perf: {exc}")


def start_perf_stat(
    args: argparse.Namespace,
    pid: int,
    output: Path,
    timeout: int,
) -> subprocess.Popen[bytes]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = perf_command(
        args,
        "stat",
        "-x",
        ",",
        "-e",
        ",".join(PERF_EVENTS),
        "-p",
        str(pid),
        "-o",
        str(output),
        "--",
        "sleep",
        str(timeout),
    )
    env = {**os.environ, "LC_ALL": "C"}
    return subprocess.Popen(command, start_new_session=True, env=env)


def start_perf_record(
    args: argparse.Namespace,
    pid: int,
    output: Path,
    frequency: int,
) -> subprocess.Popen[bytes]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = perf_command(
        args,
        "record",
        "-F",
        str(frequency),
        "-g",
        "-p",
        str(pid),
        "-o",
        str(output),
    )
    env = {**os.environ, "LC_ALL": "C"}
    return subprocess.Popen(command, start_new_session=True, env=env)


def stop_process_group(proc: subprocess.Popen[bytes], sig: int = signal.SIGINT) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def parse_perf_stat(path: Path) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    if not path.exists():
        return metrics

    for row in csv.reader(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if len(row) < 3 or not row[0].strip():
            continue
        value_raw = row[0].strip().replace(",", "")
        event = row[2].strip()
        if not event or value_raw.startswith("<not"):
            continue
        try:
            metrics[event] = float(value_raw)
        except ValueError:
            metrics[event] = value_raw

    cycles = as_float(metrics.get("cycles"))
    instructions = as_float(metrics.get("instructions"))
    cache_refs = as_float(metrics.get("cache-references"))
    cache_misses = as_float(metrics.get("cache-misses"))
    branches = as_float(metrics.get("branches"))
    branch_misses = as_float(metrics.get("branch-misses"))
    dtlb_loads = as_float(metrics.get("dTLB-loads"))
    dtlb_misses = as_float(metrics.get("dTLB-load-misses"))
    itlb_loads = as_float(metrics.get("iTLB-loads"))
    itlb_misses = as_float(metrics.get("iTLB-load-misses"))

    if cycles and instructions is not None:
        metrics["ipc"] = instructions / cycles
    if cache_refs and cache_misses is not None:
        metrics["cache_miss_rate"] = cache_misses / cache_refs
    if branches and branch_misses is not None:
        metrics["branch_miss_rate"] = branch_misses / branches
    if dtlb_loads and dtlb_misses is not None:
        metrics["dtlb_load_miss_rate"] = dtlb_misses / dtlb_loads
    if itlb_loads and itlb_misses is not None:
        metrics["itlb_load_miss_rate"] = itlb_misses / itlb_loads
    return metrics


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def estimate_cpu_frequency_hz() -> float | None:
    if psutil is None:
        return None
    try:
        frequency = psutil.cpu_freq()
    except psutil.Error:
        return None
    if frequency is None or frequency.current is None:
        return None
    return frequency.current * 1_000_000


def task_clock_seconds(raw_value: float, wall_time_seconds: float | None) -> tuple[float, str]:
    cpu_count = os.cpu_count() or 1
    if wall_time_seconds and raw_value > wall_time_seconds * 1_000 * cpu_count * 10:
        return raw_value / 1_000_000_000, "task-clock-ns"
    return raw_value / 1_000, "task-clock-ms"


def derive_blocked_time(
    wall_time_seconds: float | None,
    perf_stat: dict[str, Any],
    cpu_frequency_hz: float | None,
) -> dict[str, Any]:
    task_clock_raw = as_float(perf_stat.get("task-clock"))
    cycles = as_float(perf_stat.get("cycles"))

    if wall_time_seconds is None:
        return {
            "cpu_time_seconds": None,
            "blocked_time_seconds": None,
            "percent_blocked": None,
            "cpu_frequency_hz": cpu_frequency_hz,
            "cpu_time_source": None,
            "note": "Blocked time requires wall time.",
        }

    cpu_time_source = None
    if task_clock_raw is not None:
        cpu_time_seconds, cpu_time_source = task_clock_seconds(task_clock_raw, wall_time_seconds)
    elif cycles is not None and cpu_frequency_hz:
        cpu_time_seconds = cycles / cpu_frequency_hz
        cpu_time_source = "cycles/cpu_frequency_hz"
    else:
        return {
            "cpu_time_seconds": None,
            "blocked_time_seconds": None,
            "percent_blocked": None,
            "cpu_frequency_hz": cpu_frequency_hz,
            "cpu_time_source": None,
            "note": "Blocked time requires task-clock or cycles plus CPU frequency.",
        }

    blocked_time_seconds = wall_time_seconds - cpu_time_seconds
    percent_blocked = (blocked_time_seconds / wall_time_seconds) * 100 if wall_time_seconds else None
    note = None
    if cpu_time_source in {"task-clock-ms", "task-clock-ns"}:
        note = f"CPU time is derived from perf {cpu_time_source}, summed across the measured Ollama PID."
    if blocked_time_seconds < 0:
        note = (
            "CPU time exceeds wall time; this can happen when perf task-clock sums CPU time "
            "across parallel threads, so blocked_time_seconds should not be read as idle wall time."
        )
    elif cpu_time_source == "cycles/cpu_frequency_hz":
        note = "CPU time is estimated from cycles and CPU frequency; task-clock was unavailable."

    return {
        "cpu_time_seconds": cpu_time_seconds,
        "blocked_time_seconds": blocked_time_seconds,
        "percent_blocked": percent_blocked,
        "cpu_frequency_hz": cpu_frequency_hz,
        "cpu_time_source": cpu_time_source,
        "note": note,
    }


def summarize_memory(samples: list[dict[str, float]]) -> dict[str, Any]:
    rss_values = [sample["rss_bytes"] for sample in samples]
    if not rss_values:
        return {
            "sample_count": 0,
            "peak_rss_bytes": None,
            "average_rss_bytes": None,
            "rss_variance": None,
            "rss_stddev": None,
            "samples": [],
        }
    return {
        "sample_count": len(rss_values),
        "peak_rss_bytes": max(rss_values),
        "average_rss_bytes": statistics.fmean(rss_values),
        "rss_variance": statistics.pvariance(rss_values) if len(rss_values) > 1 else 0.0,
        "rss_stddev": statistics.pstdev(rss_values) if len(rss_values) > 1 else 0.0,
        "samples": samples,
    }


def generate_flamegraph(
    args: argparse.Namespace,
    perf_data: Path,
    flamegraph: Path,
    stack_dir: Path,
) -> dict[str, Any]:
    script_output = perf_data.with_suffix(".perf")
    folded_output = perf_data.with_suffix(".folded")
    stackcollapse = stack_dir / "stackcollapse-perf.pl"
    flamegraph_pl = stack_dir / "flamegraph.pl"

    if not stackcollapse.exists() or not flamegraph_pl.exists():
        return {"generated": False, "reason": f"FlameGraph scripts not found in {stack_dir}"}

    with script_output.open("w", encoding="utf-8") as perf_script:
        subprocess.run(
            perf_command(args, "script", "-i", str(perf_data)),
            stdout=perf_script,
            check=True,
        )
    with folded_output.open("w", encoding="utf-8") as folded:
        subprocess.run([str(stackcollapse), str(script_output)], stdout=folded, check=True)
    with flamegraph.open("w", encoding="utf-8") as svg:
        subprocess.run([str(flamegraph_pl), str(folded_output)], stdout=svg, check=True)

    return {
        "generated": True,
        "perf_script": str(script_output),
        "folded": str(folded_output),
        "svg": str(flamegraph),
    }


def coefficient_of_variation(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if len(clean) < 2:
        return None
    mean = statistics.fmean(clean)
    if mean == 0:
        return None
    return statistics.pstdev(clean) / mean


def diagnose_bottleneck(perf_stat: dict[str, Any], memory: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    """Given a single run's metrics, return a bottleneck diagnosis."""
    diagnosis: dict[str, Any] = {
        "primary_bottleneck": None,
        "confidence": None,
        "evidence": [],
        "recommendation": None,
    }

    cache_miss_rate = as_float(perf_stat.get("cache_miss_rate"))
    ipc = as_float(perf_stat.get("ipc"))

    if cache_miss_rate is not None and ipc is not None:
        if cache_miss_rate > 0.85 and ipc < 0.5:
            diagnosis["primary_bottleneck"] = "MEMORY_BANDWIDTH"
            diagnosis["confidence"] = min(1.0, (cache_miss_rate - 0.7) / 0.3)
            diagnosis["evidence"].append(f"Cache miss rate {cache_miss_rate:.1%} (>85%)")
            diagnosis["evidence"].append(f"IPC {ipc:.2f} (<0.5)")
            diagnosis["recommendation"] = "Enable huge pages, optimize memory layout, reduce model size"

    dtlb_miss_rate = as_float(perf_stat.get("dtlb_load_miss_rate"))
    if dtlb_miss_rate is not None and dtlb_miss_rate > 0.10:
        if diagnosis["primary_bottleneck"]:
            diagnosis["secondary_bottleneck"] = "TLB_THRASHING"
        else:
            diagnosis["primary_bottleneck"] = "TLB_THRASHING"
            diagnosis["confidence"] = min(1.0, dtlb_miss_rate / 0.15)
        diagnosis["evidence"].append(f"dTLB miss rate {dtlb_miss_rate:.1%} (>10%)")
        diagnosis["recommendation"] = (
            "Enable transparent huge pages (THP): "
            "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
        )

    page_faults = as_float(perf_stat.get("page-faults"))
    minor_faults = as_float(perf_stat.get("minor-faults"))
    major_faults = as_float(perf_stat.get("major-faults"))
    if major_faults is not None and major_faults > 0:
        diagnosis["primary_bottleneck"] = "I_O_BOUND"
        diagnosis["confidence"] = min(1.0, 0.6 + major_faults / 1_000)
        diagnosis["evidence"].append(f"{major_faults:,.0f} major page faults (requires disk I/O)")
        if minor_faults is not None:
            diagnosis["evidence"].append(f"{minor_faults:,.0f} minor page faults (RAM-backed mappings)")
        diagnosis["recommendation"] = "Preload model, increase RAM, use faster storage, or reduce model size"
    elif page_faults is not None and page_faults > 50_000:
        diagnosis["primary_bottleneck"] = "I_O_BOUND"
        diagnosis["confidence"] = 0.55
        diagnosis["evidence"].append(f"{page_faults:,.0f} total page faults")
        if minor_faults is not None:
            diagnosis["evidence"].append(f"{minor_faults:,.0f} minor page faults (fast, already in RAM)")
        diagnosis["recommendation"] = "Inspect major-faults; high minor faults alone may just be memory mapping"

    context_switches = as_float(perf_stat.get("context-switches"))
    if context_switches is not None and context_switches > 5_000:
        diagnosis["primary_bottleneck"] = "SCHEDULER_CONTENTION"
        diagnosis["confidence"] = min(1.0, context_switches / 10_000)
        diagnosis["evidence"].append(f"{context_switches:,.0f} context switches (>5000)")
        diagnosis["recommendation"] = "Isolate CPU cores with taskset, reduce system load"

    if (
        ipc is not None
        and ipc > 2.0
        and cache_miss_rate is not None
        and cache_miss_rate < 0.5
    ):
        diagnosis["primary_bottleneck"] = "CPU_COMPUTE_BOUND"
        diagnosis["confidence"] = min(1.0, (ipc - 1.5) / 1.5)
        diagnosis["evidence"].append(f"High IPC {ipc:.2f} with low cache misses {cache_miss_rate:.1%}")
        diagnosis["recommendation"] = "Use smaller model or optimize arithmetic intensity"

    if diagnosis["primary_bottleneck"] is None:
        avg_rss = as_float(memory.get("average_rss_bytes"))
        total_seconds = as_float(timing.get("total_execution_seconds"))
        if avg_rss is not None and total_seconds is not None:
            diagnosis["evidence"].append(
                f"No threshold crossed; avg RSS {avg_rss / (1024 ** 3):.2f} GiB over {total_seconds:.2f}s"
            )

    return diagnosis


def compare_prompt_categories(all_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare bottlenecks across prompt categories."""
    categories: dict[str, list[dict[str, Any]]] = {}
    for run in all_runs:
        cat = run.get("category", "unknown")
        categories.setdefault(cat, []).append(run)

    comparison = {}
    for cat, runs in categories.items():
        cache_miss_rates = [
            as_float(r["perf_stat"].get("cache_miss_rate"))
            for r in runs
            if as_float(r["perf_stat"].get("cache_miss_rate")) is not None
        ]
        ipcs = [
            as_float(r["perf_stat"].get("ipc"))
            for r in runs
            if as_float(r["perf_stat"].get("ipc")) is not None
        ]
        ttfts = [
            as_float(r["timing"].get("time_to_first_token_seconds"))
            for r in runs
            if as_float(r["timing"].get("time_to_first_token_seconds")) is not None
        ]

        comparison[cat] = {
            "avg_cache_miss_rate": statistics.fmean(cache_miss_rates) if cache_miss_rates else None,
            "avg_ipc": statistics.fmean(ipcs) if ipcs else None,
            "avg_ttft": statistics.fmean(ttfts) if ttfts else None,
            "runs": len(runs),
        }

    insights = []
    baseline = comparison.get("baseline")
    ram_stress = comparison.get("ram_stress")
    cpu_stress = comparison.get("cpu_stress")

    if baseline and ram_stress:
        baseline_ttft = baseline["avg_ttft"]
        ram_ttft = ram_stress["avg_ttft"]
        if ram_ttft and baseline_ttft and ram_ttft > baseline_ttft * 2:
            insights.append("RAM-stress prompts substantially increase TTFT, consistent with larger context pressure.")

        baseline_cache = baseline["avg_cache_miss_rate"]
        ram_cache = ram_stress["avg_cache_miss_rate"]
        if ram_cache and baseline_cache and ram_cache > baseline_cache * 1.2:
            insights.append("RAM-stress prompts have 20%+ higher cache misses, suggesting working-set pressure.")

    if baseline and cpu_stress:
        baseline_ipc = baseline["avg_ipc"]
        cpu_ipc = cpu_stress["avg_ipc"]
        if cpu_ipc and baseline_ipc and cpu_ipc > baseline_ipc * 1.1:
            insights.append("CPU-stress prompts raise IPC, suggesting denser compute-side work.")

    return {"comparison": comparison, "insights": insights}


def generate_report(output_dir: Path, all_runs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Generate a human-readable bottleneck analysis report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LINUX BOTTLENECK ANALYSIS REPORT - LLM WORKLOAD")
    report_lines.append("=" * 80)
    report_lines.append("")

    for run in all_runs:
        diagnosis = run.get("bottleneck_diagnosis") or diagnose_bottleneck(
            run["perf_stat"],
            run["memory"],
            run["timing"],
        )
        if diagnosis["primary_bottleneck"]:
            confidence = diagnosis["confidence"] if diagnosis["confidence"] is not None else 0.0
            report_lines.append(f"Prompt {run['prompt_id']} (Run {run['run_index']}):")
            report_lines.append(f"  -> {diagnosis['primary_bottleneck']} (confidence: {confidence:.1%})")
            for evidence in diagnosis["evidence"][:2]:
                report_lines.append(f"    - {evidence}")
            report_lines.append(f"  -> Fix: {diagnosis['recommendation']}")
            report_lines.append("")

    category_analysis = compare_prompt_categories(all_runs)
    report_lines.append("-" * 40)
    report_lines.append("CATEGORY COMPARISON")
    report_lines.append("-" * 40)
    for category, stats in category_analysis["comparison"].items():
        report_lines.append(
            f"{category}: runs={stats['runs']}, "
            f"avg_cache_miss_rate={format_optional_percent(stats['avg_cache_miss_rate'])}, "
            f"avg_ipc={format_optional_float(stats['avg_ipc'])}, "
            f"avg_ttft={format_optional_seconds(stats['avg_ttft'])}"
        )
    for insight in category_analysis["insights"]:
        report_lines.append(f"- {insight}")

    report_lines.append("")
    report_lines.append("-" * 40)
    report_lines.append("STATISTICAL SUMMARY")
    report_lines.append("-" * 40)

    for prompt_id, stats in summary.items():
        report_lines.append(f"\n{prompt_id} (CV - lower is more stable):")
        for metric, cv in stats["coefficient_of_variation"].items():
            if cv is not None and cv < 0.15:
                report_lines.append(f"  [stable] {metric}: CV={cv:.2f} (stable bottleneck)")
            elif cv is not None and cv > 0.30:
                report_lines.append(f"  [variable] {metric}: CV={cv:.2f} (high variability - scheduler noise)")

    report_path = output_dir / "BOTTLENECK_REPORT.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nBottleneck report written to {report_path}")


def format_optional_float(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else f"{number:.3f}"


def format_optional_percent(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else f"{number:.1%}"


def format_optional_seconds(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else f"{number:.3f}s"


def should_record_flamegraph(mode: str, run_index: int) -> bool:
    return mode == "all" or (mode == "first" and run_index == 1)


def run_one(
    args: argparse.Namespace,
    prompt: dict[str, Any],
    run_index: int,
    pid: int,
    run_dir: Path,
) -> dict[str, Any]:
    prompt_id = prompt["id"]
    perf_stat_path = run_dir / "perf_stat.csv"
    perf_data_path = run_dir / "perf.data"
    flamegraph_path = run_dir / "flamegraph.svg"
    run_dir.mkdir(parents=True, exist_ok=True)

    record_proc = None
    if should_record_flamegraph(args.flamegraphs, run_index):
        record_proc = start_perf_record(args, pid, perf_data_path, args.perf_frequency)
        time.sleep(args.perf_warmup_seconds)

    stat_proc = start_perf_stat(args, pid, perf_stat_path, args.timeout + 5)
    memory = MemoryTrace()
    memory.start(pid, args.memory_interval)

    error = None
    result: dict[str, Any] = {}
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    try:
        result = post_ollama_stream(
            args.server,
            args.model,
            prompt["prompt"],
            args.num_predict,
            args.timeout,
        )
    except Exception as exc:
        error = repr(exc)
    finally:
        memory_summary = memory.stop()
        stop_process_group(stat_proc)
        if record_proc is not None:
            stop_process_group(record_proc)

    wall_time_seconds = time.perf_counter() - wall_start
    perf_stat = parse_perf_stat(perf_stat_path)
    flamegraph = {"generated": False}
    if record_proc is not None and perf_data_path.exists():
        try:
            flamegraph = generate_flamegraph(args, perf_data_path, flamegraph_path, args.flamegraph_dir)
        except Exception as exc:
            flamegraph = {"generated": False, "reason": repr(exc)}

    timings = {
        "time_to_first_token_seconds": result.get("time_to_first_token_seconds"),
        "tokens_per_second": result.get("tokens_per_second"),
        "total_execution_seconds": result.get("total_execution_seconds") or wall_time_seconds,
        "wall_time_seconds": wall_time_seconds,
    }
    blocked_time = derive_blocked_time(wall_time_seconds, perf_stat, args.cpu_frequency_hz)
    bottleneck_diagnosis = diagnose_bottleneck(perf_stat, memory_summary, timings)
    return {
        "prompt_id": prompt_id,
        "category": prompt.get("category"),
        "title": prompt.get("title"),
        "run_index": run_index,
        "started_at": started_at,
        "model_metadata": {
            "model": args.model,
            "server": args.server,
            "ollama_pid": pid,
            "num_predict": args.num_predict,
        },
        "prompt_metrics": prompt.get("metrics", {}),
        "response_content": result.get("response", ""),
        "timing": timings,
        "ollama_metrics": result.get("ollama_metrics", {}),
        "memory": memory_summary,
        "perf_stat": perf_stat,
        "blocked_time": blocked_time,
        "bottleneck_diagnosis": bottleneck_diagnosis,
        "perf_artifacts": {
            "perf_stat_csv": str(perf_stat_path),
            "perf_data": str(perf_data_path) if perf_data_path.exists() else None,
            "flamegraph": flamegraph,
        },
        "error": error,
    }


def write_perf_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = [
        "prompt_id",
        "category",
        "run_index",
        "time_to_first_token_seconds",
        "tokens_per_second",
        "total_execution_seconds",
        "wall_time_seconds",
        "task-clock",
        "cpu_time_seconds",
        "blocked_time_seconds",
        "percent_blocked",
        "cpu_frequency_hz",
        "peak_rss_bytes",
        "average_rss_bytes",
        "rss_variance",
        "cycles",
        "instructions",
        "ipc",
        "cache-misses",
        "cache_miss_rate",
        "branch-misses",
        "branch_miss_rate",
        "dTLB-load-misses",
        "dtlb_load_miss_rate",
        "iTLB-load-misses",
        "itlb_load_miss_rate",
        "context-switches",
        "cpu-migrations",
        "page-faults",
        "minor-faults",
        "major-faults",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {
                "prompt_id": run["prompt_id"],
                "category": run["category"],
                "run_index": run["run_index"],
                **run["timing"],
                "cpu_time_seconds": run["blocked_time"].get("cpu_time_seconds"),
                "blocked_time_seconds": run["blocked_time"].get("blocked_time_seconds"),
                "percent_blocked": run["blocked_time"].get("percent_blocked"),
                "cpu_frequency_hz": run["blocked_time"].get("cpu_frequency_hz"),
                "peak_rss_bytes": run["memory"].get("peak_rss_bytes"),
                "average_rss_bytes": run["memory"].get("average_rss_bytes"),
                "rss_variance": run["memory"].get("rss_variance"),
            }
            for key in fields:
                if key in run["perf_stat"]:
                    row[key] = run["perf_stat"][key]
            writer.writerow(row)


def build_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["prompt_id"], []).append(run)

    summary = {}
    for prompt_id, prompt_runs in grouped.items():
        metric_values = {
            "time_to_first_token_seconds": [
                run["timing"].get("time_to_first_token_seconds") for run in prompt_runs
            ],
            "tokens_per_second": [run["timing"].get("tokens_per_second") for run in prompt_runs],
            "total_execution_seconds": [
                run["timing"].get("total_execution_seconds") for run in prompt_runs
            ],
            "blocked_time_seconds": [
                run["blocked_time"].get("blocked_time_seconds") for run in prompt_runs
            ],
            "percent_blocked": [run["blocked_time"].get("percent_blocked") for run in prompt_runs],
            "peak_rss_bytes": [run["memory"].get("peak_rss_bytes") for run in prompt_runs],
            "ipc": [run["perf_stat"].get("ipc") for run in prompt_runs],
            "cache_miss_rate": [run["perf_stat"].get("cache_miss_rate") for run in prompt_runs],
            "branch_miss_rate": [run["perf_stat"].get("branch_miss_rate") for run in prompt_runs],
            "dtlb_load_miss_rate": [
                run["perf_stat"].get("dtlb_load_miss_rate") for run in prompt_runs
            ],
            "context-switches": [run["perf_stat"].get("context-switches") for run in prompt_runs],
        }
        summary[prompt_id] = {
            "runs": len(prompt_runs),
            "category": prompt_runs[0].get("category"),
            "coefficient_of_variation": {
                metric: coefficient_of_variation([as_float(value) for value in values])
                for metric, values in metric_values.items()
            },
        }
    return summary


def run(args: argparse.Namespace) -> None:
    if psutil is None:
        raise SystemExit("psutil is required. Run with ./llm_env/bin/python Benchmark.py or install psutil.")
    if args.runs < 4 or args.runs > 8:
        raise SystemExit("--runs must be between 4 and 8")
    if args.cpu_frequency_hz is None:
        args.cpu_frequency_hz = estimate_cpu_frequency_hz()
    validate_sudo_for_perf(args)

    prompts, prompt_metadata, prompt_path = load_prompts(args.prompts)
    if args.prompt_id:
        wanted = set(args.prompt_id)
        prompts = [prompt for prompt in prompts if prompt["id"] in wanted]
        missing = wanted - {prompt["id"] for prompt in prompts}
        if missing:
            raise SystemExit(f"Prompt IDs not found: {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    restart_ollama(args)
    ollama_proc = start_ollama(args)
    try:
        wait_for_ollama(args.server, args.ollama_ready_timeout, args.ollama_poll_interval)
        if args.warmup:
            run_warmup(args)

        pid = args.pid or find_ollama_pid()

        all_runs = []
        for prompt in prompts:
            for run_index in range(1, args.runs + 1):
                print(f"Running {prompt['id']} repetition {run_index}/{args.runs}", flush=True)
                run_dir = args.output_dir / prompt["id"] / f"run_{run_index:02d}"
                all_runs.append(run_one(args, prompt, run_index, pid, run_dir))

        summary = build_summary(all_runs)
        category_comparison = compare_prompt_categories(all_runs)
        output = {
            "benchmark_metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "prompt_file": str(prompt_path),
                "prompt_metadata": prompt_metadata,
                "runs_per_prompt": args.runs,
                "perf_events": PERF_EVENTS,
                "cv_note": "Lower CV suggests stable bottlenecks; high CV suggests scheduler or OS noise.",
            },
            "summary": summary,
            "category_comparison": category_comparison,
            "runs": all_runs,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        write_perf_csv(args.csv_output, all_runs)
        generate_report(args.output_dir, all_runs, summary)

        print(f"Wrote JSON log to {args.json_output}")
        print(f"Wrote perf/timing CSV to {args.csv_output}")
    finally:
        if args.stop_ollama_after:
            stop_ollama(ollama_proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Linux bottlenecks under Ollama LLM inference workloads."
    )
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-id", action="append", help="Run only this prompt ID. Repeatable.")
    parser.add_argument("--runs", type=int, default=8, help="Repetitions per prompt, from 4 to 8.")
    parser.add_argument("--server", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="tinyllama")
    parser.add_argument("--start-ollama", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart-ollama", action="store_true")
    parser.add_argument("--ollama-command", default=DEFAULT_OLLAMA_COMMAND)
    parser.add_argument("--ollama-ready-timeout", type=float, default=60.0)
    parser.add_argument("--ollama-restart-timeout", type=float, default=15.0)
    parser.add_argument("--ollama-poll-interval", type=float, default=0.5)
    parser.add_argument("--stop-ollama-after", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-prompt", default=DEFAULT_WARMUP_PROMPT)
    parser.add_argument("--warmup-num-predict", type=int, default=4)
    parser.add_argument("--pid", type=int, help="Ollama server PID. Auto-detected if omitted.")
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--memory-interval", type=float, default=0.05)
    parser.add_argument(
        "--cpu-frequency-hz",
        type=float,
        help="CPU frequency in Hz for blocked-time estimates. Auto-detected if omitted.",
    )
    parser.add_argument("--perf-frequency", type=int, default=99)
    parser.add_argument("--perf-warmup-seconds", type=float, default=0.2)
    parser.add_argument("--sudo-perf", action="store_true", help="Run perf commands through sudo.")
    parser.add_argument(
        "--flamegraphs",
        choices=["first", "all", "none"],
        default="first",
        help="Record CPU samples for the first run of each prompt, every run, or none.",
    )
    parser.add_argument("--flamegraph-dir", type=Path, default=Path("FlameGraph"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--json-output", type=Path, default=OUTPUT_DIR / "benchmark_results.json")
    parser.add_argument("--csv-output", type=Path, default=OUTPUT_DIR / "perf_metrics.csv")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
