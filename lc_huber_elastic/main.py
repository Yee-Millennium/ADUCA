"""Batch experiment driver for the LC-Huber benchmark.

This script is intentionally similar to the experiment drivers used in:
  * Nash_Equilibrium/main.py
  * mm_cournot/main.py

It launches multiple independent runs of `run_algos.py` (different algorithms
and parameter variants) and writes:
  * JSON trajectory files to `output/traj/<timestamp>/...`
  * per-run log files to `output/log/<timestamp>/...`

Edit the *Configuration* section below to tune scenarios, algorithm variants,
and runtime budgets.
"""

from __future__ import annotations

import datetime
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore
from typing import Dict, List, Optional
import logging


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _format_param_tag(params: Dict) -> str:
    """Compact, filename-safe tag for a variant dict."""
    parts = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, float):
            parts.append(f"{k}-{v:g}")
        else:
            parts.append(f"{k}-{v}")
    return "-".join(parts) if parts else "default"


def _build_cmd(
    base: Dict,
    scenario_id: int,
    algo: str,
    variant: Dict,
    output_json: Path,
    device_override: Optional[str] = None,
) -> List[str]:
    cmd = ["python", "run_algos.py", "--scenario", str(scenario_id), "--algo", algo]

    # scenario-level arguments
    if "seed" in base:
        cmd += ["--seed", str(base["seed"])]

    # base parameters
    base_device = base.get("device")
    for k, v in base.items():
        if k in {"seed", "device"}:
            continue
        if v is None:
            continue
        cmd += [f"--{k}", str(v)]

    # variant parameters (override base if collision)
    variant_device = variant.get("device")
    for k, v in variant.items():
        if k == "device":
            continue
        cmd += [f"--{k}", str(v)]

    device_value = device_override
    if device_value is None:
        device_value = variant_device if variant_device is not None else base_device
    if device_value is not None:
        cmd += ["--device", str(device_value)]

    cmd += ["--outputdir", str(output_json)]
    return cmd


def _run_one(cmd: List[str], log_file: Path, env: Optional[Dict[str, str]] = None, gpu_id: Optional[str] = None) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(shlex.join(cmd) + "\n\n")
        if gpu_id is not None:
            f.write(f"ASSIGNED_GPU: {gpu_id}\n")
        if env is not None and "CUDA_VISIBLE_DEVICES" in env:
            f.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
        if gpu_id is not None or (env is not None and "CUDA_VISIBLE_DEVICES" in env):
            f.write("\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
        return int(p.returncode)


def _parse_cuda_visible_devices(value: Optional[object]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).split(",")
    gpu_ids = [str(item).strip() for item in items if str(item).strip()]
    return gpu_ids


def _is_cuda_device(device: Optional[object]) -> bool:
    if device is None:
        return False
    return str(device).lower().startswith("cuda")


def main() -> None:
    # ==================================================================
    # Configuration
    # ==================================================================

    # Choose which scenario IDs (see src/problems/lc_huber_instance.py)
    # you want to run. The default set includes the dense ADUCA-favorable
    # scenarios and their easy variants.

    scenarios: List[int] = []
    # scenarios: List[int] = [1,2]

    cuda_visible_devices = ""
    # Maximum concurrent jobs allowed on a single GPU.
    jobs_per_gpu = 1

    # A single global seed (you can also run multiple seeds by expanding the
    # run list below).
    seed = 33

    # Algorithms to run. To focus on "rescaled" comparisons, keep only the
    # *_normalized variants plus ADUCA_TORCH.
    algorithms: List[str] = [
        "ADUCA_TORCH",
        "GR",
        "CODER_linesearch",
        "PCCM",
        "CODER",
    ]

    # Base runtime/accuracy settings shared across all runs.
    base_params: Dict = {
        "seed": seed,
        "maxiter": 10_100_000,
        "maxtime": 200000,
        "targetaccuracy": 1e-10,
        "loggingfreq": 10,
        # Block sizes (can be overridden per-algorithm or per-scenario)
        "block_size_u": 100,
        "block_size_v": 50,
        # Torch settings
        "device": "cuda:0",
        "dtype": "float64",
        # Preconditioner used by *_normalized and ADUCA_TORCH; can be "identity" or "diag_lipschitz"
        "preconditioner": "identity",
        # Opt-measure for logging/stopping: "prox_residual" or "projected_primal_gap"
        "opt_measure": "prox_residual",
        # Lipschitz scaling for fixed-L methods (used if --lipschitz not given)
        "lipschitz_mult": 1.0,
        "lambda1": 1e-4,  # \ell_1 regularization weight on u
        "lambda2": 1e-4,  # \ell_2 regularization weight on u
    }

    # Parameter variants per algorithm.
    # You can add/remove variants to tune performance.
    algo_variants: Dict[str, List[Dict]] = {
        "ADUCA_TORCH": [
            {"beta": 0.7, "gamma": 0.1, "rho": 1.3, "preconditioner": "identity"},
            # {"beta": 0.8, "gamma": 0.2, "rho": 1.2, "preconditioner": "diag_lipschitz"},
        ],
        "GR": [
            {"beta": 0.7, },
        ],
    }

    scen_params: Dict = {
        # Scenario-specific parameter overrides can go here.
        # Baseline (0) + EN-1..EN-8 block sizes and elastic-net params
        # from lc_huber_elastic/src/problems/lc_huber_instance.py
        # 0: {"block_size_u": 10, "block_size_v": 5, "lambda1": 1e-3, "lambda2": 1e-2},
        1: {"block_size_u": 160, "block_size_v": 40},  # blocks: u=50 v=20 total=70
        2: {"block_size_u": 240, "block_size_v": 60},  # blocks: u=50 v=20 total=70
        3: {"block_size_u": 60, "block_size_v": 24},  # blocks: u=40 v=10 total=50
        4: {"block_size_u": 80, "block_size_v": 40},  # blocks: u=50 v=10 total=60
        5: {"block_size_u": 75, "block_size_v": 30},  # blocks: u=40 v=10 total=50
        6: {"block_size_u": 50, "block_size_v": 25},  # blocks: u=50 v=10 total=60
        7: {"block_size_u": 45, "block_size_v": 18},  # blocks: u=40 v=10 total=50
        8: {"block_size_u": 30, "block_size_v": 15},  # blocks: u=50 v=10 total=60
    }

    # Parallelism. Set to 1 for a single GPU.
    # If None, this defaults to (#gpus * jobs_per_gpu) when GPUs are configured.
    max_workers = None

    # ==================================================================
    # Output directories
    # ==================================================================
    stamp = _timestamp()
    out_traj_dir = Path("output") / "traj" / stamp
    out_log_dir = Path("output") / "log" / stamp
    out_traj_dir.mkdir(parents=True, exist_ok=True)
    out_log_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Build job list
    # ==================================================================
    gpu_ids = _parse_cuda_visible_devices(cuda_visible_devices)
    if gpu_ids and jobs_per_gpu < 1:
        raise ValueError("jobs_per_gpu must be >= 1 when using GPUs.")

    jobs = []
    gpu_job_idx = 0
    for scen in scenarios:
        # Scenario overrides (e.g., block sizes) take precedence over base_params.
        scenario_base = {**base_params, **scen_params.get(scen, {})}
        scenario_seed = scenario_base.get("seed", seed)
        for algo in algorithms:
            variants = algo_variants.get(algo, [{}])
            for var in variants:
                tag = _format_param_tag(var)
                json_name = f"scenario-{scen}-{algo}-{tag}-seed-{scenario_seed}.json"
                output_json = out_traj_dir / json_name
                log_file = out_log_dir / (json_name.replace(".json", ".log"))

                desired_device = var.get("device", scenario_base.get("device"))
                gpu_id = None
                device_override = None
                if gpu_ids and _is_cuda_device(desired_device):
                    gpu_id = gpu_ids[gpu_job_idx % len(gpu_ids)]
                    device_override = "cuda:0"
                    gpu_job_idx += 1

                cmd = _build_cmd(scenario_base, scen, algo, var, output_json, device_override=device_override)
                jobs.append((cmd, log_file, gpu_id))

    # ==================================================================
    # Run
    # ==================================================================
    print(f"Launching {len(jobs)} runs...")

    failures = 0
    auto_workers = max_workers
    if auto_workers is None:
        if gpu_ids:
            auto_workers = min(len(jobs), len(gpu_ids) * jobs_per_gpu)
        else:
            auto_workers = len(jobs)

    base_env = os.environ.copy()
    if cuda_visible_devices is not None:
        base_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids) if gpu_ids else str(cuda_visible_devices)

    gpu_semaphores = {gpu_id: Semaphore(jobs_per_gpu) for gpu_id in gpu_ids}

    def _run_with_limits(cmd: List[str], log: Path, gpu_id: Optional[str]) -> int:
        env = base_env.copy()
        if gpu_id is None:
            return _run_one(cmd, log, env=env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        semaphore = gpu_semaphores[gpu_id]
        semaphore.acquire()
        try:
            return _run_one(cmd, log, env=env, gpu_id=gpu_id)
        finally:
            semaphore.release()

    with ThreadPoolExecutor(max_workers=auto_workers) as ex:
        fut_to_job = {
            ex.submit(_run_with_limits, cmd, log, gpu_id): (cmd, log, gpu_id)
            for cmd, log, gpu_id in jobs
        }
        for fut in as_completed(fut_to_job):
            cmd, log, gpu_id = fut_to_job[fut]
            try:
                rc = fut.result()
            except Exception as e:  # noqa: BLE001
                rc = 999
                print(f"FAILED (exception): {e}")
            if rc != 0:
                failures += 1
                print("FAILED:")
                print("  ", shlex.join(cmd))
                print("  log:", log)

    if failures == 0:
        print("All runs finished successfully.")
    else:
        print(f"Finished with {failures} failed run(s). See log files under {out_log_dir}.")


if __name__ == "__main__":
    main()
