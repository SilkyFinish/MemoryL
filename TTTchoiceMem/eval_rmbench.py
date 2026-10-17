"""Run exactly one episode through RMBench's existing native evaluation client.

The observation adaptation adds the real action counter in this child process.
GPU visibility follows the native single-device renderer convention. Simulator
and task sources and rollout rules stay unchanged; the server is isolated.

Native ``policy_name`` handling: RMBench's eval entrypoint only needs a
``policy_name`` string as a result-directory/log label. It resolves the policy
over the shared websocket (or legacy TCP) client and never imports
``XPolicyLab/policy/<policy_name>/model.py`` -- that import belongs solely to
the shared ``XPolicyLab/setup_policy_server.py``, which this adapter does not use.
So there is deliberately no ``XPolicyLab/policy/TTTchoiceMem`` package, and the
read-only XPolicyLab checkout is never modified. The matching native server is
``serve_rmbench.py``, which imports ``client_server`` directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from datetime import datetime
from numbers import Integral
from pathlib import Path

DEFAULT_RMBENCH_ROOT = Path("/mnt/data/dylan/d2/memory/RMBench")


def install_timestep_adapter(native_evaluator) -> None:
    """Augment native RGB/state conversion, never infer time from RPC query IDs."""
    native_converter = native_evaluator.robotwin_obs_to_xpolicylab

    def convert_with_timestep(observation, **kwargs):
        task_env = kwargs.get("task_env")
        timestep = getattr(task_env, "take_action_cnt", None)
        if isinstance(timestep, bool) or not isinstance(timestep, Integral) or timestep < 0:
            raise ValueError("Native task must expose its actual take_action_cnt")
        converted = native_converter(observation, **kwargs)
        converted["timestep"] = int(timestep)
        return converted

    native_evaluator.robotwin_obs_to_xpolicylab = convert_with_timestep


def load_native_evaluator(rmbench_root: Path):
    source_path = rmbench_root / "scripts" / "eval_policy_xpolicylab.py"
    for directory in (rmbench_root, rmbench_root / "scripts", rmbench_root / "XPolicyLab"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    module_spec = importlib.util.spec_from_file_location("choice_memory_native_rmbench_eval", source_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load native evaluator: {source_path}")
    native_evaluator = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(native_evaluator)
    install_timestep_adapter(native_evaluator)
    return native_evaluator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmbench-root", type=Path, default=DEFAULT_RMBENCH_ROOT)
    parser.add_argument("--task-name", default="battery_try", help="An existing native RMBench task, not a new task")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument(
        "--env-python", default=sys.executable, help="Python in the already prepared simulator environment"
    )
    parser.add_argument(
        "--sim-gpu",
        type=int,
        choices=range(8),
        default=7,
        help="Physical GPU exposed as cuda:0 to the single-GPU SAPIEN renderer",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18820)
    parser.add_argument("--protocol", choices=("ws",), default="ws", help="Audited current native client protocol")
    parser.add_argument("--action-type", choices=("joint",), default="joint")
    parser.add_argument("--episodes", type=int, choices=(1,), default=1)
    parser.add_argument("--seed", type=int, default=0, help="Native seed group; first seed is 100000*(1+seed)+offset")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--max-seed-attempts", type=int, default=1, help="Bound native setup/expert retries; never retry policy outcomes"
    )
    parser.add_argument(
        "--skip-expert-check", action="store_true", help="Opt out explicitly; native expert checking remains the default"
    )
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without importing or running a simulator"
    )
    parser.add_argument("--native-child", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    arguments.rmbench_root = arguments.rmbench_root.expanduser().resolve()
    if not 1 <= arguments.port <= 65535 or arguments.seed < 0 or arguments.seed_offset < 0:
        parser.error("require a valid port and nonnegative seed/seed-offset")
    if arguments.max_seed_attempts < 1:
        parser.error("max-seed-attempts must be positive")
    # Validate names as single path components before selecting existing sources.
    if not arguments.task_name.isidentifier() or Path(arguments.task_config).name != arguments.task_config:
        parser.error("task-name and task-config must identify existing native files")
    for required_file in (
        arguments.rmbench_root / "scripts" / "eval_policy_xpolicylab.py",
        arguments.rmbench_root / "envs" / f"{arguments.task_name}.py",
        arguments.rmbench_root / "env_cfg" / "task_config" / f"{arguments.task_config}.yml",
    ):
        if not required_file.is_file():
            parser.error(f"missing native benchmark file: {required_file}")
    if arguments.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        arguments.output_dir = (
            Path(__file__).resolve().parent / "outputs" / f"rmbench_single_{arguments.task_name}_{timestamp}"
        )
    arguments.output_dir = arguments.output_dir.expanduser().absolute()
    if arguments.output_dir.exists():
        parser.error(f"refusing to reuse an existing evaluation output directory: {arguments.output_dir}")
    return arguments


def build_native_command(arguments) -> list[str]:
    command = [
        arguments.env_python,
        str(Path(__file__).resolve()),
        "--native-child",
        "--rmbench-root",
        str(arguments.rmbench_root),
        "--task-name",
        arguments.task_name,
        "--task-config",
        arguments.task_config,
        "--sim-gpu",
        str(arguments.sim_gpu),
        "--host",
        arguments.host,
        "--port",
        str(arguments.port),
        "--protocol",
        arguments.protocol,
        "--action-type",
        arguments.action_type,
        "--episodes",
        "1",
        "--seed",
        str(arguments.seed),
        "--seed-offset",
        str(arguments.seed_offset),
        "--max-seed-attempts",
        str(arguments.max_seed_attempts),
        "--instruction-type",
        arguments.instruction_type,
        "--output-dir",
        str(arguments.output_dir),
    ]
    if arguments.skip_expert_check:
        command.append("--skip-expert-check")
    return command


def run_native_child(arguments) -> None:
    # SAPIEN caches one rendering device per process. Use the native launcher's
    # single-visible-GPU convention before importing any CUDA/simulator module.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.sim_gpu)
    os.environ["RMBENCH_RENDER_DEVICE"] = "cuda:0"
    # websockets 15 honors ambient proxies. Only this child bypasses them for
    # the explicitly selected policy host; preserve all other proxy settings.
    for variable_name in ("NO_PROXY", "no_proxy"):
        bypass_hosts = os.environ.get(variable_name, "").split(",")
        bypass_hosts.extend(["localhost", "127.0.0.1", "::1", arguments.host])
        os.environ[variable_name] = ",".join(dict.fromkeys(host for host in bypass_hosts if host))
    native_evaluator = load_native_evaluator(arguments.rmbench_root)
    # Keep the native entrypoint's renderer readiness check, without altering it.
    from test_render import Sapien_TEST

    Sapien_TEST()
    native_evaluator.main(
        {
            "bench_name": "RMBench",
            "task_name": arguments.task_name,
            "task_config": arguments.task_config,
            "policy_name": "TTTchoiceMem",
            "ckpt_setting": "trained-choice-memory",
            "env_cfg_type": "aloha_agilex",
            "action_type": "joint",
            "host": arguments.host,
            "port": arguments.port,
            "protocol": arguments.protocol,
            "xpolicylab_root": str(arguments.rmbench_root / "XPolicyLab"),
            "root_dir": str(arguments.rmbench_root),
            "seed": arguments.seed,
            "seed_offset": arguments.seed_offset,
            "seed_stride": 1,
            "test_num": 1,
            "eval_batch": False,
            "num_workers": 1,
            "max_seed_attempts": arguments.max_seed_attempts,
            "expert_check": not arguments.skip_expert_check,
            "instruction_type": arguments.instruction_type,
            "eval_output_dir": arguments.output_dir,
            "episode_log": arguments.output_dir / "episodes.jsonl",
        }
    )


def main(argv=None) -> None:
    arguments = parse_args(argv)
    command = build_native_command(arguments)
    if arguments.dry_run:
        print(shlex.join(command))
        return
    if arguments.native_child:
        run_native_child(arguments)
    else:
        print(f"Native one-episode client: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=arguments.rmbench_root, check=True)


if __name__ == "__main__":
    main()
