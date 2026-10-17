"""Serve a trained choice-memory checkpoint with the existing XPolicyLab server.

This is the native policy server. It imports the shared communication server
directly, so it needs no ``XPolicyLab/policy/<name>/model.py`` adapter package
and no edit to the read-only XPolicyLab checkout (see eval_rmbench.py for how a
``policy_name`` label is still required by the shared eval client).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from choice_mem.config import SUPPORTED_CAMERAS, validate_cameras
from choice_mem.runtime import DEFAULT_RMBENCH_ROOT, ModelWrapper, add_native_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--allow-debug-checkpoint", action="store_true",
        help="Explicitly allow a random-frozen debug policy, never a pretrained benchmark result",
    )
    parser.add_argument("--rmbench-root", type=Path, default=DEFAULT_RMBENCH_ROOT)
    parser.add_argument("--device", default="cpu", help="Explicit model device; no GPU is selected automatically")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1", help="Bind privately; use a trusted network for legacy TCP")
    parser.add_argument("--port", type=int, default=18820)
    parser.add_argument("--protocol", choices=("ws", "legacy_tcp"), default="ws")
    parser.add_argument("--action-type", default="joint")
    parser.add_argument("--horizon", type=int, help="Assert checkpoint value, never override it")
    parser.add_argument("--execute-prefix", type=int, help="Assert checkpoint value, never override it")
    parser.add_argument(
        "--cameras", nargs="+", choices=SUPPORTED_CAMERAS,
        help="Assert the checkpoint's ordered one-to-three camera list; never override trained views",
    )
    parser.add_argument("--image-size", nargs=2, type=int, help="Assert checkpoint height and width")
    parser.add_argument("--target-offset", type=int, help="Assert the checkpoint's trained label offset")
    parser.add_argument("--label-alignment", help="Assert the checkpoint's stored training-label semantics")
    arguments = parser.parse_args(argv)
    if arguments.cameras is not None:
        validate_cameras(arguments.cameras)
    if not 1 <= arguments.port <= 65535:
        parser.error("port must be in [1,65535]")
    if arguments.action_type != "joint":
        parser.error("only action-type=joint is supported")
    if arguments.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    return arguments


def main(argv=None) -> None:
    arguments = parse_args(argv)
    import torch

    torch.set_num_threads(arguments.cpu_threads)
    if str(arguments.device).startswith("cuda"):
        torch.cuda.set_device(arguments.device)
    add_native_paths(arguments.rmbench_root)
    expected = {
        field_name: getattr(arguments, field_name)
        for field_name in ("horizon", "execute_prefix", "cameras", "image_size")
        if getattr(arguments, field_name) is not None
    }
    expected_protocol = {
        field_name: getattr(arguments, field_name)
        for field_name in ("target_offset", "label_alignment")
        if getattr(arguments, field_name) is not None
    } or None
    model = ModelWrapper(
        arguments.checkpoint,
        arguments.device,
        expected=expected,
        action_type=arguments.action_type,
        expected_protocol=expected_protocol,
        allow_debug_checkpoint=arguments.allow_debug_checkpoint,
    )
    contract = model.protocol_summary()
    print(
        f"Serving trained checkpoint {arguments.checkpoint.resolve()} via native XPolicyLab "
        f"{arguments.protocol} on {arguments.host}:{arguments.port}; "
        f"horizon={contract['horizon']}, execute_prefix={contract['execute_prefix']}, "
        f"target_offset={contract['target_offset']!r}, "
        f"gripper_binarization={contract['gripper_binarization']}, "
        f"label_alignment={contract['label_alignment']!r}, "
        f"backbone_initialization={contract['backbone_initialization']}, "
        f"backbone_initialization_seed={contract['backbone_initialization_seed']}",
        flush=True,
    )
    if arguments.protocol == "ws":
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig

        server = PolicyServer(model, PolicyServerConfig(host=arguments.host, port=arguments.port))
        try:
            asyncio.run(server.serve_forever())
        except KeyboardInterrupt:
            pass
    else:
        from client_server.tcp.model_server import ModelServer

        server = ModelServer(model, host=arguments.host, port=arguments.port)
        try:
            server.start()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()


if __name__ == "__main__":
    main()
