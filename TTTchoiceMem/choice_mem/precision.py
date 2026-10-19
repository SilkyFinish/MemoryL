"""Training parameter storage; autocast is configured by the training loop."""

import torch

from .config import debug_parameter_dtype, is_debug_initialization


def prepare_training_precision(policy):
    configuration = getattr(policy, "config", None)
    initialization = getattr(configuration, "backbone_initialization", "pretrained")
    if not is_debug_initialization(initialization):
        return policy.float()
    if not configuration.freeze_backbone or configuration.dtype != "bfloat16":
        raise ValueError("Random-frozen debug requires a frozen BF16 backbone")
    if any(parameter.requires_grad for parameter in policy.backbone.parameters()):
        raise ValueError("Random-frozen debug backbone must already be frozen")

    if initialization == "turbo-random-frozen-debug" and not isinstance(
        getattr(policy.backbone, "vision_language_interaction", None), torch.nn.Module
    ):
        raise ValueError("Turbo debug requires the explicit FP32 vision_language_interaction module")

    # Do not transiently expand the frozen visual/text towers to FP32, or round
    # the FP32 interaction parameters through BF16 before converting them back.
    for name, module in policy.named_children():
        if name == "backbone":
            if initialization == "turbo-random-frozen-debug":
                for child_name, child in module.named_children():
                    dtype = debug_parameter_dtype("backbone." + child_name + ".", initialization)
                    child.to(dtype=getattr(torch, dtype))
            else:
                module.to(dtype=torch.bfloat16)
            module.eval()
        else:
            module.float()
    for name, parameter in policy.named_parameters():
        expected_dtype = getattr(torch, debug_parameter_dtype(name, initialization))
        if parameter.dtype != expected_dtype:
            raise ValueError(f"Unexpected training parameter dtype: {name}={parameter.dtype}")
    return policy
