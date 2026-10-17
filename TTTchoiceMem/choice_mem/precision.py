"""Training parameter storage; autocast is configured by the training loop."""

import torch


def prepare_training_precision(policy):
    configuration = getattr(policy, "config", None)
    if getattr(configuration, "backbone_initialization", "pretrained") != "random-frozen-debug":
        return policy.float()
    if not configuration.freeze_backbone or configuration.dtype != "bfloat16":
        raise ValueError("Random-frozen debug requires a frozen BF16 backbone")
    if any(parameter.requires_grad for parameter in policy.backbone.parameters()):
        raise ValueError("Random-frozen debug backbone must already be frozen")

    # Do not transiently expand the entire frozen VLM to FP32 on the GPU.
    for name, module in policy.named_children():
        if name == "backbone":
            module.to(dtype=torch.bfloat16).eval()
        else:
            module.float()
    for name, parameter in policy.named_parameters():
        expected_dtype = torch.bfloat16 if name.startswith("backbone.") else torch.float32
        if parameter.dtype != expected_dtype:
            raise ValueError(f"Unexpected training parameter dtype: {name}={parameter.dtype}")
    return policy
