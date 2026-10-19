"""Narrow, online TurboVLA text/interaction core; no vision/action dependencies.

Ported and modified from TurboVLA commit
6727c875666f8d5dda8d8cca0043da200738fe73: models/turbovla.py,
models/components/{fusion,transformer}.py, models/text_encoder.py and text/bert.py.
Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
Copyright (c) 2023 IDEA. All Rights Reserved.
Copyright (c) Aishwarya Kamath & Nicolas Carion. All Rights Reserved.
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
Licensed under Apache-2.0; see TURBO_MODULES_LICENSE in the project root.

Changes: retain only normalized-residual manual attention; fix batch/head mask
ordering, enforce key padding, isolate padding queries, and keep interaction
storage/computation FP32. BERT is standard Hugging Face BERT, with strictly local
assets, eager attention, batch-invariant delimiter segmentation and no caching.
"""

from copy import deepcopy
from pathlib import Path
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class _DropPath(nn.Module):
    """Per-example stochastic depth, matching timm's scale-by-keep default."""

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, value):
        if not self.training or self.drop_prob == 0:
            return value
        keep = 1 - self.drop_prob
        gate = value.new_empty((value.shape[0],) + (1,) * (value.ndim - 1)).bernoulli_(keep)
        return value * gate.div_(keep)


class _BiMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.attention_backend = "manual"
        # These names intentionally match the GroundingDINO/Turbo subset.
        self.v_proj = nn.Linear(hidden_dim, embed_dim)
        self.l_proj = nn.Linear(hidden_dim, embed_dim)
        self.values_v_proj = nn.Linear(hidden_dim, embed_dim)
        self.values_l_proj = nn.Linear(hidden_dim, embed_dim)
        self.out_v_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_l_proj = nn.Linear(embed_dim, hidden_dim)
        for module in self.children():
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _shape(self, value):
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, v, l, attention_mask_l):
        query = self._shape(self.v_proj(v)) * self.scale
        key = self._shape(self.l_proj(l))
        value_v = self._shape(self.values_v_proj(v))
        value_l = self._shape(self.values_l_proj(l))
        scores = query @ key.transpose(-1, -2)
        # A common scalar shift preserves BOTH directions' softmax. Unlike the
        # upstream global maximum, this excludes padded keys and other samples.
        key_padding = attention_mask_l[:, None, None, :]
        shift = scores.masked_fill(key_padding, -torch.inf).amax(dim=(-2, -1), keepdim=True)
        scores = (scores - shift).clamp(min=-50000, max=50000)
        reverse = scores.transpose(-1, -2)
        reverse = (reverse - reverse.amax(dim=-1, keepdim=True)).clamp(-50000, 50000)
        probs_l = F.dropout(reverse.softmax(-1), self.dropout, self.training)
        probs_v = F.dropout(scores.masked_fill(key_padding, -torch.inf).softmax(-1),
                            self.dropout, self.training)
        output_v = (probs_v @ value_l).transpose(1, 2).reshape(v.shape[0], v.shape[1], self.embed_dim)
        output_l = (probs_l @ value_v).transpose(1, 2).reshape(l.shape[0], l.shape[1], self.embed_dim)
        return self.out_v_proj(output_v), self.out_l_proj(output_l)


class _BiAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, embed_dim, num_heads, dropout, drop_path):
        super().__init__()
        self.layer_norm_v = nn.LayerNorm(hidden_dim)
        self.layer_norm_l = nn.LayerNorm(hidden_dim)
        self.attn = _BiMultiHeadAttention(hidden_dim, embed_dim, num_heads, dropout)
        self.drop_path = _DropPath(drop_path) if drop_path else nn.Identity()
        self.gamma_v = nn.Parameter(torch.full((hidden_dim,), 1e-4))
        self.gamma_l = nn.Parameter(torch.full((hidden_dim,), 1e-4))
        self.residual_style = "normalized"

    def forward(self, v, l, attention_mask_l):
        v, l = self.layer_norm_v(v), self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l)
        return (v + self.drop_path(self.gamma_v * delta_v),
                l + self.drop_path(self.gamma_l * delta_l))


def _safe_allowed_mask(allowed, key_padding):
    """Exclude padding keys for real queries, but let padding queries see self.

    The exception is query-local: a padded diagonal NEVER becomes visible to a
    real query. Keeping a padded self row avoids all-masked manual-softmax NaNs.
    """
    real_queries = ~key_padding[:, :, None]
    safe = allowed & real_queries & ~key_padding[:, None, :]
    if ((~safe.any(-1)) & ~key_padding).any().item():
        raise ValueError("every non-padding text query must allow a non-padding key")
    diagonal = torch.eye(allowed.shape[-1], dtype=torch.bool, device=allowed.device)
    return safe | (key_padding[:, :, None] & diagonal[None])


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, inner_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, inner_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(inner_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.nhead = num_heads

    def forward(self, src, src_mask, src_key_padding_mask):
        # src_mask uses torch's True=blocked convention, [B,T,T]. Merge padding
        # BEFORE head expansion; repeat(nhead,...) interleaves samples wrongly.
        allowed = _safe_allowed_mask(~src_mask, src_key_padding_mask)
        blocked = (~allowed).repeat_interleave(self.nhead, dim=0)
        # Padding is already merged per query. Passing key_padding_mask again
        # would block padded self diagonals and reintroduce all-masked rows.
        # need_weights=True selects the manual (not SDPA) torch MHA path.
        update = self.self_attn(src, src, src, attn_mask=blocked, need_weights=True)[0]
        src = self.norm1(src + self.dropout1(update))
        update = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(update))


class VisionLanguageInteraction(nn.Module):
    """Fixed Turbo interaction profile, with small dimensions allowed for tests.

    ``nheads`` is the configured count; the source uses max(1, nheads // 2).
    Defaults: six layers, hidden=256, four effective heads, inner=1024.
    Masks are mandatory bool tensors: allowed[B,T,T] (True=allowed) and
    key_padding[B,T] (True=padding). There is no state/cache across calls.
    """

    def __init__(self, hidden_dim=256, nheads=8, num_layers=6,
                 enhancer_inner_dim=1024, *, text_dropout=0.0,
                 fusion_dropout=0.0, fusion_droppath=0.1):
        super().__init__()
        dimensions = (hidden_dim, nheads, num_layers, enhancer_inner_dim)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("interaction dimensions, heads and layers must be positive integers")
        heads = max(1, nheads // 2)
        if hidden_dim % heads or enhancer_inner_dim % heads:
            raise ValueError("hidden and enhancer dimensions must be divisible by effective heads")
        for value in (text_dropout, fusion_dropout, fusion_droppath):
            if not 0 <= value < 1:
                raise ValueError("dropout/drop-path probabilities must lie in [0, 1)")
        self.hidden_dim = hidden_dim
        self.nheads, self.effective_heads = nheads, heads
        self.padding_strategy = "key_padding_mask"
        text = _TransformerEncoderLayer(hidden_dim, heads, enhancer_inner_dim, text_dropout)
        fusion = _BiAttentionBlock(hidden_dim, enhancer_inner_dim, heads,
                                   fusion_dropout, fusion_droppath)
        self.text_layers = nn.ModuleList(deepcopy(text) for _ in range(num_layers))
        self.fusion_layers = nn.ModuleList(deepcopy(fusion) for _ in range(num_layers))
        self.float()

    def _apply(self, fn, recurse=True):
        # Parent .to(dtype=...), .half() and .bfloat16() must not downcast this
        # subtree. Preserve the original FP32 values (do not cast back rounded
        # low-precision values); still honor the requested destination device.
        def keep_fp32(value):
            converted = fn(value)
            if converted.is_floating_point() and converted.dtype != torch.float32:
                return value.to(device=converted.device, dtype=torch.float32)
            return converted
        return super()._apply(keep_fp32, recurse=recurse)

    def _validate(self, visual, text, allowed, padding):
        for name, value in (("visual_tokens", visual), ("text_tokens", text)):
            if not isinstance(value, torch.Tensor) or value.ndim != 3 or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating [B,N,H] tensor")
            if not all(value.shape) or value.shape[-1] != self.hidden_dim:
                raise ValueError(f"{name} has an empty or incompatible shape")
        if visual.shape[0] != text.shape[0] or visual.device != text.device:
            raise ValueError("visual and text batches/devices must match")
        batch, length, _ = text.shape
        for name, value, shape in (("text_allowed_mask", allowed, (batch, length, length)),
                                    ("text_key_padding_mask", padding, (batch, length))):
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool
                    or tuple(value.shape) != shape or value.device != text.device):
                raise ValueError(f"{name} must be a bool tensor with shape {shape} on the token device")
        if padding.all(-1).any().item():
            raise ValueError("each sample requires at least one non-padding text token")
        if next(self.parameters()).device != visual.device:
            raise ValueError("interaction parameters and input tokens must be on the same device")

    def forward(self, visual_tokens, text_tokens, text_allowed_mask, text_key_padding_mask):
        self._validate(visual_tokens, text_tokens, text_allowed_mask, text_key_padding_mask)
        # Validate/normalize once, without ever modifying caller-owned masks.
        allowed = _safe_allowed_mask(text_allowed_mask, text_key_padding_mask)
        with torch.autocast(device_type=visual_tokens.device.type, enabled=False):
            visual, text = visual_tokens.float(), text_tokens.float()
            for fusion, text_layer in zip(self.fusion_layers, self.text_layers):
                visual, text = fusion(visual, text, text_key_padding_mask)
                text = text_layer(text.transpose(0, 1), ~allowed,
                                  text_key_padding_mask).transpose(0, 1)
        return visual, text


def _delimiter_masks(input_ids, valid, cls_id, sep_id, delimiter_ids):
    """Batch-width-independent GroundingDINO delimiter blocks/position resets."""
    batch, length = input_ids.shape
    allowed = torch.eye(length, dtype=torch.bool, device=input_ids.device)[None].repeat(batch, 1, 1)
    positions = torch.zeros_like(input_ids)
    for row, ids in enumerate(input_ids.tolist()):
        start = None
        for col, token_id in enumerate(ids):
            if not valid[row, col].item():
                continue
            if token_id in (cls_id, sep_id):
                if start is not None:
                    allowed[row, start:col, start:col] = True
                    positions[row, start:col] = torch.arange(col - start, device=input_ids.device)
                    start = None
                continue
            if start is None:
                start = col
            if token_id in delimiter_ids:
                end = col + 1
                allowed[row, start:end, start:end] = True
                positions[row, start:end] = torch.arange(end - start, device=input_ids.device)
                start = None
        if start is not None:
            end = int(valid[row].sum().item())
            allowed[row, start:end, start:end] = True
            positions[row, start:end] = torch.arange(end - start, device=input_ids.device)
    return _safe_allowed_mask(allowed, ~valid), positions


class TurboTextEncoder(nn.Module):
    """Unfrozen standard local BERT-base -> 256 tokens, without CLS pooling.

    ``bert_path`` must be the local directory validated by inspect_turbo_assets.
    Random/config initialization never opens BERT weights. The caller owns
    freezing and dtype policy; subclassing keeps bert/text_projection top-level.
    """

    def __init__(self, bert_path, *, random_init=False, initialize_from_config=False):
        super().__init__()
        path = Path(bert_path)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("bert_path must be an audited absolute local directory")
        import transformers
        from transformers import BertConfig, BertModel, BertTokenizerFast

        config = BertConfig.from_pretrained(str(path), local_files_only=True)
        if (config.model_type != "bert" or config.hidden_size != 768
                or config.is_decoder or config.add_cross_attention
                or getattr(config, "auto_map", None)
                or getattr(config, "position_embedding_type", "absolute") != "absolute"
                or config.max_position_embeddings < 256):
            raise ValueError("TurboTextEncoder requires audited standard absolute-position BERT-base config")
        config.use_cache = False
        config._attn_implementation = "eager"
        self.tokenizer = BertTokenizerFast.from_pretrained(str(path), local_files_only=True)
        if self.tokenizer.padding_side != "right":
            raise ValueError("Turbo text segmentation requires right padding")
        # The unused CLS pooler is omitted: token math is unchanged and DDP has
        # no permanently unused trainable pooler parameters.
        if random_init or initialize_from_config:
            self.bert = BertModel(config, add_pooling_layer=False)
        else:
            # Only the audited single-file formats are supported. Explicitly
            # prefer safetensors; never let HF discover shards/other formats.
            use_safetensors = (path / "model.safetensors").is_file()
            if not use_safetensors:
                from .turbo_initialization import validate_torch_weights_archive
                validate_torch_weights_archive(path / "pytorch_model.bin")
            self.bert = BertModel.from_pretrained(
                str(path), config=config, add_pooling_layer=False,
                local_files_only=True, attn_implementation="eager",
                use_safetensors=use_safetensors,
            )
        # HF 4 BERT accepts [B,T,T] allowed bool masks; HF 5 BERT's common
        # masking utility instead accepts a pre-expanded [B,1,T,T] additive mask.
        self._bert_additive_mask = int(transformers.__version__.split(".")[0]) >= 5
        self.text_projection = nn.Linear(768, 256)
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)
        self._delimiter_ids = tuple(self.tokenizer.convert_tokens_to_ids([".", "?"]))
        required_ids = (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                        self.tokenizer.pad_token_id, *self._delimiter_ids)
        if (any(value is None for value in required_ids)
                or len(set(required_ids)) != len(required_ids)
                or self.tokenizer.unk_token_id in self._delimiter_ids):
            raise ValueError("BERT tokenizer must define distinct CLS/SEP/PAD/period/question IDs")

    def forward(self, instructions, device):
        if (isinstance(instructions, (str, bytes)) or not isinstance(instructions, Sequence)
                or not instructions or any(not isinstance(item, str) for item in instructions)):
            raise ValueError("instructions must be a nonempty sequence of raw strings")
        tokenized = self.tokenizer(list(instructions), padding="longest", truncation=True,
                                   max_length=256, return_tensors="pt")
        ids = tokenized["input_ids"]
        valid = tokenized["attention_mask"].bool()
        content = valid.clone()
        for token_id in set(self.tokenizer.all_special_ids) | set(self._delimiter_ids):
            content &= ids != token_id
        if (~content.any(-1)).any().item():
            raise ValueError("every instruction must contain a non-special, non-delimiter content token")
        allowed, positions = _delimiter_masks(ids, valid, self.tokenizer.cls_token_id,
                                               self.tokenizer.sep_token_id, self._delimiter_ids)
        device = torch.device(device)
        inputs = {name: value.to(device) for name, value in tokenized.items()
                  if name in {"input_ids", "token_type_ids"}}
        allowed, content, valid = allowed.to(device), content.to(device), valid.to(device)
        if self._bert_additive_mask:
            mask = torch.zeros(allowed.shape, dtype=self.bert.embeddings.word_embeddings.weight.dtype,
                               device=device).masked_fill(~allowed, -torch.inf)[:, None]
        else:
            mask = allowed
        outputs = self.bert(**inputs, attention_mask=mask, position_ids=positions.to(device),
                            use_cache=False, return_dict=True)
        tokens = self.text_projection(outputs.last_hidden_state.to(self.text_projection.weight.dtype))
        return tokens, ~valid, allowed, content


def masked_mean_cognition(tokens, content_mask):
    """Content-only mean [B,1,H]; reject empty rows instead of silently pooling CLS."""
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3
            or not tokens.is_floating_point() or not all(tokens.shape)):
        raise ValueError("tokens must be a nonempty floating [B,T,H] tensor")
    if (not isinstance(content_mask, torch.Tensor) or content_mask.dtype != torch.bool
            or tuple(content_mask.shape) != tuple(tokens.shape[:2])
            or content_mask.device != tokens.device):
        raise ValueError("content_mask must be a bool [B,T] tensor on the token device")
    counts = content_mask.sum(-1, keepdim=True)
    if (counts == 0).any().item():
        raise ValueError("each cognition row requires at least one content token")
    # masked_fill, unlike multiplication by zero, also excludes NaN/Inf values
    # outside the content mask. Accumulate in FP32 for reduced-precision inputs.
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        values = tokens.float() if tokens.dtype in (torch.float16, torch.bfloat16) else tokens
        return values.masked_fill(~content_mask[..., None], 0).sum(1, keepdim=True) / counts[..., None]
