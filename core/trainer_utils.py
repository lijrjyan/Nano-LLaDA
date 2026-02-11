import math
from typing import Any, Dict, Optional, Tuple

import torch


def auto_device(device_arg: Optional[str] = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(step: int, total_steps: int, max_lr: float, min_lr_ratio: float = 0.1) -> float:
    if total_steps <= 0:
        return max_lr
    min_lr = max_lr * min_lr_ratio
    progress = min(max(step / total_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"], ckpt_obj
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"], ckpt_obj
    return ckpt_obj, None


def extract_state_dict_and_meta(ckpt_obj, prefer_args_meta: bool = False) -> Tuple[Any, Optional[Dict[str, Any]]]:
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            if prefer_args_meta and isinstance(ckpt_obj.get("args"), dict):
                return ckpt_obj["model_state_dict"], ckpt_obj["args"]
            return ckpt_obj["model_state_dict"], ckpt_obj
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"], ckpt_obj
    return ckpt_obj, None


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod.") :]
        if new_k.startswith("model."):
            new_k = new_k[len("model.") :]
        out[new_k] = v
    return out


def load_matching_weights(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    model_state = model.state_dict()
    matched = {}
    shape_mismatch = 0
    missing_name = 0

    for k, v in state_dict.items():
        if k not in model_state:
            missing_name += 1
            continue
        if model_state[k].shape != v.shape:
            shape_mismatch += 1
            continue
        matched[k] = v

    model.load_state_dict(matched, strict=False)
    return len(matched), shape_mismatch, missing_name
