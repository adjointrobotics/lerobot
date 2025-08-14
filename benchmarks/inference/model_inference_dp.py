# Import necessary libraries
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import json
from lerobot.utils.utils import get_safe_torch_device
# Ensure all policy configs are registered (act, smolvla, pi0, pi0fast, etc.)
import lerobot.policies  # noqa: F401
from lerobot.policies.factory import get_policy_class

# When running as __main__, pass required CLI arguments (no hardcoded defaults). 


def prepare_observation_tensors(observation_frame: dict, device: torch.device, policy=None) -> dict:
    """Mimic predict_action preprocessing: to torch, normalize images, add batch dim, move to device."""
    batch = {}
    for name, value in observation_frame.items():
        # value is numpy array
        tensor = torch.from_numpy(value)
        if "image" in name:
            # to [0,1] float32, HWC -> CHW
            tensor = tensor.type(torch.float32) / 255.0
            tensor = tensor.permute(2, 0, 1).contiguous()
            
            # Resize image to expected dimensions if policy is provided and has image_features config
            if policy is not None and hasattr(policy.config, 'image_features'):
                expected_shape = policy.config.image_features.get(name)
                if expected_shape is not None:
                    try:
                        current_shape = tensor.shape  # (C, H, W)
                        expected_dims = (expected_shape.shape[1], expected_shape.shape[2])  # (H, W)
                        
                        if current_shape[1:] != expected_dims:
                            print(f"Resizing image '{name}' from {current_shape[1:]} to {expected_dims}")
                            # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
                            tensor_batched = tensor.unsqueeze(0)
                            # Interpolate and remove batch dimension: (1, C, H, W) -> (C, H, W)
                            tensor = torch.nn.functional.interpolate(
                                tensor_batched, 
                                size=expected_dims, 
                                mode="bilinear", 
                                align_corners=False
                            ).squeeze(0)
                    except (IndexError, AttributeError) as e:
                        print(f"Warning: Could not access expected dimensions for image '{name}': {e}")
            # Fallback: check input_features if image_features is not available
            elif policy is not None and hasattr(policy.config, 'input_features'):
                expected_shape = policy.config.input_features.get(name)
                if expected_shape is not None and hasattr(expected_shape, 'shape'):
                    try:
                        current_shape = tensor.shape  # (C, H, W)
                        expected_dims = (expected_shape.shape[1], expected_shape.shape[2])  # (H, W)
                        
                        if current_shape[1:] != expected_dims:
                            print(f"Resizing image '{name}' from {current_shape[1:]} to {expected_dims}")
                            # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
                            tensor_batched = tensor.unsqueeze(0)
                            # Interpolate and remove batch dimension: (1, C, H, W) -> (C, H, W)
                            tensor = torch.nn.functional.interpolate(
                                tensor_batched, 
                                size=expected_dims, 
                                mode="bilinear", 
                                align_corners=False
                            ).squeeze(0)
                    except (IndexError, AttributeError) as e:
                        print(f"Warning: Could not access expected dimensions for image '{name}': {e}")
            elif "image" in name:
                print(f"Warning: No expected dimensions found for image '{name}'. Images may have inconsistent sizes.")
        
        # add batch dim and move to device
        tensor = tensor.unsqueeze(0).to(device)
        batch[name] = tensor
    return batch


def run_benchmark(observation_path: str, model_root_path: str, num_iters: int = 50, task_description: str = ""):
    # Resolve model path: allow either a pretrained_model dir or a checkpoints root (like .../last)
    model_root = Path(model_root_path)
    pretrained_path = model_root / "pretrained_model"
    if pretrained_path.is_dir():
        model_path = pretrained_path
    else:
        model_path = model_root

    # Load observation payload
    with open(observation_path, "rb") as f:
        payload = pickle.load(f)
    observation_frame = payload["observation_frame"]
    task = task_description if task_description else (payload.get("task", "") or "")
    robot_type = payload.get("robot_type", "") or ""

    # Load policy
    with open(Path(model_path) / "config.json", "r") as f:
        cfg = json.load(f)
    policy_type = cfg.get("type") or cfg.get("policy", {}).get("type")
    if not policy_type:
        raise RuntimeError("Could not determine policy type from config.json")
    PolicyCls = get_policy_class(policy_type)
    policy = PolicyCls.from_pretrained(config=None, pretrained_name_or_path=str(model_path))
    device = get_safe_torch_device(policy.config.device)
    policy.eval()

    # Prepare tensors
    batch = prepare_observation_tensors(observation_frame, device, policy)

    # Restrict to inputs declared by the policy (if available)
    expected_inputs = set(getattr(policy.config, "input_features", {}).keys())
    if expected_inputs:
        batch = {k: v for k, v in batch.items() if k in expected_inputs}

    # Always provide task/robot_type; ignored if not expected
    batch["task"] = task
    batch["robot_type"] = robot_type

    # Apply optional transforms and/or normalization for specialized generate paths
    if getattr(policy.config, "adapt_to_pi_aloha", False):
        state_key = next((k for k in batch if k.endswith("state") or "state" in k), None)
        if state_key is not None and hasattr(policy, "_pi_aloha_decode_state"):
            batch[state_key] = policy._pi_aloha_decode_state(batch[state_key])

    use_generate_actions = hasattr(getattr(policy, "model", None), "generate_actions")
    norm_batch = policy.normalize_inputs(batch) if use_generate_actions else None

    # Time action generation
    times_s: list[float] = []
    with torch.inference_mode():
        enable_amp = (device.type == "cuda") and getattr(policy.config, "use_amp", False)
        with torch.autocast(device_type="cuda", enabled=enable_amp):
            for _ in range(num_iters):
                start = time.time()
                if use_generate_actions:
                    _ = policy.model.generate_actions(norm_batch)
                else:
                    _ = policy.select_action(batch)
                elapsed = time.time() - start
                times_s.append(elapsed)

    arr = np.array(times_s, dtype=np.float64)
    mean_s = float(arr.mean()) if arr.size else 0.0
    var_s = float(arr.var(ddof=1)) if arr.size > 1 else 0.0

    print(f"Mean: {mean_s:.6f}s | Variance: {var_s:.6f}")
    return mean_s, var_s


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline inference benchmark")
    parser.add_argument("--observation", required=True, help="Path to pickled single observation payload")
    parser.add_argument("--model-root", required=True, help="Path to model root (pretrained_model or checkpoints root like .../last)")
    parser.add_argument("--num-iters", type=int, default=50, help="Number of iterations for timing")
    parser.add_argument("--task-description", required=True, help="Task description string")
    args = parser.parse_args()

    run_benchmark(args.observation, args.model_root, args.num_iters, args.task_description)


if __name__ == "__main__":
    main()