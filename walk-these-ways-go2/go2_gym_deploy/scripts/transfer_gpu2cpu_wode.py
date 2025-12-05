# ...existing code...
from pathlib import Path
import torch
import pickle
import glob
import sys

logdir ="/home/majunchi/ws_wtw_go2/walk-these-ways-go2/runs/gait-conditioned-agility/2025-11-28/train/155719.605790"
print(f"Using logdir: {logdir}")

params_path = "/home/majunchi/ws_wtw_go2/walk-these-ways-go2/runs/gait-conditioned-agility/2025-11-28/train/155719.605790/parameters.pkl"


def to_cpu_recursive(obj):
    if torch.is_tensor(obj):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: to_cpu_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu_recursive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu_recursive(v) for v in obj)
    # numpy arrays containing tensors are rare; 保持原样
    return obj

# 首选：使用 torch.load(map_location="cpu") 直接把所有 tensor 映射到 CPU
try:
    pkl_cfg = torch.load(params_path, map_location="cpu")
    print("Loaded with torch.load(map_location='cpu').")
except Exception as e:
    print(f"torch.load failed: {e}; 尝试使用 pickle.load 并递归转换（可能仍会失败）。", file=sys.stderr)
    with open(params_path, 'rb') as f:
        pkl_cfg = pickle.load(f)

# 递归确保所有张量转为 CPU
pkl_cfg_cpu = to_cpu_recursive(pkl_cfg)

# 保存为 CPU 版（使用 torch.save 可确保后续 torch.load 不再引用 CUDA）
# save transferred .pkl file
with open(logdir+"/parameters_cpu_wode.pkl", 'wb') as file:
    pickle.dump(pkl_cfg_cpu, file)
    print("Transferred Pickle File has been saved as parameters_cpu_wpde.pkl")
