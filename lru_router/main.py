"""
LRU-Friendly Expert Router with Differentiable Loss

A PyTorch implementation of a batched top-k router with LRU-aware loss.
CPU-only version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import OrderedDict

# ==========================================
# 0. 全局设置 (CPU Only)
# ==========================================
# 强制使用双精度 (Float64) 以获得最大数值稳定性
# 在处理微小的相似度差异（如 0.9999 vs 1.0）时，这至关重要
torch.set_default_dtype(torch.float64)
# CPU only - no CUDA
device = torch.device("cpu")


# ==========================================
# 1. 模型组件
# ==========================================
class BatchedRouter(nn.Module):
    """
    标准的 Top-K 路由器，返回路由概率和索引。
    """

    def __init__(self, input_dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

        # 正交初始化：确保初始状态下专家分布相对均匀
        with torch.no_grad():
            init_weights = torch.randn(num_experts, input_dim, dtype=torch.float64)
            self.gate.weight.copy_(F.normalize(init_weights, p=2, dim=1))

    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        # 获取前 k 个专家的值和索引
        top_k_vals, top_k_indices = torch.topk(probs, k=self.top_k, dim=-1)

        # 构建完整的概率矩阵，未选中的位置为 0
        full_probs = torch.zeros_like(probs)
        # 归一化选中的概率，使其和为 1
        top_k_vals_norm = top_k_vals / top_k_vals.sum(dim=-1, keepdim=True)
        full_probs.scatter_(1, top_k_indices, top_k_vals_norm)

        return full_probs, top_k_indices


def calculate_lru_loss(probs, indices, window):
    """
    计算可微损失，惩罚未出现在近期历史窗口中的专家。
    这鼓励路由器重复使用最近使用过的专家（LRU 友好）。
    """
    # 创建当前选择的二进制掩码 [Batch, Experts]
    current_mask = torch.zeros_like(probs)
    current_mask.scatter_(1, indices, 1.0)

    # 准备历史窗口 (严格因果关系 Strict Causal)
    # 维度变换: [Batch, Experts] -> [Experts, Batch] -> 增加维度 -> [1, Experts, Batch]
    current_mask_t = current_mask.permute(1, 0).unsqueeze(0)

    # 向右填充/移动 1 位，确保当前步无法"看到"自己（因果性）
    # 截断最后一位以保持长度一致
    history_mask = F.pad(current_mask_t, (1, 0))[:, :, :-1]

    # 使用 MaxPool1d 充当窗口上的逻辑 "OR" 操作
    # 只要在窗口期内出现过，结果就是 1.0
    history_padded = F.pad(history_mask, (window - 1, 0))
    lru_mask = F.max_pool1d(history_padded, kernel_size=window, stride=1)

    # 变回 [Batch, Experts] 维度
    lru_mask = lru_mask.squeeze(0).permute(1, 0)

    # Loss 计算: 最小化分配给"未在近期历史中出现"的专家的概率
    # loss = prob * (1 - is_in_history)
    loss = (probs * (1.0 - lru_mask)).sum(dim=-1).mean()
    return loss


# ==========================================
# 2. 指标计算与数据生成
# ==========================================
def calculate_hard_metrics(indices_cpu, capacity):
    """
    模拟标准 LRU 缓存以计算实际未命中率 (Miss Rate)。
    这是不可微的硬指标，用于验证 Loss 的有效性。
    """
    cache = OrderedDict()
    misses = 0
    flat_indices = indices_cpu.flatten()

    for idx in flat_indices:
        if idx in cache:
            # 命中：移动到末尾（表示最近刚使用）
            cache.move_to_end(idx)
        else:
            # 未命中
            misses += 1
            cache[idx] = True
            # 如果超出容量，移除最久未使用的项（第一个项）
            if len(cache) > capacity:
                cache.popitem(last=False)
    return misses


def generate_batch_inputs(steps, dim, target_smoothness, device):
    """
    生成具有目标余弦相似度 (target_smoothness) 的向量序列。
    返回向量序列以及实际实现的平均相似度用于验证。
    """
    # 极端情况：完全静止
    if target_smoothness == 1.0:
        vec = torch.randn(1, dim, device=device)
        vectors = F.normalize(vec.expand(steps, dim), p=2, dim=1)
        return vectors, 1.0

    vectors = torch.empty(steps, dim, device=device)
    # 初始化第一个向量
    current = F.normalize(torch.randn(1, dim, device=device), p=2, dim=1)
    vectors[0] = current

    # 预先生成所有噪声
    noises = torch.randn(steps - 1, dim, device=device)

    # 混合系数计算 (Float64)
    rho = torch.tensor(target_smoothness, dtype=torch.float64, device=device)
    sqrt_rho = torch.sqrt(1.0 - rho**2)

    for t in range(steps - 1):
        noise = noises[t]
        # 施密特正交化 (Gram-Schmidt): 确保噪声垂直于当前向量
        noise = noise - (torch.sum(noise * current) * current)
        noise = F.normalize(noise, p=2, dim=0)

        # 更新状态：混合旧向量和新噪声
        current = rho * current + sqrt_rho * noise
        current = F.normalize(current, p=2, dim=1)
        vectors[t + 1] = current

    # 验证环节：计算相邻向量之间的实际点积
    v_t = vectors[:-1]
    v_t1 = vectors[1:]
    actual_sim = (v_t * v_t1).sum(dim=1).mean().item()

    return vectors, actual_sim


# ==========================================
# 3. 实验运行器
# ==========================================
def run_single_experiment(params):
    # 初始化路由器
    router = BatchedRouter(params["dim"], params["experts"], params["top_k"]).to(device)

    # 生成数据
    inputs, actual_sim = generate_batch_inputs(
        params["steps"], params["dim"], params["smoothness"], device
    )

    # 前向传播
    probs, indices = router(inputs)

    # 计算软 Loss (LRU Loss)
    loss = calculate_lru_loss(probs, indices, params["window"])

    # 计算硬指标 (实际 Miss Rate)
    total_misses = calculate_hard_metrics(indices.cpu().numpy(), params["capacity"])
    avg_miss_rate = total_misses / params["steps"]

    return {
        "Capacity": params["capacity"],
        "Target Smooth": params["smoothness"],
        "Actual Smooth": actual_sim,
        "Loss": float(loss.item()),
        "Miss Rate": avg_miss_rate,
    }


def main():
    print(f"Running LRU Router Experiment on {device}")

    # --- 配置 ---
    CONSTANTS = {
        "experts": 128,  # 专家总数
        "dim": 256,  # 向量维度
        "steps": 4096,  # 时间步长
        "window": 16,  # LRU 历史窗口大小
        "top_k": 8,  # 每次激活的专家数
        "capacity": 10,  # 缓存容量
    }

    # --- 采样策略 ---
    # 对数空间采样，重点关注 [0.9, 1.0] 这个高平滑度区域
    # Epsilon 从 ~0.3 到 0.00001
    epsilons = np.logspace(-0.5, -5.5, num=15)

    # 将 epsilon 转换为平滑度 (1 - epsilon)，并加入锚点 0.5 和 1.0
    smoothness_levels = [0.5] + (1.0 - epsilons).tolist() + [1.0]
    # 去重并排序
    smoothness_levels = sorted(list(set(smoothness_levels)))

    print(f"Sampling {len(smoothness_levels)} points (log scale around 1.0)")

    results = []
    for smooth in smoothness_levels:
        params = CONSTANTS.copy()
        params["smoothness"] = smooth
        # 运行实验
        results.append(run_single_experiment(params))

    # --- 输出结果 ---
    df = pd.DataFrame(results)
    print("\n=== 实验结果 ===")

    # 格式化输出，保持 Key 不变
    print(
        df[["Target Smooth", "Actual Smooth", "Loss", "Miss Rate"]].to_string(
            index=False,
            formatters={
                "Target Smooth": "{:.6f}".format,
                "Actual Smooth": "{:.6f}".format,
                "Loss": "{:.5f}".format,
                "Miss Rate": "{:.4f}".format,
            },
        )
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
