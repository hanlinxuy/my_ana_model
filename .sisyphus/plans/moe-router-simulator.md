# MoE Router Simulator 重构计划

## TL;DR

> **目标**：将现有 LRU loss 代码重构为支持多级存储架构（DDR + Flash/PIM）的 MoE Router 模拟器
> 
> **核心功能**：
> - 多级存储带宽模型（Flash→DDR→NPU vs Flash→PIM）
> - 多种路由策略（LRU、PIM-Only、Hybrid、Fixed-Split、Cache-Only）
> - 可配置参数接口
> - 全面的测试用例（不同 smoothness、边界情况）
> 
> **评估指标**：
> - 平均延迟 per token（并行路径取 max）
> - DDR Miss Rate（加载次数）
> - k1/k2/k3 分布统计
> 
> **Estimated Effort**: Medium (4-6 小时)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Core Refactor → Strategy Implementations → Test Suite

---

## Context

### Original Request
将现有的 LRU router 代码改造为 MoE Router 模拟器，支持多级存储架构：
- DDR + NPU：计算 topk1（高优先级，加载并更新 LRU）
- Flash + PIM：计算 topk2（次优先级，不更新 LRU）
- DDR Cache：复用 topk3（命中缓存，更新 LRU）

### Interview Summary

**Key Discussions**:
- **专家选择逻辑**：每个 token 选 K 个专家，按得分从高到低分类
  - 在 cache 中 → topk3（复用）
  - 不在 cache，k1 预算未满 → topk1（加载到 DDR）
  - 不在 cache，k1 满 k2 未满 → topk2（PIM 计算）
  - k1+k2 预算都满 → 从 cache 选得分高的补足
- **带宽模型**：归一化，专家大小=1
  - Flash→DDR: 1, DDR→NPU: 8, Flash→PIM: 4
- **延迟计算**：单 token 延迟 = max(time_topk1, time_topk2, time_topk3)
- **测试策略**：LRU、PIM-Only、Hybrid、Fixed-Split、Cache-Only

### Design Decisions

| 参数 | 值 | 说明 |
|------|-----|------|
| num_experts | 128 | 总专家数 |
| K | 8 | 每 token 选 K 个专家 |
| cache_size | 32 | DDR cache 容量 |
| k1 (default) | 3 | DDR 加载预算 |
| k2 (default) | 2 | PIM 计算预算 |
| k3 (implicit) | K - k1 - k2 = 3 | cache 复用 |
| steps | 4096 | 模拟时间步长 |

---

## Verification Strategy

### Test Strategy
- **Framework**: pytest
- **Test Level**: Unit + Integration
- **Coverage**: Core logic, edge cases, parameter validation

### Agent-Executed QA Scenarios

每个策略实现后，通过自动化测试验证：

```
Scenario: 完全静态工作负载 (smoothness=1.0)
  Tool: Bash (pytest)
  Preconditions: 代码已重构
  Steps:
    1. 运行 pytest tests/test_static_workload.py
    2. 验证 LRU cache 不更新（所有专家已在 cache）
    3. 验证 k3 = K（全部 cache 命中）
    4. 验证延迟 = K × 0.125
  Expected Result: 测试通过，断言成功
  Evidence: test output log

Scenario: 完全随机工作负载 (smoothness=0.0)
  Tool: Bash (pytest)
  Steps:
    1. 运行 pytest tests/test_random_workload.py
    2. 验证 miss rate ≈ 100%（cache_size << num_experts）
    3. 验证 k1 + k2 ≈ K（cache 几乎不命中）
  Expected Result: miss rate > 90%
  Evidence: test output log

Scenario: Hybrid 策略正确分类
  Tool: Bash (pytest)
  Steps:
    1. 运行 pytest tests/test_hybrid_classification.py
    2. 验证 k1 + k2 + k3 = K（每个 token）
    3. 验证 k1 ≤ k1_limit, k2 ≤ k2_limit
    4. 验证不在 cache 的专家优先填满 k1，再填 k2
  Expected Result: 分类逻辑正确
  Evidence: test output log

Scenario: 延迟计算正确性
  Tool: Bash (pytest)
  Steps:
    1. 运行 pytest tests/test_latency_calculation.py
    2. 验证 latency = max(k1×1.125, k2×0.25, k3×0.125)
    3. 验证边界情况（k1=0, k2=0, k3=0）
  Expected Result: 计算正确
  Evidence: test output log
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Core Architecture):
├── Task 1: 重构核心数据结构（ExpertCache, RouterState）
├── Task 2: 实现带宽模型和延迟计算
└── Task 3: 实现基础策略接口

Wave 2 (Strategies & Tests):
├── Task 4: 实现 LRU 策略
├── Task 5: 实现 PIM-Only 策略
├── Task 6: 实现 Hybrid 策略
├── Task 7: 实现 Fixed-Split 策略
├── Task 8: 实现 Cache-Only 策略
├── Task 9: 创建测试套件
└── Task 10: 创建主入口和配置接口

Critical Path: Task 1 → Task 3 → Task 4/6 → Task 9 → Task 10
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3 | None |
| 2 | 1 | 4-8 | 3 |
| 3 | 1 | 4-8 | 2 |
| 4 | 2, 3 | 9 | 5-8 |
| 5 | 2, 3 | 9 | 4, 6-8 |
| 6 | 2, 3 | 9 | 4-5, 7-8 |
| 7 | 2, 3 | 9 | 4-6, 8 |
| 8 | 2, 3 | 9 | 4-7 |
| 9 | 4-8 | 10 | None |
| 10 | 9 | None | None |

---

## TODOs

- [x] 1. 重构核心数据结构

  **What to do**:
  - 创建 `ExpertCache` 类：LRU cache 管理（capacity, hit/miss tracking）
  - 创建 `RouterConfig` 类：配置参数（num_experts, K, cache_size, k1, k2, bandwidths）
  - 创建 `TokenRouter` 类：路由决策核心（输入专家得分，输出分类）
  - 重构现有代码，移除 PyTorch 依赖（模拟器不需要梯度）
  
  **File structure**:
  ```
  moe_simulator/
  ├── __init__.py
  ├── core/
  │   ├── __init__.py
  │   ├── cache.py        # ExpertCache
  │   ├── config.py       # RouterConfig
  │   └── router.py       # TokenRouter
  └── main.py
  ```

  **Must NOT do**:
  - 不要保留 PyTorch nn.Module（模拟器不需要训练）
  - 不要硬编码参数
  - 不要混合策略逻辑和基础数据结构

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Justification**: 主要是重构和类设计，不涉及复杂算法

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Task 2, 3

  **References**:
  - Current: `lru_router/main.py:90-113` - LRU cache 逻辑
  
  **Acceptance Criteria**:
  - [ ] `ExpertCache` 支持 put/get/evict 操作
  - [ ] `RouterConfig` 可序列化/反序列化（dataclass）
  - [ ] `TokenRouter` 有清晰的 `route(scores) -> classification` 接口
  - [ ] 所有类有类型注解和 docstring

- [x] 2. 实现带宽模型和延迟计算

  **What to do**:
  - 实现 `BandwidthModel` 类：存储归一化带宽（1, 8, 4）
  - 实现 `LatencyCalculator` 类：根据 (k1, k2, k3) 计算单 token 延迟
  - 延迟公式：`latency = max(k1×1.125, k2×0.25, k3×0.125)`
  
  **Must NOT do**:
  - 不要假设带宽值（从 config 读取）
  - 不要用浮点数比较（用 epsilon）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocked By**: Task 1

  **Acceptance Criteria**:
  - [ ] 延迟计算与手工验证一致
  - [ ] 支持边界情况（k1=0, k2=0, k3=0）
  - [ ] 单元测试覆盖所有路径

- [x] 3. 实现基础策略接口

  **What to do**:
  - 创建抽象基类 `RoutingStrategy`
  - 定义接口：`select_experts(scores, cache_state) -> (k1_list, k2_list, k3_list)`
  - 创建策略工厂 `StrategyFactory`
  
  **Must NOT do**:
  - 不要在基类中实现具体逻辑
  - 不要硬编码策略参数

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocked By**: Task 1

  **Acceptance Criteria**:
  - [ ] 抽象基类定义清晰
  - [ ] 工厂模式支持策略注册
  - [ ] 示例策略（StubStrategy）可运行

- [x] 4. 实现 LRU 策略

  **What to do**:
  - 创建 `LRUStrategy` 类继承 `RoutingStrategy`
  - 所有专家都尝试加载到 DDR（k1 = min(K, 不在 cache 的数量)）
  - 不区分 k2（k2 = 0）
  - cache miss 时加载并更新 LRU
  
  **Must NOT do**:
  - 不要用 PIM 路径（这是纯 DDR 策略）
  - 不要限制 k1（尽可能加载）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 2, 3

  **Acceptance Criteria**:
  - [ ] 完全静态工作负载：k3 = K，延迟 = K × 0.125
  - [ ] 完全随机工作负载：miss rate ≈ 100%，延迟 ≈ K × 1.125

- [x] 5. 实现 PIM-Only 策略

  **What to do**:
  - 创建 `PIMOnlyStrategy` 类
  - 所有专家都走 Flash→PIM 路径
  - k2 = K，k1 = 0，k3 = 0
  - 不更新 LRU cache
  
  **Must NOT do**:
  - 不要加载到 DDR
  - 不要更新 cache

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 2, 3

  **Acceptance Criteria**:
  - [ ] 所有 token：k2 = K
  - [ ] 延迟 = K × 0.25
  - [ ] miss rate = 0（因为没有加载）

- [x] 6. 实现 Hybrid 策略

  **What to do**:
  - 创建 `HybridStrategy` 类
  - 实现核心算法（见 Context 部分）
  - 按得分优先级分配 k1 → k2 → cache 补足
  
  **Algorithm**:
  ```python
  def select_experts(scores, cache, k1_limit, k2_limit, K):
      sorted_experts = sort_by_score_desc(scores)
      k1_list, k2_list, k3_list = [], [], []
      
      for expert in sorted_experts:
          if len(k1_list) + len(k2_list) + len(k3_list) >= K:
              break
              
          if expert in cache:
              k3_list.append(expert)
              cache.update_lru(expert)
          else:
              if len(k1_list) < k1_limit:
                  k1_list.append(expert)
                  cache.put(expert)  # 可能触发 eviction
              elif len(k2_list) < k2_limit:
                  k2_list.append(expert)
              else:
                  # 从 cache 中选得分高的补足
                  candidates = [e for e in cache if e not in k3_list]
                  best = max(candidates, key=lambda e: scores[e])
                  k3_list.append(best)
                  cache.update_lru(best)
      
      return k1_list, k2_list, k3_list
  ```

  **Must NOT do**:
  - 不要打破 k1 + k2 + k3 = K 的约束
  - 不要在 k2 路径更新 LRU

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 2, 3

  **Acceptance Criteria**:
  - [ ] k1 ≤ k1_limit, k2 ≤ k2_limit, k1 + k2 + k3 = K
  - [ ] cache 命中的专家优先进入 k3
  - [ ] 不在 cache 的专家优先填满 k1，再填 k2

- [x] 7. 实现 Fixed-Split 策略

  **What to do**:
  - 创建 `FixedSplitStrategy` 类
  - 固定前 k1_limit 个专家走 DDR，接下来 k2_limit 个走 PIM
  - 不考虑是否在 cache
  
  **Must NOT do**:
  - 不要用得分判断（固定位置分割）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 2, 3

  **Acceptance Criteria**:
  - [ ] 位置 0..k1_limit-1 走 k1
  - [ ] 位置 k1_limit..k1_limit+k2_limit-1 走 k2
  - [ ] 剩余走 k3（cache 允许时）或替换

- [x] 8. 实现 Cache-Only 策略

  **What to do**:
  - 创建 `CacheOnlyStrategy` 类
  - 只有 cache 命中的专家走 DDR（k1 = 0）
  - 其他全部走 PIM（k2）
  - 如果 cache 命中数 > K，选得分高的 K 个
  
  **Must NOT do**:
  - 不要从 Flash 加载到 DDR

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 2, 3

  **Acceptance Criteria**:
  - [ ] k1 = 0（从不加载）
  - [ ] k3 = min(K, cache_hit_count)
  - [ ] k2 = K - k3

- [x] 9. 创建测试套件

  **What to do**:
  - 创建 `tests/` 目录
  - 实现 `test_static_workload.py`：smoothness=1.0
  - 实现 `test_random_workload.py`：smoothness=0.0
  - 实现 `test_hybrid_classification.py`：分类逻辑
  - 实现 `test_latency_calculation.py`：延迟计算
  - 实现 `test_strategies_comparison.py`：策略对比
  
  **Test cases**:
  ```python
  # 边界情况
  - cache_size = 0（无缓存）
  - cache_size >= num_experts（全缓存）
  - k1 = 0, k2 = 0, k3 = K（纯 cache）
  - k1 = K, k2 = 0, k3 = 0（纯 DDR）
  - k1 = 0, k2 = K, k3 = 0（纯 PIM）
  
  # 工作负载
  - smoothness = [0.0, 0.5, 0.9, 0.99, 1.0]
  - steps = [100, 1000, 4096]
  ```

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocked By**: Task 4-8

  **Acceptance Criteria**:
  - [ ] pytest 运行通过，覆盖率 > 80%
  - [ ] 所有边界情况有测试
  - [ ] 测试文档说明每个 case 的期望行为

- [x] 10. 创建主入口和配置接口

  **What to do**:
  - 重构 `main.py` 作为主入口
  - 实现 `ConfigLoader`：从 YAML/JSON 加载配置
  - 实现 `SimulationRunner`：运行批量模拟
  - 实现 `ResultsAggregator`：汇总多策略结果
  - 添加命令行参数解析（argparse）
  
  **CLI interface**:
  ```bash
  python -m moe_simulator \
    --config config.yaml \
    --strategies lru,pim-only,hybrid,fixed-split,cache-only \
    --output results.csv
  ```

  **Config format**:
  ```yaml
  num_experts: 128
  K: 8
  cache_size: 32
  steps: 4096
  smoothness_levels: [0.0, 0.5, 0.9, 0.99, 1.0]
  
  strategies:
    hybrid:
      k1: 3
      k2: 2
    fixed-split:
      k1: 4
      k2: 2
  
  bandwidth:
    flash_to_ddr: 1
    ddr_to_npu: 8
    flash_to_pim: 4
  ```

  **Must NOT do**:
  - 不要硬编码任何参数
  - 不要让配置过于复杂（保持简单）

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocked By**: Task 9

  **Acceptance Criteria**:
  - [ ] 可从命令行运行
  - [ ] 支持配置文件
  - [ ] 输出 CSV 格式的对比结果
  - [ ] 文档说明如何使用

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `refactor: extract core data structures` | `moe_simulator/core/` |
| 2 | `feat: implement bandwidth and latency models` | `moe_simulator/core/latency.py` |
| 3 | `feat: add routing strategy interface` | `moe_simulator/strategies/base.py` |
| 4-8 | `feat: implement {strategy} routing strategy` | `moe_simulator/strategies/{name}.py` |
| 9 | `test: add comprehensive test suite` | `tests/` |
| 10 | `feat: add CLI and configuration interface` | `main.py`, `config.yaml` |

---

## Success Criteria

### Verification Commands
```bash
# 运行所有测试
pytest tests/ -v --cov=moe_simulator

# 运行完整模拟
python -m moe_simulator --config examples/default.yaml

# 验证输出格式
head results.csv
```

### Final Checklist
- [ ] 5 种策略全部实现并测试
- [ ] 配置系统工作正常
- [ ] 测试覆盖率 > 80%
- [ ] 文档完整（README + 代码注释）
- [ ] 示例配置文件可用
