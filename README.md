# MoE 路由模拟器

用于评估不同路由策略的模拟框架，适用于多级存储层次结构（DDR + Flash/PIM）的 MoE（混合专家）架构。

## 功能特点

- **多种路由策略**: LRU、PIM-Only、Hybrid、Fixed-Split、Cache-Only
- **可配置带宽模型**: 归一化带宽比例（Flash→DDR、DDR→NPU、Flash→PIM）
- **灵活工作负载**: 可调节的时间局部性（smoothness）
- **全面指标**: 延迟、缓存命中率、k1/k2/k3 分布

## 快速开始

```bash
# 克隆仓库
git clone git@github.com:hanlinxuy/my_ana_model.git
cd my_ana_model

# 运行（使用默认配置）
PYTHONPATH=. python3 moe_simulator/main.py

# 运行特定策略
PYTHONPATH=. python3 moe_simulator/main.py --strategies lru,hybrid,pim-only

# 自定义配置
PYTHONPATH=. python3 moe_simulator/main.py --config examples/config.yaml --output results.csv
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | 内置默认值 |
| `--strategies` | 策略列表（逗号分隔） | 所有策略 |
| `--smoothness` | 平滑度级别 | 0.0, 0.5, 0.9, 0.99, 1.0 |
| `--output` | CSV 输出路径 | results.csv |
| `--num-tokens` | 模拟 token 数量 | 1000 |
| `--num-experts` | 专家总数 | 128 |
| `--K` | 每个 token 选出的专家数 | 8 |
| `--cache-size` | DDR 缓存容量 | 32 |
| `--k1` | DDR 加载预算 | 3 |
| `--k2` | PIM 计算预算 | 2 |

## 架构

### 存储层次

```
┌─────────────────────────────────────────┐
│               Flash 存储                  │
│         (慢速, 大容量)                   │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
   加载到 DDR     直接到 PIM
        │             │
        ▼             ▼
┌──────────────┐ ┌──────────────┐
│     DDR      │ │     PIM      │
│  (快速内存)  │ │  (内存计算)  │
└──────┬───────┘ └──────────────┘
       │
       ▼
┌──────────────┐
│     NPU      │
│   (计算)    │
└──────────────┘
```

### 带宽模型（归一化）

| 路径 | 带宽 | 每个专家延迟 |
|------|------|-------------|
| Flash → DDR → NPU | 1 + 8 | 1.125 |
| Flash → PIM | 4 | 0.25 |
| DDR → NPU（缓存命中） | 8 | 0.125 |

### 路由策略

| 策略 | 描述 |
|------|------|
| **LRU** | 所有专家加载到 DDR，纯 LRU 管理 |
| **PIM-Only** | 所有专家走 Flash→PIM，无 DDR 缓存 |
| **Hybrid** | 智能分配：k1→DDR，k2→PIM，k3→缓存 |
| **Fixed-Split** | 固定位置分割（前 k1→DDR，后 k2→PIM） |
| **Cache-Only** | 只使用缓存专家，其余走 PIM |

### 延迟计算

每个 token 的延迟是并行路径的**最大值**：

```
latency = max(k1 × 1.125, k2 × 0.25, k3 × 0.125)
```

其中：
- k1 = 通过 Flash→DDR→NPU 的专家数
- k2 = 通过 Flash→PIM 的专家数
- k3 = DDR 缓存命中的专家数

## 项目结构

```
pim_estimation/
├── moe_simulator/
│   ├── core/
│   │   ├── cache.py          # ExpertCache (LRU)
│   │   ├── config.py         # RouterConfig
│   │   ├── latency.py        # 带宽模型和延迟计算
│   │   ├── config_loader.py  # YAML/JSON 配置加载
│   │   ├── runner.py         # 模拟运行器
│   │   └── results.py        # 结果聚合
│   ├── strategies/
│   │   ├── base.py           # 路由策略基类
│   │   ├── factory.py        # 策略工厂
│   │   ├── lru.py           # LRU 策略
│   │   ├── pim_only.py      # PIM-Only 策略
│   │   ├── hybrid.py        # Hybrid 策略
│   │   ├── fixed_split.py   # Fixed-Split 策略
│   │   └── cache_only.py    # Cache-Only 策略
│   └── main.py               # CLI 入口
├── tests/                    # 测试套件（111 个测试）
├── examples/
│   └── config.yaml          # 示例配置
└── AGENTS.md               # Agent 开发指南
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_strategies.py -v

# 带覆盖率
pytest tests/ --cov=moe_simulator
```

## 示例输出

```
strategy   smoothness   avg_latency   cache_hit_rate
---------  -----------  ------------   -------------
lru        0.5         13.56          0.0%
lru        1.0         0.07           0.0%
hybrid     0.5         51.27          80.0%
hybrid     1.0         63.94          99.9%
pim_only   0.5         32.0           0.0%
pim_only   1.0         32.0           0.0%
```

## 原始 LRU 路由器

`lru_router/` 目录包含原始的 PyTorch 实现的可微 LRU 路由器。保留用于参考，但新的模拟器（`moe_simulator/`）是推荐用于评估路由策略的方式。

## 许可证

MIT License

## 作者

Hanlin Xu - hanlinxuy@gmail.com
