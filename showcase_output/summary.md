# 展示摘要（Baseline vs Macro Stable）

## 1) 我们做了什么
我们对比了两套策略：Baseline 风险平价引擎与 Macro Stable 宏观驱动稳定版。
在宏观版本中，我们让宏观信号只用于调整风险预算，再通过协方差结构映射为实际权重，不直接由宏观信号拍板持仓。

## 2) 关键设定与约束
我们使用月度调仓框架，并以风险预算/风险贡献思想构建多资产配置。
我们设置 TARGET_VOL=0.05 作为展示参照线，用于观察滚动波动率是否贴近目标风险水平。
对权重约束方面，我们在结果中观察到单资产与单步变动受到明显限制；若需声明为模型内硬约束，应以回测脚本实现为准。

## 3) 结果对比（自动填充）
### 回测段（2018-2023）
- Baseline: return=4.25%, vol=2.78%, sharpe=1.529, maxDD=-3.27%, calmar=1.298
- Macro Stable: return=4.54%, vol=3.99%, sharpe=1.137, maxDD=-6.12%, calmar=0.742
### 验证段（2024-2025）
- Baseline: return=5.11%, vol=4.71%, sharpe=1.086, maxDD=-6.65%, calmar=0.769
- Macro Stable: return=6.77%, vol=4.44%, sharpe=1.527, maxDD=-5.47%, calmar=1.239
- 资金利用率（avg_cash）: Baseline=4.60%, Macro Stable=15.39%
- 集中度（max_single_asset_weight）: Baseline=59.46%, Macro Stable=38.83%
- 单步变动（max_single_step_weight_change）: Baseline=24.62%, Macro Stable=20.16%
- 换手率均值: Baseline=3.82%, Macro Stable=8.92%

## 4) 宏观版波动/集中变化的解释（谨慎表述）
宏观版本通过风险预算再分配改变了资产风险暴露结构，而非直接设定权重。
当宏观信号持续偏向某类资产时，组合可能出现更高权益或信用暴露，从而引起波动、回撤与换手特征的阶段性变化。
因此，在展示中应同时报告收益、波动、回撤、换手与现金占比，避免只看单一绩效指标。
