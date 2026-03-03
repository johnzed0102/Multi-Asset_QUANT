# 展示摘要（Baseline vs Macro Stable）

## 1) 我们做了什么
我们对比了 Baseline 风险平价策略与 Macro Stable 宏观驱动稳定版。
宏观版本中，宏观信号仅用于调整风险预算，最终权重仍通过协方差结构映射得到。

## 2) 关键约束与设定
我们使用月度调仓框架，并以风险预算/风险贡献思想构建多资产组合。
我们设置 TARGET_VOL=0.05 作为展示参照线，用于观察滚动波动率区间。
对于单资产权重上限与单次变动限制，如需判断是否模型内硬约束，应以回测引擎代码为准；本文仅如实报告结果表现。

## 3) 结果对比（自动填充）
### 回测段（2018-2023）
- Baseline: return=4.25%, vol=2.78%, sharpe=1.529, maxDD=-3.27%, calmar=1.298
- Macro Stable: return=4.54%, vol=3.99%, sharpe=1.137, maxDD=-6.12%, calmar=0.742
### 验证段（2024-2025）
- Baseline: return=5.11%, vol=4.71%, sharpe=1.086, maxDD=-6.65%, calmar=0.769
- Macro Stable: return=6.77%, vol=4.44%, sharpe=1.527, maxDD=-5.47%, calmar=1.239
- 现金占比（avg_cash）: Baseline=4.60%, Macro Stable=15.39%
- 最大单资产权重: Baseline=59.46%, Macro Stable=38.83%
- 最大单次权重变化: Baseline=24.62%, Macro Stable=20.16%
- 月度换手率均值: Baseline=3.82%, Macro Stable=8.92%

## 4) 对宏观版波动/集中变化的谨慎解释
宏观信号通过风险预算再分配影响组合风险结构，而不是直接指定目标权重。
当宏观状态持续偏向某类资产时，组合可能阶段性提高权益或信用暴露，从而带来波动率、回撤和换手率特征变化。
因此展示时建议联动观察收益、波动、回撤、现金占比和换手率，避免单指标解读。
