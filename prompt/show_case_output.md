你是一个专业的量化研究工程师 + Prompt Engineer。
你现在需要在“模拟盘课题”项目中，生成可用于展示与PPT制作的“第6步：展示输出（output assets）”。

【重要背景（请严格遵守，不要臆造目录）】
当前项目根目录下已经存在三套回测结果文件夹（由先前脚本生成）：
1) 回测引擎/
   - weights.csv
   - nav_series.csv
   - backtest_results.csv
2) 回测引擎_macro/                （这是过渡版，可读取但默认不进入最终展示对比）
   - weights_macro.csv
   - nav_series_macro.csv
   - backtest_results_macro.csv
3) 回测引擎_macro_stable/         （这是最终要展示的宏观驱动版本）
   - weights_macro_stable.csv
   - nav_series_macro_stable.csv
   - backtest_results_macro_stable.csv

另外，数据面板在：
processed_data/
   - asset_price_panel.csv
   - asset_return_panel.csv
   - macro_panel.csv

【你的任务】
请在项目根目录新增一个脚本：
generate_showcase_outputs.py

该脚本的目标是：读取 baseline（回测引擎）与 macro_stable（回测引擎_macro_stable）的CSV结果，
生成一套“美观、专业、可直接用于PPT粘贴”的输出（表格 + 图片 + 摘要文档）。

【输出目录（必须自动创建，且不要覆盖原始回测输出）】
在项目根目录创建：
showcase_output/
    tables/
    figures/
    summary.md

【读入文件（必须健壮：路径不存在要报清晰错误）】
baseline：
- 回测引擎/weights.csv
- 回测引擎/nav_series.csv
- 回测引擎/backtest_results.csv

macro_stable：
- 回测引擎_macro_stable/weights_macro_stable.csv
- 回测引擎_macro_stable/nav_series_macro_stable.csv
- 回测引擎_macro_stable/backtest_results_macro_stable.csv

（可选）macro过渡版：默认不进入最终对比，但代码结构允许未来打开开关读入。

【必须生成的“表”输出（保存到 showcase_output/tables/）】
1) perf_comparison.csv
   - 行：baseline、macro_stable
   - 列：annual_return(回测/验证)、annual_volatility(回测/验证)、sharpe(回测/验证)、max_drawdown(回测/验证)、calmar(回测/验证)
        avg_cash（全样本）、max_single_asset_weight（全样本）、max_single_step_weight_change（全样本）
   - 注意：backtest_results.csv里period字段命名可能不同（如 backtest_2018_2023 / validation_2024_2025 或 2018-01-01_2023-12-31 等），
     需要写一个解析函数：只要能识别“2018-2023”为回测段，“2024-2025”为验证段即可（通过字符串包含 2018/2023/2024/2025 判断）。

2) turnover_summary.csv
   - 计算月度换手率：turnover_t = 0.5 * sum_i |w_i(t) - w_i(t-1)|
   - 输出 baseline 与 macro_stable 的：均值、P50、P90、最大值。

【必须生成的“图”输出（保存到 showcase_output/figures/，PNG，dpi>=200，白底，字号适中）】
A. nav_comparison.png
   - baseline 与 macro_stable 的净值曲线（同一张图两条线），起点=1
B. drawdown_comparison.png
   - baseline 与 macro_stable 的回撤曲线（同一张图两条线）
C. rolling_vol_252.png
   - 252日滚动年化波动率，两条线对比，并画一条水平线 TARGET_VOL=0.05（仅用于展示参照，不代表硬约束）
D. allocation_area_baseline.png
   - baseline 月度权重堆叠面积图（asset columns：hs300, sp500, cgb10y, gold, policy_bond, credit_bond, cash；实际有哪些列就画哪些）
E. allocation_area_macro_stable.png
   - macro_stable 月度权重堆叠面积图
F. turnover_timeseries.png
   - baseline 与 macro_stable 的月度换手时间序列对比
G. cash_weight_comparison.png
   - baseline 与 macro_stable 的现金权重时间序列对比（同一张图两条线）

【summary.md（必须写得像可直接放进报告的文字）】
在 showcase_output/summary.md 输出一页纸内容，包含：
1) 我们做了什么：baseline 风险平价 vs 宏观驱动稳定版（宏观信号仅用于调整风险预算，再经协方差得到权重）
2) 关键约束与设定（用“我们使用/我们设置”口吻）：
   - 月度调仓
   - 风险预算思想（risk parity / 风险贡献）
   - 目标波动率参照 TARGET_VOL=0.05（说明是参照/缩放设置）
   - 单资产权重上限、单次权重变化限制（若weights里体现了上限/限速，就如实描述；若代码无法确认来源，请不要臆造“模型内硬约束”，只能描述“结果上观察到”）
3) 结果对比（用你生成的 perf_comparison.csv 中数据自动填充）：
   - 回测段（2018-2023）与验证段（2024-2025）的 return/vol/sharpe/maxDD/calmar
   - 宏观版本相对baseline的提升点与代价（如换手、现金占比变化）
4) 一段“如何解释宏观版更波动/更集中（如果存在）”的谨慎说明：强调宏观信号带来风险预算再分配，可能导致更高权益暴露/更高信用暴露，从而改变风险结构。

【实现要求】
- 只使用标准库 + numpy/pandas/matplotlib。
- 代码必须可直接运行：python generate_showcase_outputs.py
- 需要在终端打印清晰日志：读取路径、输出路径、生成了哪些文件。
- 对日期字段要健壮：允许 '2016/1/29' 或 '2016-01-29'，统一解析为 pandas datetime，并排序去重。
- 对齐频率：weights 是月度、nav 是日度。图表按各自频率绘制，不要强行合并到同频。
- 不要修改任何原始CSV；只读入并生成新输出。
- 所有输出文件名必须与上述一致。

最后，请给出 generate_showcase_outputs.py 的完整代码。