你现在是量化策略研究员 + 可视化专家。

请基于以下文件生成一套专业、统一风格、可用于答辩展示的图表输出：

输入文件：
1) 回测引擎/backtest_results.csv
2) 回测引擎/nav_series.csv
3) 回测引擎/weights.csv
4) 回测引擎_macro_stable/backtest_results_macro_stable.csv
5) 回测引擎_macro_stable/nav_series_macro_stable.csv
6) 回测引擎_macro_stable/weights_macro_stable.csv

目标：生成“答辩级展示图”，风格统一，金融研究报告风格。

--------------------------------------------------
全局绘图风格要求：
--------------------------------------------------

1) 使用 matplotlib + seaborn
2) 统一风格：
   - sns.set_style("whitegrid")
   - 字体大小：
        title = 16
        axis label = 13
        legend = 11
   - 图尺寸统一： (14, 6)
3) 所有百分比类数据（收益、波动率、回撤、权重、换手率）全部显示为百分比格式
4) legend 统一放在图外右侧
5) 所有图增加：
      训练 / 验证分割线：
      vertical line at 2024-01-01
      并标注 "Validation Start"
6) 所有图片保存为：
      output_showcase/xxx.png
      分辨率 dpi=300
7) 颜色统一：
      baseline = "#1f77b4"
      macro_stable = "#ff7f0e"
      target line = grey dashed

--------------------------------------------------
生成以下图表：
--------------------------------------------------

1️⃣ Allocation Area - Baseline
   - 堆叠面积图
   - y轴 0–100%
   - 标题：
     "Baseline Portfolio Allocation (Monthly)"
   - 资产颜色固定统一（两张 allocation 图必须一致）

2️⃣ Allocation Area - Macro Stable
   - 同样格式
   - 标题：
     "Macro-Driven Stable Allocation (Monthly)"

3️⃣ NAV Comparison
   - baseline vs macro_stable
   - 标题：
     "Normalized NAV Comparison"
   - 图右上角小文字展示：
       Backtest Sharpe / Validation Sharpe

4️⃣ Drawdown Comparison
   - y轴固定范围 -10% 到 0%
   - 标题：
     "Drawdown Comparison"
   - 标出最大回撤点（红色圆点）

5️⃣ Rolling 252D Annualized Volatility
   - 添加 TARGET_VOL=5% 虚线
   - 标题：
     "Rolling 252-Day Annualized Volatility"
   - y轴百分比

6️⃣ Monthly Turnover Comparison
   - 添加 12-month rolling average 细线
   - 标题：
     "Monthly Portfolio Turnover"

7️⃣ Cash Weight Comparison
   - baseline vs macro_stable
   - 添加平均值虚线
   - 标题：
     "Cash Allocation Comparison"

--------------------------------------------------
表格输出：
--------------------------------------------------

生成一个合并对比表：

strategy | annual_return_backtest | annual_return_validation | 
annual_volatility_validation | sharpe_validation | 
max_drawdown_validation | calmar_validation | 
avg_cash | turnover_mean

保存为：
output_showcase/performance_summary.csv

--------------------------------------------------
额外要求：
--------------------------------------------------

- 所有日期必须正确对齐
- 所有图表必须防止 NaN 造成断线
- 所有图必须紧凑布局 tight_layout()
- 输出生成提示说明打印在终端

最后，请给出 generate_showcase_outputs.py 的完整代码。