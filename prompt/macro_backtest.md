你现在需要为一个“宏观驱动风险预算”的多资产风险平价模型构建回测引擎。

当前项目结构如下（不要假设其它路径存在）：

project/
    raw_data/
    processed_data/
        asset_price_panel.csv
        asset_return_panel.csv
        macro_panel.csv
    prompt/
    build_asset_price_panel.py
    build_asset_return_panel.py
    build_macro_panel.py
    build_backtest_engine.py
    .venv/

注意：
processed_data 只允许存放清洗后的标准输入数据。
模型输出必须单独目录。

--------------------------------------------------
请新建脚本：

build_backtest_macro.py

--------------------------------------------------
【目标】

在现有“月度调仓 + 动态资产池 + 目标波动率缩放”的基础上，
加入宏观驱动的风险预算调整机制。

宏观信号不直接决定权重，
只用于调整风险预算，
最终权重仍通过协方差矩阵计算。

--------------------------------------------------
【数据读取】

1）读取日频收益率：

processed_data/asset_return_panel.csv

要求：
- 解析 date 为 datetime
- 升序排序
- 不填充 credit_bond 缺失值
- 动态资产池自然决定纳入时间

2）读取宏观数据：

processed_data/macro_panel.csv

列：
date, PMI, CPI, CN10Y

要求：
- 月度数据
- 升序排序
- 不使用未来数据

--------------------------------------------------
【回测设置】

- 月度调仓（每月最后一个交易日）
- 协方差窗口：252日
- 权重约束：w_i >= 0
- 不允许杠杆
- 目标年化波动率：0.05
- 若 vol > 0.05：
    scale = 0.05 / vol
    若 scale > 1，则 scale = 1
- 剩余权重为 cash

--------------------------------------------------
【宏观状态识别】

对每个调仓月：

计算历史滚动中位数（仅使用过去数据）：
- median_PMI
- median_CPI
- median_CN10Y

定义三维信号：
- growth_signal = 1 if PMI > median_PMI else 0
- inflation_signal = 1 if CPI > median_CPI else 0
- rate_signal = 1 if CN10Y > median_CN10Y else 0

--------------------------------------------------
【基础风险预算】

hs300        0.20
sp500        0.20
cgb10y       0.20
gold         0.20
policy_bond  0.10
credit_bond  0.10

--------------------------------------------------
【宏观驱动调整逻辑】

若 growth_signal == 1：
    hs300 +0.05
    sp500 +0.05
    从债券中等比例扣除

若 inflation_signal == 1：
    gold +0.05
    从债券中扣除

若 rate_signal == 1：
    cgb10y -0.05
    policy_bond -0.05
    增加 cash 或 credit_bond

调整后：
- 不允许负权重
- 重新归一化风险预算，使总和=1

--------------------------------------------------
【风险平价计算】

1. 用协方差矩阵
2. 按风险预算求解权重
3. 计算组合年化波动率
4. 执行目标波动率缩放
5. 记录现金比例

--------------------------------------------------
【统计区间】

2018-01-01 至 2023-12-31
2024-01-01 至 2025-12-01

输出指标：
- annual_return
- annual_volatility
- sharpe
- max_drawdown
- calmar

--------------------------------------------------
【输出目录】

创建：

回测引擎_macro/

保存：

weights_macro.csv
nav_series_macro.csv
backtest_results_macro.csv

--------------------------------------------------
【终端必须打印】

1）第一个调仓日
2）宏观状态切换次数
3）最大单次权重变化
4）平均现金比例
5）两阶段绩效表
6）两阶段年化波动率是否 <= 0.05

--------------------------------------------------
代码要求：

- 仅使用 pandas + numpy
- 不使用黑箱优化库
- 不使用机器学习
- 不允许未来函数
- 结构清晰
- 注释完整