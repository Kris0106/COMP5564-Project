# COMP5564 Project 2: 股票市场分析与算法交易系统 (Stock Analysis & Algo Trading)

## 1. 项目概述 (Overview)
本项目是一个专业级的量化交易机器学习流水线，旨在通过多模型融合预测股票的 **Alpha (超额收益)**。项目包含从数据清洗、特征工程、模型训练到回测诊断的完整流程，重点解决了“未来函数 (Look-ahead Bias)”和“数据泄露 (Data Leakage)”等常见的量化陷阱。

## 2. 核心功能 (Key Features)
- **多模型对比架构**: 横向评测 **XGBoost**, **Random Forest**, **LightGBM**, 和 **Logistic/Linear Regression**。
- **市场状态感知 (Market Regime)**: 基于市场趋势（20日均线）动态调整交易阈值（0.50 vs 0.55），在熊市中自动收紧开仓条件。
- **严格的风控设计**:
    - **防止未来函数 (No Look-ahead Bias)**: 所有特征和状态判断严格使用历史数据（Shift处理）。
    - **防止数据泄露 (No Data Leakage)**: 严格遵循 `Train-Test Split -> Scaler Fit` 的顺序，杜绝全量标准化带来的信息泄露。
    - **交易成本模拟**: 回测中包含 0.1% 的单边交易成本。
- **专业级诊断图表**:
    - **Signal Overlay**: 买卖信号与股价叠加图。
    - **Underwater Plot**: 策略回撤深度分析。
    - **Calibration Plot**: 预测值与真实值的校准度分析。
    - **Feature Importance**: 特征重要性与相关性热力图。

## 3. 任务模块 (Tasks)

### Task 1: 分类任务 (Classification)
预测股票在次日是否会跑赢大盘（S&P 500）。
- **评价指标**: Accuracy, Precision, Sharpe Ratio。

### Task 2: 回归任务 (Regression)
预测具体的超额收益率 (Alpha)。
- **评价指标**: R2 Score, MAE, RMSE。

### Task 3 & 5: 组合回测 (Portfolio Backtest)
构建包含 5 只科技股 (AAPL, AMZN, MSFT, GOOGL, NFLX) 的投资组合。
- **策略**: 当预测概率 > 动态阈值时做多。
- **输出**: 年化收益率, 夏普比率 (Sharpe), 最大回撤 (Max Drawdown)。

## 4. 环境安装与运行

### 依赖库
请确保安装以下 Python 库：
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
```

### 运行
```bash
python stock_analysis.py
```

## 5. 性能表现 (Performance)
*基于最新一次的随机森林 (Random Forest) 回测结果（已修复数据泄露）：*
- **年化收益率 (Annual Return)**: ~30.03%
- **夏普比率 (Sharpe Ratio)**: ~4.08
- **最大回撤 (Max Drawdown)**: -2.95%
*(注：RandomForest 在多轮测试中表现出最佳的风险收益比，优于 XGBoost 和 LightGBM)*

## 6. 文件结构
- `stock_analysis.py`: 核心代码，包含所有逻辑。
- `archive (1)/`: 数据文件夹，包含个股和标普500数据。
- `diagnosis_*.png`: 程序运行时自动生成的诊断图表。
- `final_multi_stock_backtest.png`: 最终的组合回测净值曲线。

---

