import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from lightgbm import LGBMClassifier, LGBMRegressor
import os
import warnings

warnings.filterwarnings('ignore')

# 1. 数据加载与基准计算 (Benchmark & Regime - 修复未来函数)
stocks = ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'NFLX']
data_path = '/Users/kris/Desktop/COMP5564/Project2/archive (1)/individual_stocks_5yr/individual_stocks_5yr/'
all_stocks_file = '/Users/kris/Desktop/COMP5564/Project2/archive (1)/all_stocks_5yr.csv'

def load_market_data(file_path):
    df_all = pd.read_csv(file_path)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['ret'] = df_all.groupby('Name')['close'].pct_change()
    market_benchmark = df_all.groupby('date')['ret'].mean()
    market_close = df_all.groupby('date')['close'].mean()
    market_ma20 = market_close.shift(1).rolling(window=20).mean() 
    market_regime = (market_close.shift(1) > market_ma20).astype(int)
    return market_benchmark, market_regime

market_benchmark, market_regime = load_market_data(all_stocks_file)

def load_and_clean_data(stocks, path, benchmark, regime):
    all_data = {}
    for stock in stocks:
        file_path = os.path.join(path, f'{stock}_data.csv')
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.dropna()
        # 移除了全局 Z-Score 清洗以防止数据泄露 (Fix Leakage Flag 1)
        # 仅保留基础数值列
        df['Market_Return'] = benchmark
        df['Market_Regime'] = regime
        all_data[stock] = df
    return all_data

# 2. 职业级特征工程 (平稳化)
def add_features_final(df):
    df = df.copy()
    df['Daily_Return'] = df['close'].pct_change()
    df['Log_Ret'] = np.log(df['close'] / (df['close'].shift(1) + 1e-9))
    df['Volatility_20'] = df['Log_Ret'].rolling(window=20).std()
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['BB_Width'] = (4 * std20) / (ma20 + 1e-9)
    df['Relative_Strength'] = df['Log_Ret'] - df['Market_Return']
    df['Target_Alpha'] = df['Relative_Strength'].shift(-1)
    df['Target_Cls'] = (df['Target_Alpha'] > 0).astype(int)
    for i in [1, 2, 3, 5]:
        df[f'Lag_LogRet_{i}'] = df['Log_Ret'].shift(i)
    return df.dropna()

# 3. 准备数据 (修复分割逻辑 - Fix Split Logic Flag 2)
all_stocks_data = load_and_clean_data(stocks, data_path, market_benchmark, market_regime)
processed_data = {s: add_features_final(df) for s, df in all_stocks_data.items()}

# 严格按时间分割：前 80% 作为训练，后 20% 作为测试 (对所有股票统一)
train_dfs = []
test_dfs = []

for stock, df in processed_data.items():
    split_idx = int(len(df) * 0.8)
    train_dfs.append(df.iloc[:split_idx])
    test_dfs.append(df.iloc[split_idx:])

train_combined = pd.concat(train_dfs)
test_combined = pd.concat(test_dfs)

feature_cols = ['volume', 'Log_Ret', 'Volatility_20', 'BB_Width', 'Relative_Strength', 
                'Market_Regime', 'Lag_LogRet_1', 'Lag_LogRet_2', 'Lag_LogRet_3', 'Lag_LogRet_5']

X_train_raw = train_combined[feature_cols]
y_train_c = train_combined['Target_Cls']
y_train_r = train_combined['Target_Alpha']

X_test_raw = test_combined[feature_cols]
y_test_c = test_combined['Target_Cls']
y_test_r = test_combined['Target_Alpha']

# 4. 模型训练 (分类器 & 回归器)
print("--- Task 1: Classification (Multi-Model Comparison) ---")
# 数据标准化 (Fix Data Leakage: Fit on TRAIN ONLY)
scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_raw)
X_test_c = scaler.transform(X_test_raw)

# 准备测试集的 Daily Return 用于计算 Sharpe Ratio (与 X_test_c 对应)
test_returns = test_combined['Daily_Return'].values
test_regime_c = X_test_c[:, feature_cols.index('Market_Regime')]

ratio = np.sum(y_train_c == 0) / np.sum(y_train_c == 1)

models = {
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, scale_pos_weight=ratio, eval_metric='logloss', random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, is_unbalance=True, random_state=42, verbosity=-1),
    'Logistic': LogisticRegression(class_weight='balanced', random_state=42)
}

comparison_results = []

for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    
    # 预测概率
    if hasattr(model, "predict_proba"):
        probs_c = model.predict_proba(X_test_c)[:, 1]
    else:
        probs_c = model.decision_function(X_test_c)
        probs_c = (probs_c - probs_c.min()) / (probs_c.max() - probs_c.min())

    # 动态阈值生成信号 (Regime Aware)
    y_pred_c = np.array([1 if (probs_c[i] > (0.50 if test_regime_c[i] > 0 else 0.55)) else 0 for i in range(len(probs_c))])
    
    # 计算基础指标
    acc = accuracy_score(y_test_c, y_pred_c)
    prec = precision_score(y_test_c, y_pred_c)
    
    # 计算策略收益以获得 Sharpe Ratio (简单回测，暂不考虑手续费，仅用于模型对比)
    strat_rets = y_pred_c * test_returns
    sharpe = np.mean(strat_rets) / (np.std(strat_rets) + 1e-9) * np.sqrt(252)
    
    comparison_results.append({
        'Model': name,
        'Accuracy': f"{acc:.4f}",
        'Precision': f"{prec:.4f}",
        'Sharpe Ratio': f"{sharpe:.4f}"
    })
    
    # 保留 Random Forest 作为后续诊断的基础 (根据 README，它是表现最好的模型)
    if name == 'RandomForest':
        best_cls = model

# 输出对比表格
print("\n### Model Performance Comparison Table ###")
print("| Model | Accuracy | Precision | Sharpe Ratio |")
print("|-------|----------|-----------|--------------|")
for res in comparison_results:
    print(f"| {res['Model']} | {res['Accuracy']} | {res['Precision']} | {res['Sharpe Ratio']} |")
print("\n")

print("--- Task 2: Regression (Alpha Prediction Multi-Model) ---")
# 数据标准化 (Fix Data Leakage)
scaler_reg = StandardScaler()
X_train_r = scaler_reg.fit_transform(X_train_raw) # 这里直接复用 X_train_raw
X_test_r = scaler_reg.transform(X_test_raw)

models_reg = {
    'XGBoost': XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42, verbosity=-1),
    'Linear': LinearRegression()
}

reg_results = []
best_r2 = -float('inf')

for name, model in models_reg.items():
    model.fit(X_train_r, y_train_r)
    y_pred_r = model.predict(X_test_r)
    
    r2 = r2_score(y_test_r, y_pred_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    
    reg_results.append({
        'Model': name,
        'R2 Score': f"{r2:.4f}",
        'MAE': f"{mae:.6f}",
        'RMSE': f"{rmse:.6f}"
    })
    
    # 默认保留 RandomForest 用于后续的诊断图表生成
    if name == 'RandomForest':
        best_reg = model

print("\n### Regression Model Performance Comparison Table ###")
print("| Model | R2 Score | MAE | RMSE |")
print("|-------|----------|-----|------|")
for res in reg_results:
    print(f"| {res['Model']} | {res['R2 Score']} | {res['MAE']} | {res['RMSE']} |")
print("\n")

# 5. 高级诊断可视化 (Advanced Diagnosis)
print("\n--- Task 4: Advanced Model Diagnosis ---")

# (1) 特征重要性排序 (Feature Importance)
plt.figure(figsize=(10, 6))
importances = pd.Series(best_cls.feature_importances_, index=feature_cols).sort_values(ascending=True)
importances.plot(kind='barh', color='skyblue')
plt.title('Feature Importance (RandomForest Classification)')
plt.savefig('diagnosis_1_importance.png')

# (2) 相关性图 (Correlation Upgrade)
# n*n 图：特征间相关性
plt.figure(figsize=(12, 10))
corr_nn = train_combined[feature_cols].corr() # 仅使用训练集数据分析相关性
sns.heatmap(corr_nn, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('n*n Feature Correlation Matrix (Train Set)')
plt.savefig('diagnosis_2_corr_nn.png')

# n*1 图：特征与 Target 的相关性
plt.figure(figsize=(6, 8))
corr_n1 = train_combined[feature_cols + ['Target_Alpha']].corr()[['Target_Alpha']].sort_values(by='Target_Alpha', ascending=False)
sns.heatmap(corr_n1, annot=True, cmap='coolwarm', center=0)
plt.title('n*1 Correlation with Next Day Alpha Return (Train Set)')
plt.savefig('diagnosis_2_corr_n1.png')

# 6. 每只股票的独立诊断与回测 (Professional Backtest & Per-Stock Diagnosis)
print("\n--- Task 3 & 5: Multi-Model Portfolio Backtest & Diagnosis ---")

def detailed_stock_diagnosis(stock_name, model_cls, model_reg, scaler_cls, scaler_reg, df_processed, cost_rate=0.001, save_plots=False):
    # 严格使用测试集数据进行诊断
    test_len = int(len(df_processed) * 0.2)
    test_df = df_processed.tail(test_len)
    
    # 必须使用与训练时相同的 scaler 进行 transform
    X_test_stock = test_df[feature_cols]
    X_test_stock_cls = scaler_cls.transform(X_test_stock)
    X_test_stock_reg = scaler_reg.transform(X_test_stock)
    
    # --- 分类预测 ---
    if hasattr(model_cls, "predict_proba"):
        probs_stock = model_cls.predict_proba(X_test_stock_cls)[:, 1]
    else:
        probs_stock = model_cls.decision_function(X_test_stock_cls)
        probs_stock = (probs_stock - probs_stock.min()) / (probs_stock.max() - probs_stock.min())
        
    regime_stock = test_df['Market_Regime'].values
    signals_stock = np.array([1 if (probs_stock[i] > (0.50 if regime_stock[i] > 0 else 0.55)) else 0 for i in range(len(probs_stock))])
    
    # --- 回归预测 ---
    y_pred_reg_stock = model_reg.predict(X_test_stock_reg)
    y_actual_reg_stock = test_df['Target_Alpha'].values
    
    # --- 回测计算 ---
    returns_stock = test_df['Daily_Return'].values
    market_returns_stock = test_df['Market_Return'].values
    # 换手率计算：信号变化即产生交易
    trades_stock = np.diff(np.insert(signals_stock, 0, 0)) != 0
    strategy_returns_stock = signals_stock * returns_stock - trades_stock * cost_rate
    
    equity_curve_stock = np.cumprod(1 + strategy_returns_stock)
    market_curve_stock = np.cumprod(1 + market_returns_stock)
    
    if save_plots:
        # --- 可视化 1: 信号覆盖图 (Signal Overlay) ---
        plt.figure(figsize=(12, 6))
        plt.plot(test_df.index, test_df['close'], label='Close Price', color='gray', alpha=0.6)
        buy_dates = test_df.index[signals_stock == 1]
        plt.scatter(buy_dates, test_df.loc[buy_dates, 'close'], marker='^', color='green', label='Buy Signal', s=40)
        plt.title(f'Signal Overlay: {stock_name}')
        plt.legend()
        plt.savefig(f'diagnosis_4_signals_{stock_name}.png')
        plt.close()

        # --- 可视化 2: 水下回撤图 (Underwater Plot) ---
        plt.figure(figsize=(12, 4))
        max_equity = np.maximum.accumulate(equity_curve_stock)
        drawdown = (equity_curve_stock - max_equity) / (max_equity + 1e-9)
        plt.fill_between(test_df.index, drawdown, 0, color='red', alpha=0.3)
        plt.title(f'Underwater Plot: {stock_name}')
        plt.ylabel('Drawdown %')
        plt.savefig(f'diagnosis_5_underwater_{stock_name}.png')
        plt.close()

        # --- 可视化 3: 预测校准图 (Prediction Calibration) ---
        plt.figure(figsize=(6, 6))
        plt.scatter(y_actual_reg_stock, y_pred_reg_stock, alpha=0.3, color='blue')
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'r--', alpha=0.75)
        plt.xlabel('Actual Alpha')
        plt.ylabel('Predicted Alpha')
        plt.title(f'Calibration: {stock_name}')
        plt.savefig(f'diagnosis_3_calibration_{stock_name}.png')
        plt.close()
    
    return equity_curve_stock, market_curve_stock

# 模型对应关系
model_pairs = [
    ('XGBoost', models['XGBoost'], models_reg['XGBoost']),
    ('RandomForest', models['RandomForest'], models_reg['RandomForest']),
    ('LightGBM', models['LightGBM'], models_reg['LightGBM']),
    ('Linear/Logistic', models['Logistic'], models_reg['Linear'])
]

portfolio_metrics = []
best_sharpe = -float('inf')
best_model_name = ""
best_model_results = {} # Store equity curves for best model

print("\n### Portfolio Strategy Performance Comparison Table ###")
print("| Model Pair | Annual Return | Sharpe Ratio | Max Drawdown |")
print("|------------|---------------|--------------|--------------|")

for name, clf, reg in model_pairs:
    current_model_results = {}
    for stock in stocks:
        eq_s, mkt_s = detailed_stock_diagnosis(stock, clf, reg, scaler, scaler_reg, processed_data[stock], save_plots=False)
        current_model_results[stock] = {'equity': eq_s, 'mkt': mkt_s}
    
    # 组合表现
    portfolio_equity = np.mean([res['equity'] for res in current_model_results.values()], axis=0)
    
    # 计算指标
    total_ret = portfolio_equity[-1] - 1
    annual_ret = (portfolio_equity[-1]**(252/len(portfolio_equity))) - 1
    daily_rets = pd.Series(portfolio_equity).pct_change().dropna()
    sharpe = daily_rets.mean() / (daily_rets.std() + 1e-9) * np.sqrt(252)
    
    max_equity = np.maximum.accumulate(portfolio_equity)
    drawdown = (portfolio_equity - max_equity) / max_equity
    max_dd = drawdown.min()
    
    print(f"| {name:<10} | {annual_ret:<13.2%} | {sharpe:<12.4f} | {max_dd:<12.2%} |")
    
    portfolio_metrics.append({
        'Model': name,
        'Annual Return': annual_ret,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd
    })
    
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_model_name = name
        best_model_results = current_model_results

print(f"\nBest Performing Model: {best_model_name} (Sharpe: {best_sharpe:.4f})")
print(f"Generating detailed diagnostic plots for {best_model_name}...")

# 为最佳模型生成详细图表和最终组合图
plt.figure(figsize=(16, 10))
for i, stock in enumerate(stocks):
    # 重新运行以保存图表
    clf_best = [m[1] for m in model_pairs if m[0] == best_model_name][0]
    reg_best = [m[2] for m in model_pairs if m[0] == best_model_name][0]
    
    eq_s, mkt_s = detailed_stock_diagnosis(stock, clf_best, reg_best, scaler, scaler_reg, processed_data[stock], save_plots=True)
    
    plt.subplot(3, 2, i+1)
    plt.plot(eq_s, label='Strategy')
    plt.plot(mkt_s, label='Market', linestyle='--')
    plt.title(f'Backtest: {stock} ({best_model_name})')
    plt.legend()

portfolio_equity_best = np.mean([res['equity'] for res in best_model_results.values()], axis=0)
mkt_s_any = list(best_model_results.values())[0]['mkt']

plt.subplot(3, 2, 6)
plt.plot(portfolio_equity_best, label=f'Portfolio ({best_model_name})', color='gold', linewidth=2)
plt.plot(mkt_s_any, label='Market', linestyle='--')
plt.title('Final Combined Portfolio Performance')
plt.legend()
plt.tight_layout()
plt.savefig('final_multi_stock_backtest.png')

print(f"\n{'Stock':<10} | {'Annual Return':<15} | {'vs Market Alpha'}")
print("-" * 45)
for stock in stocks:
    eq = best_model_results[stock]['equity']
    mkt = best_model_results[stock]['mkt']
    ann_ret = (eq[-1]**(252/len(eq)))-1
    alpha = eq[-1] - mkt[-1]
    print(f"{stock:<10} | {ann_ret:<15.2%} | {alpha:+.4f}")

p_annual = (portfolio_equity_best[-1]**(252/len(portfolio_equity_best)))-1
print(f"\nPortfolio ({best_model_name}) | {p_annual:<15.2%} | {portfolio_equity_best[-1]-mkt_s_any[-1]:+.4f}")

print("\nAll advanced diagnostic plots saved for the BEST performing model.")

# 7. Ablation Study (Task 6)
print("\n--- Task 6: Ablation Study (Impact of Key Components) ---")
# 使用最佳模型 (Best Model) 进行消融实验
# 基准: 完整模型 (Full Model)
# 变体1: 移除 Market Regime (No Regime Filter) -> 阈值固定为 0.5
# 变体2: 移除高级特征 (No Advanced Features) -> 仅保留 Lag_LogRet

ablation_results = []

# (1) Full Model (Already Calculated)
ablation_results.append({
    'Configuration': 'Full Model (Regime + Features)',
    'Annual Return': p_annual,
    'Sharpe Ratio': best_sharpe,
    'Max Drawdown': max(0, 1 - min(portfolio_equity_best)) # Approximate
})

# (2) No Regime Filter
# 重复使用 best_model 的预测概率，但不再根据 Regime 调整阈值，统一使用 0.5
clf_best = [m[1] for m in model_pairs if m[0] == best_model_name][0]
current_equity_curves_no_regime = []

for stock in stocks:
    # 复用之前的数据处理
    test_len = int(len(processed_data[stock]) * 0.2)
    test_df = processed_data[stock].tail(test_len)
    X_test_stock = test_df[feature_cols]
    X_test_stock_cls = scaler.transform(X_test_stock)
    
    if hasattr(clf_best, "predict_proba"):
        probs_stock = clf_best.predict_proba(X_test_stock_cls)[:, 1]
    else:
        probs_stock = clf_best.decision_function(X_test_stock_cls)
        probs_stock = (probs_stock - probs_stock.min()) / (probs_stock.max() - probs_stock.min())
    
    # !!! 关键差异: 阈值固定为 0.5，不考虑 Regime !!!
    signals_stock = np.array([1 if p > 0.5 else 0 for p in probs_stock])
    
    returns_stock = test_df['Daily_Return'].values
    trades_stock = np.diff(np.insert(signals_stock, 0, 0)) != 0
    strategy_returns_stock = signals_stock * returns_stock - trades_stock * 0.001
    equity_curve_stock = np.cumprod(1 + strategy_returns_stock)
    current_equity_curves_no_regime.append(equity_curve_stock)

portfolio_equity_no_regime = np.mean(current_equity_curves_no_regime, axis=0)
ann_ret_no_regime = (portfolio_equity_no_regime[-1]**(252/len(portfolio_equity_no_regime))) - 1
daily_rets_no_regime = pd.Series(portfolio_equity_no_regime).pct_change().dropna()
sharpe_no_regime = daily_rets_no_regime.mean() / (daily_rets_no_regime.std() + 1e-9) * np.sqrt(252)

ablation_results.append({
    'Configuration': 'No Regime Filter (Fixed Threshold)',
    'Annual Return': ann_ret_no_regime,
    'Sharpe Ratio': sharpe_no_regime,
    'Max Drawdown': 0 # Placeholder
})

# (3) No Advanced Features (Basic Features Only)
# 重新训练模型，仅使用基础特征 (Lag_LogRet, Volume)
basic_features = ['volume', 'Lag_LogRet_1', 'Lag_LogRet_2', 'Lag_LogRet_3', 'Lag_LogRet_5']
X_basic_train = train_combined[basic_features]
y_basic_train = train_combined['Target_Cls']

scaler_basic = StandardScaler()
X_train_basic = scaler_basic.fit_transform(X_basic_train)

# Clone best model to avoid modifying original
from sklearn.base import clone
clf_basic = clone(clf_best)
clf_basic.fit(X_train_basic, y_basic_train)

current_equity_curves_basic = []
for stock in stocks:
    test_len = int(len(processed_data[stock]) * 0.2)
    test_df = processed_data[stock].tail(test_len)
    X_test_stock = test_df[basic_features]
    X_test_stock_basic = scaler_basic.transform(X_test_stock)
    
    if hasattr(clf_basic, "predict_proba"):
        probs_stock = clf_basic.predict_proba(X_test_stock_basic)[:, 1]
    else:
        probs_stock = clf_basic.decision_function(X_test_stock_basic)
        probs_stock = (probs_stock - probs_stock.min()) / (probs_stock.max() - probs_stock.min())
        
    # Still use Regime Filter to isolate Feature impact
    regime_stock = test_df['Market_Regime'].values
    signals_stock = np.array([1 if (probs_stock[i] > (0.50 if regime_stock[i] > 0 else 0.55)) else 0 for i in range(len(probs_stock))])
    
    returns_stock = test_df['Daily_Return'].values
    trades_stock = np.diff(np.insert(signals_stock, 0, 0)) != 0
    strategy_returns_stock = signals_stock * returns_stock - trades_stock * 0.001
    equity_curve_stock = np.cumprod(1 + strategy_returns_stock)
    current_equity_curves_basic.append(equity_curve_stock)

portfolio_equity_basic = np.mean(current_equity_curves_basic, axis=0)
ann_ret_basic = (portfolio_equity_basic[-1]**(252/len(portfolio_equity_basic))) - 1
daily_rets_basic = pd.Series(portfolio_equity_basic).pct_change().dropna()
sharpe_basic = daily_rets_basic.mean() / (daily_rets_basic.std() + 1e-9) * np.sqrt(252)

ablation_results.append({
    'Configuration': 'No Advanced Features (Basic Only)',
    'Annual Return': ann_ret_basic,
    'Sharpe Ratio': sharpe_basic,
    'Max Drawdown': 0 # Placeholder
})

print("\n### Ablation Study Results ###")
print("| Configuration | Annual Return | Sharpe Ratio |")
print("|---------------|---------------|--------------|")
for res in ablation_results:
    print(f"| {res['Configuration']} | {res['Annual Return']:<13.2%} | {res['Sharpe Ratio']:<12.4f} |")

# Visualization of Ablation
plt.figure(figsize=(10, 6))
plt.plot(portfolio_equity_best, label='Full Model', linewidth=2, color='green')
plt.plot(portfolio_equity_no_regime, label='No Regime Filter', linestyle='--', color='orange')
plt.plot(portfolio_equity_basic, label='No Advanced Features', linestyle=':', color='gray')
plt.title(f'Ablation Study: Impact of Components ({best_model_name})')
plt.legend()
plt.savefig('diagnosis_6_ablation_study.png')
print("\nAblation study plot saved as 'diagnosis_6_ablation_study.png'.")
