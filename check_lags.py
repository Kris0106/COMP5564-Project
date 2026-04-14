import pandas as pd
import numpy as np
import os

# Path to the data
data_path = '/Users/kris/Desktop/COMP5564/Project2/archive (1)/individual_stocks_5yr/individual_stocks_5yr/'
all_stocks_file = '/Users/kris/Desktop/COMP5564/Project2/archive (1)/all_stocks_5yr.csv'
stocks = ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'NFLX']

def load_market_data(file_path):
    df_all = pd.read_csv(file_path)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['ret'] = df_all.groupby('Name')['close'].pct_change()
    market_benchmark = df_all.groupby('date')['ret'].mean()
    return market_benchmark

market_benchmark = load_market_data(all_stocks_file)

def add_features_final(df, benchmark):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['Market_Return'] = benchmark
    df['Log_Ret'] = np.log(df['close'] / (df['close'].shift(1) + 1e-9))
    df['Relative_Strength'] = df['Log_Ret'] - df['Market_Return']
    df['Target_Alpha'] = df['Relative_Strength'].shift(-1)
    for i in [1, 2, 3, 5]:
        df[f'Lag_LogRet_{i}'] = df['Log_Ret'].shift(i)
    return df.dropna()

all_data = []
for stock in stocks:
    file_path = os.path.join(data_path, f'{stock}_data.csv')
    df = pd.read_csv(file_path)
    df_processed = add_features_final(df, market_benchmark)
    all_data.append(df_processed)

combined_df = pd.concat(all_data)

# Check correlations
lag_features = ['Lag_LogRet_1', 'Lag_LogRet_2', 'Lag_LogRet_3', 'Lag_LogRet_5']
correlations = combined_df[lag_features + ['Target_Alpha']].corr()['Target_Alpha'].sort_values(ascending=False)

print("Correlations with Target_Alpha:")
print(correlations)
