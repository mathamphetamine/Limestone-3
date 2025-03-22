import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load data for Table 2
df = pd.read_csv("data/train.csv", header=[0, 1, 2])

# Extract Table 2 player spy and card values
# First we need to select the appropriate columns
player_spy_col = df[('table_2', 'player', 'spy')]
player_card_col = df[('table_2', 'player', 'card')]
dealer_spy_col = df[('table_2', 'dealer', 'spy')]
dealer_card_col = df[('table_2', 'dealer', 'card')]

# Convert to numpy arrays for easier analysis
player_spy = player_spy_col.values
player_card = player_card_col.values
dealer_spy = dealer_spy_col.values
dealer_card = dealer_card_col.values

# Print basic statistics
print("Table 2 Player Spy Statistics:")
print(f"Mean: {np.mean(player_spy):.2f}")
print(f"Std: {np.std(player_spy):.2f}")
print(f"Min: {np.min(player_spy):.2f}")
print(f"Max: {np.max(player_spy):.2f}")
print(f"First 20 values: {player_spy[:20]}")

# Print card statistics
print("\nTable 2 Player Card Statistics:")
print(f"Unique values: {np.unique(player_card)}")
print(f"Frequency: {np.bincount(player_card.astype(int))}")

# Analyze auto-correlation for different lag values
lags = [1, 2, 3, 4, 5]
print("\nPlayer Spy Auto-correlations:")
for lag in lags:
    correlation = np.corrcoef(player_spy[:-lag], player_spy[lag:])[0, 1]
    print(f"Lag {lag}: {correlation:.4f}")

# Create a simple linear model based on last value
def simple_persistence_prediction(values):
    X = values[:-1]
    y = values[1:]
    pred = X
    mse = mean_squared_error(y, pred)
    print(f"Simple persistence model MSE: {mse:.4f}")
    return mse

# Try median of last n values
def median_prediction(values, window=3):
    y = values[window:]
    preds = []
    for i in range(len(y)):
        window_vals = values[i:i+window]
        preds.append(np.median(window_vals))
    mse = mean_squared_error(y, preds)
    print(f"Median window {window} model MSE: {mse:.4f}")
    return mse

# Mean of last n values
def mean_prediction(values, window=3):
    y = values[window:]
    preds = []
    for i in range(len(y)):
        window_vals = values[i:i+window]
        preds.append(np.mean(window_vals))
    mse = mean_squared_error(y, preds)
    print(f"Mean window {window} model MSE: {mse:.4f}")
    return mse

# Linear regression on window
def linear_trend_prediction(values, window=5):
    y = values[window:]
    preds = []
    
    for i in range(len(y)):
        window_vals = values[i:i+window]
        # Create X as sequential indices
        X = np.arange(window).reshape(-1, 1)
        # Fit linear regression
        coeffs = np.polyfit(X.flatten(), window_vals, 1)
        # Predict next value based on trend
        pred = coeffs[0] * window + coeffs[1]
        preds.append(pred)
        
    mse = mean_squared_error(y, preds)
    print(f"Linear trend window {window} model MSE: {mse:.4f}")
    return mse

# Try different prediction methods
print("\nTesting prediction models:")
simple_persistence_prediction(player_spy)

for window in [2, 3, 5, 10]:
    median_prediction(player_spy, window)
    mean_prediction(player_spy, window)
    linear_trend_prediction(player_spy, window)

# Look for patterns in player sequences
print("\nChecking for patterns in differences:")
diffs = np.diff(player_spy)
print(f"Mean diff: {np.mean(diffs):.4f}")
print(f"Std diff: {np.std(diffs):.4f}")
print(f"First 20 diffs: {diffs[:20]}")

# Pattern of differences
print("\nDifference patterns:")
second_diffs = np.diff(diffs)
print(f"Mean second diff: {np.mean(second_diffs):.4f}")
print(f"Std second diff: {np.std(second_diffs):.4f}")

# Test prediction using last difference
def last_diff_prediction(values):
    diffs = np.diff(values)
    y = values[1:]
    preds = []
    for i in range(len(y)-1):
        pred = values[i] + diffs[i]
        preds.append(pred)
    mse = mean_squared_error(y[1:], preds)
    print(f"Last diff model MSE: {mse:.4f}")
    return mse

# Average of differences
def avg_diff_prediction(values, window=3):
    diffs = np.diff(values)
    y = values[window:]
    preds = []
    
    for i in range(len(y)-1):
        window_diffs = diffs[i:i+window-1]
        avg_diff = np.mean(window_diffs)
        pred = values[i+window-1] + avg_diff
        preds.append(pred)
        
    mse = mean_squared_error(y[1:], preds)
    print(f"Average diff window {window} model MSE: {mse:.4f}")
    return mse

print("\nTesting difference-based predictions:")
last_diff_prediction(player_spy)
for window in [2, 3, 5]:
    avg_diff_prediction(player_spy, window)

# Check if there's any pattern in card mapping
print("\nAnalyzing spy-card relationship:")
for c in range(1, 12):  # Cards go up to 11
    card_indices = np.where(player_card == c)[0]
    if len(card_indices) > 0:
        card_spy_values = player_spy[card_indices]
        print(f"Card {c}: Mean spy = {np.mean(card_spy_values):.2f}, Std = {np.std(card_spy_values):.2f}, Count = {len(card_indices)}")

# Conditional models based on current value range
print("\nTesting range-based models:")
# Group values by range
ranges = [(-np.inf, 0), (0, 50), (50, 100), (100, np.inf)]
for range_min, range_max in ranges:
    indices = np.where((player_spy >= range_min) & (player_spy < range_max))[0]
    if len(indices) > 1:  # Need at least 2 values for diff
        range_values = player_spy[indices]
        next_indices = indices + 1
        next_indices = next_indices[next_indices < len(player_spy)]
        if len(next_indices) > 0:
            next_values = player_spy[next_indices]
            # Persistence in this range
            mse = mean_squared_error(next_values, range_values[:len(next_values)])
            print(f"Range [{range_min}, {range_max}): Count = {len(indices)}, Persistence MSE = {mse:.4f}")
            
            # Average change in this range
            if len(next_indices) > 1:
                diffs = next_values - range_values[:len(next_values)]
                avg_diff = np.mean(diffs)
                print(f"  Average diff in range: {avg_diff:.4f}") 