import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load data for Table 4
df = pd.read_csv("data/train.csv", header=[0, 1, 2])

# Extract Table 4 player spy and card values
player_spy_col = df[('table_4', 'player', 'spy')]
player_card_col = df[('table_4', 'player', 'card')]
dealer_spy_col = df[('table_4', 'dealer', 'spy')]
dealer_card_col = df[('table_4', 'dealer', 'card')]

# Convert to numpy arrays for easier analysis
player_spy = player_spy_col.values
player_card = player_card_col.values
dealer_spy = dealer_spy_col.values
dealer_card = dealer_card_col.values

# Print basic statistics
print("Table 4 Player Spy Statistics:")
print(f"Mean: {np.mean(player_spy):.2f}")
print(f"Std: {np.std(player_spy):.2f}")
print(f"Min: {np.min(player_spy):.2f}")
print(f"Max: {np.max(player_spy):.2f}")
print(f"First 20 values: {player_spy[:20]}")

# Print card statistics
print("\nTable 4 Player Card Statistics:")
print(f"Unique values: {np.unique(player_card)}")
print(f"Frequency: {np.bincount(player_card.astype(int))}")

# Analyze auto-correlation for different lag values
lags = [1, 2, 3, 4, 5]
print("\nPlayer Spy Auto-correlations:")
for lag in lags:
    correlation = np.corrcoef(player_spy[:-lag], player_spy[lag:])[0, 1]
    print(f"Lag {lag}: {correlation:.4f}")

# Check for alternating patterns
def check_alternating_pattern(values):
    diffs = np.diff(values)
    pos_to_neg = np.sum((diffs[:-1] > 0) & (diffs[1:] < 0))
    neg_to_pos = np.sum((diffs[:-1] < 0) & (diffs[1:] > 0))
    total_transitions = len(diffs) - 1
    alternating_ratio = (pos_to_neg + neg_to_pos) / total_transitions if total_transitions > 0 else 0
    return alternating_ratio, diffs

alt_ratio, diffs = check_alternating_pattern(player_spy)
print(f"\nAlternating pattern ratio: {alt_ratio:.4f}")
print(f"Mean diff: {np.mean(diffs):.4f}")
print(f"Std diff: {np.std(diffs):.4f}")
print(f"First 20 diffs: {diffs[:20]}")

# Try different prediction methods
prediction_models = {
    "Last Value": lambda hist: hist[-1],
    "Mean of Last 3": lambda hist: np.mean(hist[-3:]) if len(hist) >= 3 else hist[-1],
    "Mean of Last 5": lambda hist: np.mean(hist[-5:]) if len(hist) >= 5 else hist[-1],
    "Median of Last 3": lambda hist: np.median(hist[-3:]) if len(hist) >= 3 else hist[-1],
    "Median of Last 5": lambda hist: np.median(hist[-5:]) if len(hist) >= 5 else hist[-1],
    "Last Value + Last Diff": lambda hist: hist[-1] + (hist[-1] - hist[-2]) if len(hist) >= 2 else hist[-1],
    "Alternating": lambda hist: hist[-1] - (hist[-1] - hist[-2]) if len(hist) >= 2 else hist[-1],
    "Exp. Weighted Avg (0.7)": lambda hist: 0.7 * hist[-1] + 0.3 * (np.mean(hist[:-1]) if len(hist) > 1 else hist[-1])
}

# Test each model
min_hist_len = 10  # Start predictions after this many observations
print("\nTesting prediction models:")
for name, model in prediction_models.items():
    errors = []
    for i in range(min_hist_len, len(player_spy)):
        hist = player_spy[:i]
        pred = model(hist)
        errors.append((player_spy[i] - pred) ** 2)
    
    mse = np.mean(errors)
    print(f"{name}: MSE = {mse:.4f}")

# Check if there's any pattern in card mapping
print("\nAnalyzing spy-card relationship:")
for c in range(1, 12):  # Cards go up to 11
    card_indices = np.where(player_card == c)[0]
    if len(card_indices) > 0:
        card_spy_values = player_spy[card_indices]
        print(f"Card {c}: Mean spy = {np.mean(card_spy_values):.2f}, Std = {np.std(card_spy_values):.2f}, Count = {len(card_indices)}")

# Test prediction by segmenting the data
def segment_based_prediction(spy_values, window_size=100):
    segments = []
    for i in range(0, len(spy_values), window_size):
        segment = spy_values[i:i+window_size]
        if len(segment) > 10:  # Only consider segments with enough data
            segments.append(segment)
    
    # Train and test models on each segment
    models = prediction_models.copy()
    
    results = {}
    for name in models:
        results[name] = []
    
    for i, segment in enumerate(segments):
        for name, model in models.items():
            segment_errors = []
            for j in range(min_hist_len, len(segment)):
                hist = segment[:j]
                pred = model(hist)
                segment_errors.append((segment[j] - pred) ** 2)
            
            if segment_errors:
                segment_mse = np.mean(segment_errors)
                results[name].append(segment_mse)
    
    # Calculate average MSE for each model across all segments
    for name, mse_values in results.items():
        avg_mse = np.mean(mse_values)
        print(f"Segment-based {name}: Avg MSE = {avg_mse:.4f}")
    
    return results

print("\nSegment-based prediction results:")
segment_results = segment_based_prediction(player_spy)

# Analyze if there are patterns based on card values
def card_based_models(spy_values, card_values):
    # Group spy values by card
    card_spy_groups = {}
    for c in range(1, 12):  # Cards 1-11
        indices = np.where(card_values == c)[0]
        if len(indices) > 10:
            card_spy_groups[c] = spy_values[indices]
    
    # Test models for each card group
    for card, spies in card_spy_groups.items():
        print(f"\nCard {card} predictions:")
        for name, model in prediction_models.items():
            errors = []
            for i in range(min_hist_len, len(spies)):
                hist = spies[:i]
                pred = model(hist)
                errors.append((spies[i] - pred) ** 2)
            
            if errors:
                mse = np.mean(errors)
                print(f"  {name}: MSE = {mse:.4f}")

print("\nCard-based prediction results:")
card_based_models(player_spy, player_card) 