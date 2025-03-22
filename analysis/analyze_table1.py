import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load data for Table 1
df = pd.read_csv("data/train.csv", header=[0, 1, 2])

# Extract Table 1 player spy and card values
player_spy_col = df[('table_1', 'player', 'spy')]
player_card_col = df[('table_1', 'player', 'card')]
dealer_spy_col = df[('table_1', 'dealer', 'spy')]
dealer_card_col = df[('table_1', 'dealer', 'card')]

# Convert to numpy arrays for easier analysis
player_spy = player_spy_col.values
player_card = player_card_col.values
dealer_spy = dealer_spy_col.values
dealer_card = dealer_card_col.values

# Print basic statistics
print("Table 1 Player Spy Statistics:")
print(f"Mean: {np.mean(player_spy):.2f}")
print(f"Std: {np.std(player_spy):.2f}")
print(f"Min: {np.min(player_spy):.2f}")
print(f"Max: {np.max(player_spy):.2f}")
print(f"First 20 values: {player_spy[:20]}")

# Print card statistics
print("\nTable 1 Player Card Statistics:")
print(f"Unique values: {np.unique(player_card)}")
print(f"Frequency: {np.bincount(player_card.astype(int))}")

# Analyze auto-correlation for different lag values
lags = [1, 2, 3, 4, 5]
print("\nPlayer Spy Auto-correlations:")
for lag in lags:
    correlation = np.corrcoef(player_spy[:-lag], player_spy[lag:])[0, 1]
    print(f"Lag {lag}: {correlation:.4f}")

# Create segments for prediction and analyze them
def analyze_segments(values, window=100):
    segments = []
    for i in range(0, len(values), window):
        segment = values[i:i+window]
        if len(segment) > 10:  # Require at least 10 values per segment
            segments.append(segment)
    
    # Calculate statistics for each segment
    for i, segment in enumerate(segments):
        print(f"Segment {i}: Mean={np.mean(segment):.2f}, Std={np.std(segment):.2f}")
        
        # Calculate autocorrelation within segment
        if len(segment) > 5:
            ac1 = np.corrcoef(segment[:-1], segment[1:])[0, 1]
            ac2 = np.corrcoef(segment[:-2], segment[2:])[0, 1] if len(segment) > 2 else 0
            print(f"  AC(1)={ac1:.4f}, AC(2)={ac2:.4f}")
    
    return segments

# Analyze data in segments
print("\nSegment Analysis:")
segments = analyze_segments(player_spy)

# Try different prediction methods
prediction_models = {
    "Last Value": lambda hist: hist[-1],
    "Mean of Last 3": lambda hist: np.mean(hist[-3:]) if len(hist) >= 3 else hist[-1],
    "Mean of Last 5": lambda hist: np.mean(hist[-5:]) if len(hist) >= 5 else hist[-1],
    "Median of Last 3": lambda hist: np.median(hist[-3:]) if len(hist) >= 3 else hist[-1],
    "Median of Last 5": lambda hist: np.median(hist[-5:]) if len(hist) >= 5 else hist[-1],
    "Last Value + Last Diff": lambda hist: hist[-1] + (hist[-1] - hist[-2]) if len(hist) >= 2 else hist[-1],
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

# Check if card values can predict spy values
print("\nAnalyzing spy-card relationship:")
for c in range(1, 12):  # Cards go up to 11
    card_indices = np.where(player_card == c)[0]
    if len(card_indices) > 0:
        card_spy_values = player_spy[card_indices]
        # Calculate mean and standard deviation of spy values for this card
        mean_spy = np.mean(card_spy_values)
        std_spy = np.std(card_spy_values)
        print(f"Card {c}: Mean spy = {mean_spy:.2f}, Std = {std_spy:.2f}, Count = {len(card_indices)}")
        
        # Calculate MSE if we always predict mean for this card
        mse = np.mean((card_spy_values - mean_spy) ** 2)
        print(f"  MSE for mean prediction: {mse:.4f}")

# Analyze card transitions
print("\nAnalyzing card transitions:")
card_transitions = {}
for i in range(1, len(player_card)):
    prev_card = player_card[i-1]
    curr_card = player_card[i]
    key = f"{prev_card}->{curr_card}"
    if key not in card_transitions:
        card_transitions[key] = []
    
    # Store the spy value
    card_transitions[key].append(player_spy[i])

# Find transitions with clear patterns
for key, values in card_transitions.items():
    if len(values) >= 10:
        mean_val = np.mean(values)
        std_val = np.std(values)
        # Look for transitions with low variance
        if std_val < 0.5 * np.std(player_spy):
            print(f"{key}: Mean={mean_val:.2f}, Std={std_val:.2f}, Count={len(values)}")

# Check for patterns between player and dealer
print("\nPlayer-Dealer relationship:")
# Calculate correlation
corr = np.corrcoef(player_spy, dealer_spy)[0, 1]
print(f"Correlation: {corr:.4f}")

# Test if dealer card predicts player spy
dealer_card_player_spy = {}
for i in range(len(dealer_card)):
    d_card = dealer_card[i]
    p_spy = player_spy[i]
    
    if d_card not in dealer_card_player_spy:
        dealer_card_player_spy[d_card] = []
    
    dealer_card_player_spy[d_card].append(p_spy)

for d_card, p_spy_values in dealer_card_player_spy.items():
    if len(p_spy_values) > 10:
        mean_spy = np.mean(p_spy_values)
        std_spy = np.std(p_spy_values)
        print(f"Dealer Card {d_card}: Player Mean Spy = {mean_spy:.2f}, Std = {std_spy:.2f}, Count = {len(p_spy_values)}") 