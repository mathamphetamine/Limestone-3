import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load data for Table 3
df = pd.read_csv("data/train.csv", header=[0, 1, 2])

# Extract Table 3 player spy and card values
player_spy_col = df[('table_3', 'player', 'spy')]
player_card_col = df[('table_3', 'player', 'card')]
dealer_spy_col = df[('table_3', 'dealer', 'spy')]
dealer_card_col = df[('table_3', 'dealer', 'card')]

# Convert to numpy arrays for easier analysis
player_spy = player_spy_col.values
player_card = player_card_col.values
dealer_spy = dealer_spy_col.values
dealer_card = dealer_card_col.values

# Print basic statistics
print("Table 3 Player Spy Statistics:")
print(f"Mean: {np.mean(player_spy):.2f}")
print(f"Std: {np.std(player_spy):.2f}")
print(f"Min: {np.min(player_spy):.2f}")
print(f"Max: {np.max(player_spy):.2f}")
print(f"First 20 values: {player_spy[:20]}")

# Print card statistics
print("\nTable 3 Player Card Statistics:")
print(f"Unique values: {np.unique(player_card)}")
print(f"Frequency: {np.bincount(player_card.astype(int))}")

# Analyze auto-correlation for different lag values
lags = [1, 2, 3, 4, 5]
print("\nPlayer Spy Auto-correlations:")
for lag in lags:
    correlation = np.corrcoef(player_spy[:-lag], player_spy[lag:])[0, 1]
    print(f"Lag {lag}: {correlation:.4f}")

# Check for cross-correlation with dealer values
print("\nCross-correlation with dealer spy values:")
for lag in range(-3, 4):  # Check lags from -3 to 3
    if lag < 0:
        # Player lags behind dealer
        corr_data = np.corrcoef(player_spy[:lag], dealer_spy[-lag:])[0, 1]
        print(f"Player behind dealer by {-lag}: {corr_data:.4f}")
    elif lag > 0:
        # Dealer lags behind player
        corr_data = np.corrcoef(player_spy[:-lag], dealer_spy[lag:])[0, 1]
        print(f"Dealer behind player by {lag}: {corr_data:.4f}")
    else:
        # Contemporaneous
        corr_data = np.corrcoef(player_spy, dealer_spy)[0, 1]
        print(f"Contemporaneous: {corr_data:.4f}")

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

# Check if there's any pattern in card mapping
print("\nAnalyzing spy-card relationship:")
for c in range(1, 12):  # Cards go up to 11
    card_indices = np.where(player_card == c)[0]
    if len(card_indices) > 0:
        card_spy_values = player_spy[card_indices]
        print(f"Card {c}: Mean spy = {np.mean(card_spy_values):.2f}, Std = {np.std(card_spy_values):.2f}, Count = {len(card_indices)}")

# Check for card sequence patterns
print("\nAnalyzing card sequence patterns:")
card_transitions = {}
for i in range(1, len(player_card)):
    prev_card = player_card[i-1]
    curr_card = player_card[i]
    key = f"{prev_card}->{curr_card}"
    if key not in card_transitions:
        card_transitions[key] = []
    
    # Store the spy value change
    card_transitions[key].append(player_spy[i] - player_spy[i-1])

# Report the most common transitions and their effects
sorted_transitions = sorted(card_transitions.items(), key=lambda x: len(x[1]), reverse=True)
for key, diffs in sorted_transitions[:10]:  # Top 10 most common
    print(f"{key}: Count = {len(diffs)}, Mean Diff = {np.mean(diffs):.4f}, Std = {np.std(diffs):.4f}")

# Conditional prediction based on card sequence
print("\nTesting card-sequence conditional prediction:")
errors = []
for i in range(min_hist_len, len(player_spy)):
    hist_spy = player_spy[:i]
    hist_card = player_card[:i]
    
    # Base prediction on last value
    base_pred = hist_spy[-1]
    
    # Adjust based on card transition if we've seen it before
    if i > 0:
        prev_card = hist_card[-1]
        curr_card = player_card[i]
        key = f"{prev_card}->{curr_card}"
        
        if key in card_transitions and len(card_transitions[key]) > 5:
            # Use the average effect of this transition
            adjustment = np.mean(card_transitions[key])
            pred = base_pred + adjustment
        else:
            pred = base_pred
    else:
        pred = base_pred
    
    errors.append((player_spy[i] - pred) ** 2)

card_seq_mse = np.mean(errors)
print(f"Card Sequence Model: MSE = {card_seq_mse:.4f}") 