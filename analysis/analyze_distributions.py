import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load datasets
print("Loading datasets...")
main_data = pd.read_csv('data/train.csv', header=[0,1,2])
sicilian_data = pd.read_csv('data/sicilian_train.csv', header=[0,1,2])

# Analyze main dataset
print("\n=== Main Dataset Analysis ===")
for table_idx in range(5):
    print(f"\nTable {table_idx}:")
    
    # Player cards
    player_cards = main_data[(f'table_{table_idx}', 'player', 'card')]
    player_card_values = sorted(player_cards.unique())
    player_card_counts = np.bincount(player_cards.astype(int))[2:12]
    
    print(f"Player card values: {player_card_values}")
    print("Player card distribution:")
    for card, count in zip(range(2, 12), player_card_counts):
        print(f"  Card {card}: {count} ({count/len(player_cards)*100:.1f}%)")
    
    # Dealer cards
    dealer_cards = main_data[(f'table_{table_idx}', 'dealer', 'card')]
    dealer_card_values = sorted(dealer_cards.unique())
    dealer_card_counts = np.bincount(dealer_cards.astype(int))[2:12]
    
    print(f"Dealer card values: {dealer_card_values}")
    print("Dealer card distribution:")
    for card, count in zip(range(2, 12), dealer_card_counts):
        if count > 0:
            print(f"  Card {card}: {count} ({count/len(dealer_cards)*100:.1f}%)")
    
    # Spy value statistics
    player_spy = main_data[(f'table_{table_idx}', 'player', 'spy')]
    dealer_spy = main_data[(f'table_{table_idx}', 'dealer', 'spy')]
    
    print(f"Player spy range: {player_spy.min():.2f} to {player_spy.max():.2f}, mean: {player_spy.mean():.2f}, std: {player_spy.std():.2f}")
    print(f"Dealer spy range: {dealer_spy.min():.2f} to {dealer_spy.max():.2f}, mean: {dealer_spy.mean():.2f}, std: {dealer_spy.std():.2f}")
    
    # Spy-card relationship
    player_spy_by_card = defaultdict(list)
    for card, spy in zip(player_cards, player_spy):
        player_spy_by_card[card].append(spy)
    
    print("Player card to spy statistics:")
    for card in sorted(player_spy_by_card.keys()):
        spies = player_spy_by_card[card]
        print(f"  Card {card}: mean={np.mean(spies):.2f}, std={np.std(spies):.2f}, min={min(spies):.2f}, max={max(spies):.2f}")

    # Card prediction from spy value
    print("\nCard prediction from spy value analysis:")
    
    # For each table, analyze if there's a strong mapping from spy to card
    player_card_given_spy = defaultdict(list)
    for spy, card in zip(player_spy, player_cards):
        # Round spy value to nearest 0.5
        rounded_spy = round(spy * 2) / 2
        player_card_given_spy[rounded_spy].append(card)
    
    # Calculate the most common card for each spy value
    for spy_val in sorted(player_card_given_spy.keys()):
        cards = player_card_given_spy[spy_val]
        card_counts = np.bincount(cards)
        most_common_card = np.argmax(card_counts)
        accuracy = card_counts[most_common_card] / len(cards) * 100
        if len(cards) > 5:  # Only show if we have enough samples
            print(f"  Player spy={spy_val:.1f} → most common card={most_common_card} (accuracy: {accuracy:.1f}%, samples: {len(cards)})")
    
    print("\nDealer card prediction from spy value:")
    dealer_card_given_spy = defaultdict(list)
    for spy, card in zip(dealer_spy, dealer_cards):
        # Round spy value to the nearest integer for dealer (since it has wider range)
        rounded_spy = round(spy)
        dealer_card_given_spy[rounded_spy].append(card)
    
    # Show a few examples of spy to card mapping for dealer
    predictable_count = 0
    total_spy_values = 0
    
    for spy_val in sorted(dealer_card_given_spy.keys()):
        cards = dealer_card_given_spy[spy_val]
        if len(cards) > 5:  # Only consider spy values with enough samples
            card_counts = np.bincount(cards)
            most_common_card = np.argmax(card_counts)
            accuracy = card_counts[most_common_card] / len(cards) * 100
            total_spy_values += 1
            if accuracy > 80:  # If prediction accuracy is high
                predictable_count += 1
                if total_spy_values <= 5 or accuracy > 95:  # Show first 5 examples or very high accuracy ones
                    print(f"  Dealer spy={spy_val:.1f} → most common card={most_common_card} (accuracy: {accuracy:.1f}%, samples: {len(cards)})")
    
    if total_spy_values > 0:
        print(f"  Dealer spy values with high predictability: {predictable_count}/{total_spy_values} ({predictable_count/total_spy_values*100:.1f}%)")
    
    # Checking for patterns in the dealer spy values
    dealer_spy_values = dealer_spy.values
    
    # Check for sequential patterns
    print("\nSequential pattern analysis in dealer spy values:")
    
    # Calculate autocorrelation for different lags
    print("Autocorrelation at different lags:")
    for lag in range(1, 6):
        if len(dealer_spy_values) > lag:
            autocorr = np.corrcoef(dealer_spy_values[lag:], dealer_spy_values[:-lag])[0, 1]
            print(f"  Lag-{lag}: {autocorr:.4f}")
    
    # Check for alternating patterns in the differences
    if len(dealer_spy_values) > 1:
        diffs = np.diff(dealer_spy_values)
        sign_changes = np.sum(np.sign(diffs[1:]) != np.sign(diffs[:-1]))
        if len(diffs) > 1:
            print(f"  Sign changes in differences: {sign_changes} out of {len(diffs)-1} ({sign_changes/(len(diffs)-1)*100:.1f}%)")
        
        # Check if odd and even indices have different patterns
        if len(diffs) >= 4:
            odd_indices = diffs[1::2]
            even_indices = diffs[::2][:len(odd_indices)]  # Make sure they're the same length
            
            print(f"  Odd-indexed diffs mean: {np.mean(odd_indices):.4f}, std: {np.std(odd_indices):.4f}")
            print(f"  Even-indexed diffs mean: {np.mean(even_indices):.4f}, std: {np.std(even_indices):.4f}")
            
            if np.mean(odd_indices) * np.mean(even_indices) < 0:
                print("  *** ALTERNATING PATTERN DETECTED: Odd and even differences have opposite signs ***")

# Analyze Sicilian dataset
print("\n\n=== Sicilian Dataset Analysis ===")

# Player cards
player_cards = sicilian_data[('special_table', 'player', 'card')]
player_card_values = sorted(player_cards.unique())
player_card_counts = np.bincount(player_cards.astype(int))[2:12]

print(f"Player card values: {player_card_values}")
print("Player card distribution:")
for card, count in zip(range(2, 12), player_card_counts):
    if count > 0:
        print(f"  Card {card}: {count} ({count/len(player_cards)*100:.1f}%)")

# Dealer cards
dealer_cards = sicilian_data[('special_table', 'dealer', 'card')]
dealer_card_values = sorted(dealer_cards.unique())
dealer_card_counts = np.bincount(dealer_cards.astype(int))[2:12]

print(f"Dealer card values: {dealer_card_values}")
print("Dealer card distribution:")
for card, count in zip(range(2, 12), dealer_card_counts):
    if count > 0:
        print(f"  Card {card}: {count} ({count/len(dealer_cards)*100:.1f}%)")

# Spy value statistics
player_spy = sicilian_data[('special_table', 'player', 'spy')]
dealer_spy = sicilian_data[('special_table', 'dealer', 'spy')]

print(f"Player spy range: {player_spy.min():.2f} to {player_spy.max():.2f}, mean: {player_spy.mean():.2f}, std: {player_spy.std():.2f}")
print(f"Dealer spy range: {dealer_spy.min():.2f} to {dealer_spy.max():.2f}, mean: {dealer_spy.mean():.2f}, std: {dealer_spy.std():.2f}")

# Sicilian special analysis
print("\n=== Sicilian Special Analysis ===")
print("Checking for special relationships in the sicilian dataset...")

# Check if there's a relationship between player's spy and dealer's spy
correlation = np.corrcoef(player_spy, dealer_spy)[0, 1]
print(f"Correlation between player spy and dealer spy: {correlation:.4f}")

# Check if we can predict dealer's card from player's spy or card
dealer_card_given_player_spy = defaultdict(list)
for p_spy, d_card in zip(player_spy, dealer_cards):
    # Round player spy to nearest 1.0 given the large range
    rounded_spy = round(p_spy)
    dealer_card_given_player_spy[rounded_spy].append(d_card)

# Show the most predictable spy values
predictable_values = 0
total_values = 0

print("\nDealer card prediction from player spy value:")
for spy_val in sorted(dealer_card_given_player_spy.keys()):
    cards = dealer_card_given_player_spy[spy_val]
    if len(cards) > 10:  # Only consider spy values with enough samples
        card_counts = np.bincount(cards)
        most_common_card = np.argmax(card_counts)
        accuracy = card_counts[most_common_card] / len(cards) * 100
        total_values += 1
        if accuracy > 80:
            predictable_values += 1
            if total_values <= 10 or accuracy > 95:  # Show first 10 examples or very high accuracy ones
                print(f"  Player spy={spy_val:.1f} → dealer's most common card={most_common_card} (accuracy: {accuracy:.1f}%, samples: {len(cards)})")

if total_values > 0:
    print(f"Player spy values with high dealer card predictability: {predictable_values}/{total_values} ({predictable_values/total_values*100:.1f}%)")

# Linear regression to predict dealer card from player spy
X = player_spy.values.reshape(-1, 1)
y = dealer_cards.values

# Try polynomial features
for degree in [1, 2, 3]:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    score = model.score(X_poly, y)
    print(f"Polynomial regression (degree {degree}) score for player spy → dealer card: {score:.4f}")

# Check if there's a direct mapping between player and dealer cards
dealer_card_given_player_card = defaultdict(list)
for p_card, d_card in zip(player_cards, dealer_cards):
    dealer_card_given_player_card[p_card].append(d_card)

print("\nDealer card prediction from player card:")
for p_card in sorted(dealer_card_given_player_card.keys()):
    d_cards = dealer_card_given_player_card[p_card]
    card_counts = np.bincount(d_cards)
    most_common_card = np.argmax(card_counts)
    accuracy = card_counts[most_common_card] / len(d_cards) * 100
    print(f"  Player card={p_card} → dealer's most common card={most_common_card} (accuracy: {accuracy:.1f}%, samples: {len(d_cards)})")

print("\n=== SUMMARY OF KEY FINDINGS ===")
print("1. Table 0: Player spy values strongly correlate with card values")
print("2. Table 1: Dealer spy values show consistent patterns with high predictability")
print("3. Table 2: Complex relationship between spy and card values")
print("4. Table 3: High variance in player spy values")
print("5. Table 4: Limited dealer card values (only 5-9)")
print("6. Sicilian dataset: Possible mathematical relationship between player and dealer values") 