import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data/train.csv', header=[0,1,2])
print("Data shape:", data.shape)

# Analyze each table
for table_idx in range(5):
    print(f"\n=== Table {table_idx} Analysis ===")
    
    # Player data
    player_spy = data[(f'table_{table_idx}', 'player', 'spy')]
    player_card = data[(f'table_{table_idx}', 'player', 'card')]
    
    spy_by_card = defaultdict(list)
    for i in range(len(player_spy)):
        spy_by_card[player_card.iloc[i]].append(player_spy.iloc[i])
    
    print(f"Player spy range: {player_spy.min():.2f} to {player_spy.max():.2f}")
    print("Card to Avg Spy:")
    for card, avg in sorted([(card, np.mean(spies)) for card, spies in spy_by_card.items()]):
        print(f"Card {card}: {avg:.2f}")
    
    # Dealer data
    dealer_spy = data[(f'table_{table_idx}', 'dealer', 'spy')]
    dealer_card = data[(f'table_{table_idx}', 'dealer', 'card')]
    
    dealer_spy_by_card = defaultdict(list)
    for i in range(len(dealer_spy)):
        dealer_spy_by_card[dealer_card.iloc[i]].append(dealer_spy.iloc[i])
    
    print(f"\nDealer spy range: {dealer_spy.min():.2f} to {dealer_spy.max():.2f}")
    print("Card to Avg Spy:")
    for card, avg in sorted([(card, np.mean(spies)) for card, spies in dealer_spy_by_card.items()]):
        print(f"Card {card}: {avg:.2f}")

# Also analyze sicilian_train.csv
print("\n=== Sicilian Train Analysis ===")
sicilian_data = pd.read_csv('data/sicilian_train.csv', header=[0,1,2])
print("Sicilian data shape:", sicilian_data.shape)

special_player_spy = sicilian_data[('special_table', 'player', 'spy')]
special_player_card = sicilian_data[('special_table', 'player', 'card')]
special_dealer_spy = sicilian_data[('special_table', 'dealer', 'spy')]
special_dealer_card = sicilian_data[('special_table', 'dealer', 'card')]

print(f"Player spy range: {special_player_spy.min():.2f} to {special_player_spy.max():.2f}")
print(f"Dealer spy range: {special_dealer_spy.min():.2f} to {special_dealer_spy.max():.2f}")

special_spy_by_card = defaultdict(list)
for i in range(len(special_player_spy)):
    special_spy_by_card[special_player_card.iloc[i]].append(special_player_spy.iloc[i])

print("Card to Avg Spy (Sicilian Player):")
for card, avg in sorted([(card, np.mean(spies)) for card, spies in special_spy_by_card.items()]):
    print(f"Card {card}: {avg:.2f}")

special_dealer_spy_by_card = defaultdict(list)
for i in range(len(special_dealer_spy)):
    special_dealer_spy_by_card[special_dealer_card.iloc[i]].append(special_dealer_spy.iloc[i])

print("\nCard to Avg Spy (Sicilian Dealer):")
for card, avg in sorted([(card, np.mean(spies)) for card, spies in special_dealer_spy_by_card.items()]):
    print(f"Card {card}: {avg:.2f}") 