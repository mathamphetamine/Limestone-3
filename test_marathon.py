import pandas as pd
import numpy as np
import sys
import os

# Add the codebase directory to Python path
sys.path.append(os.path.join(os.getcwd(), 'codebase', 'Marathon Of Twenty-One'))

# Import your solution
from marathon_of_twenty_one_solution import MyPlayer

# Import the scoring function
import score

# Function to load data
def data_loader(path, table_idx, player_or_dealer):
    data = pd.read_csv(path, header=[0,1,2])
    spy = data[(f'table_{table_idx}', player_or_dealer, 'spy')]
    card = data[(f'table_{table_idx}', player_or_dealer, 'card')]
    return np.array([spy, card]).T

# Test your solution for multiple tables
for table_index in range(5):
    # Create your player
    player = MyPlayer(table_index)

    # Load the data
    player_data = data_loader("data/train.csv", table_index, 'player')
    dealer_data = data_loader("data/train.csv", table_index, 'dealer')

    # Run the game simulation and get the score
    print(f"\nRunning Marathon of Twenty-One simulation for table {table_index}...")
    score_result = score.score_game(player_data, dealer_data, player, num_games=200, debug=False)
    print(f"Final score for table {table_index}: {score_result}") 