#!/usr/bin/env python

"""
Quick test script to verify the spy series solution works properly.
This tests with a small sample of data for a single table.
"""

import sys
import os
import numpy as np

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.spy_series_solution import MyPlayer

def main():
    """Run a quick test of the spy series prediction."""
    print("Quick test of Spy Series Seer")
    print("=" * 30)
    
    # Create a test instance for Table 0 (the simplest one)
    try:
        player = MyPlayer(0)
        print("Successfully initialized MyPlayer")
        
        # Test with some sample data
        sample_hist = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        
        # Test player prediction
        player_pred = player.get_player_spy_prediction(sample_hist)
        print(f"Player prediction with history {sample_hist}: {player_pred:.4f}")
        
        # Test dealer prediction
        dealer_pred = player.get_dealer_spy_prediction(sample_hist)
        print(f"Dealer prediction with history {sample_hist}: {dealer_pred:.4f}")
        
        # Test card value prediction
        card_pred = player.get_card_value_from_spy_value(8.5, is_player=True)
        print(f"Card prediction for player spy 8.5: {card_pred}")
        
        print("\nAll tests passed successfully!")
        return 0
    except Exception as e:
        print(f"Error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 