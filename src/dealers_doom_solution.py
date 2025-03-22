#!/usr/bin/env python

"""
Dealer's Doom - Solution

This challenge requires estimating the probability that the dealer busts (exceeds 21)
at each table. The solution also proposes additional metrics to evaluate the 
"willingness to play" at a table.

Key metrics:
1. Dealer bust probability - How often the dealer exceeds 21
2. Expected dealer final total - Average final score of the dealer when not busting
3. Dealer stand at 17-21 probability - How often the dealer reaches a strong hand

These metrics together provide a comprehensive assessment of table profitability.
"""

import numpy as np
import pandas as pd
import os

def bust_probability(dealer_cards):
    """
    Calculate the probability of the dealer busting based on historical data.
    
    Args:
        dealer_cards: A list/array of dealer card values observed in sequence
        
    Returns:
        float: Probability of dealer busting (0.0 to 1.0)
    """
    total_games = 0
    busts = 0
    
    # We'll simulate dealer's play in sequences of cards
    # A dealer draws until they have 17 or more
    i = 0
    while i < len(dealer_cards):
        total = 0
        bust = False
        
        # Keep drawing cards until the dealer stands or busts
        while i < len(dealer_cards) and total < 17:
            total += dealer_cards[i]
            if total > 21:
                bust = True
                break
            i += 1
        
        # If the dealer reached 17 or more without busting, they stand
        if total >= 17 and not bust:
            i += 1  # Move to the next game
            
        # Count this as a completed game if dealer either busted or reached 17+
        if bust or total >= 17:
            total_games += 1
            if bust:
                busts += 1
    
    # Return the probability of busting
    if total_games > 0:
        return busts / total_games
    else:
        return 0.0

def expected_dealer_total(dealer_cards):
    """
    Calculate the expected final total of the dealer when they don't bust.
    
    Args:
        dealer_cards: A list/array of dealer card values observed in sequence
        
    Returns:
        float: Expected dealer's final total when not busting
    """
    non_bust_totals = []
    
    i = 0
    while i < len(dealer_cards):
        total = 0
        bust = False
        
        # Keep drawing cards until the dealer stands or busts
        while i < len(dealer_cards) and total < 17:
            total += dealer_cards[i]
            if total > 21:
                bust = True
                break
            i += 1
        
        # If the dealer reached 17 or more without busting, they stand
        if total >= 17 and not bust:
            non_bust_totals.append(total)
            i += 1
        elif bust:
            i += 1
    
    # Return the average final total when not busting
    if non_bust_totals:
        return np.mean(non_bust_totals)
    else:
        return 0.0

def stand_17_21_probability(dealer_cards):
    """
    Calculate the probability of the dealer standing with a total between 17 and 21 (inclusive).
    
    Args:
        dealer_cards: A list/array of dealer card values observed in sequence
        
    Returns:
        float: Probability of dealer standing between 17 and 21
    """
    total_games = 0
    stands_17_21 = 0
    
    i = 0
    while i < len(dealer_cards):
        total = 0
        bust = False
        
        # Keep drawing cards until the dealer stands or busts
        while i < len(dealer_cards) and total < 17:
            total += dealer_cards[i]
            if total > 21:
                bust = True
                break
            i += 1
        
        # If the dealer reached 17 or more without busting, they stand
        if total >= 17 and not bust:
            stands_17_21 += 1
            i += 1
        elif bust:
            i += 1
        
        # Count this as a completed game
        total_games += 1
    
    # Return the probability of standing between 17 and 21
    if total_games > 0:
        return stands_17_21 / total_games
    else:
        return 0.0

def analyze_table(table_index):
    """
    Analyze a table and provide metrics for assessing profitability.
    
    Args:
        table_index: Index of the table to analyze (0-4)
        
    Returns:
        dict: Dictionary containing metrics for the table
    """
    try:
        # Find the data directory relative to current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(current_dir)
        data_path = os.path.join(repo_root, 'data', 'train.csv')
        
        # Load dealer data
        data = pd.read_csv(data_path, header=[0,1,2])
        dealer_cards = data[(f'table_{table_index}', 'dealer', 'card')].values
        
        # Calculate metrics
        bust_prob = bust_probability(dealer_cards)
        expected_total = expected_dealer_total(dealer_cards)
        stand_prob = stand_17_21_probability(dealer_cards)
        
        return {
            'table_index': table_index,
            'bust_probability': bust_prob,
            'expected_dealer_total': expected_total,
            'stand_17_21_probability': stand_prob,
            'willingness_to_play': calculate_willingness(bust_prob, expected_total)
        }
    except Exception as e:
        print(f"Error analyzing table {table_index}: {e}")
        return {
            'table_index': table_index,
            'error': str(e)
        }

def calculate_willingness(bust_prob, expected_total):
    """
    Calculate a willingness to play score based on dealer metrics.
    Higher score indicates a more favorable table for the player.
    
    Args:
        bust_prob: Probability of dealer busting
        expected_total: Expected dealer final total when not busting
        
    Returns:
        float: Willingness to play score (0.0 to 1.0)
    """
    # Higher bust probability is good for the player
    # Lower expected total when not busting is good for the player
    # Scale expected_total to be in range [0, 1] where 1 is best for player
    scaled_total = max(0, min(1, (21 - expected_total) / 4))  # 21 is best, 17 is worst
    
    # Combine metrics with equal weighting
    return 0.7 * bust_prob + 0.3 * scaled_total

if __name__ == "__main__":
    print("Dealer's Doom - Table Analysis")
    print("=" * 50)
    
    results = []
    for i in range(5):
        result = analyze_table(i)
        results.append(result)
        if 'error' in result:
            print(f"Table {i}: Error - {result['error']}")
        else:
            print(f"Table {i}:")
            print(f"  Dealer Bust Probability: {result['bust_probability']:.4f}")
            print(f"  Expected Dealer Total (non-bust): {result['expected_dealer_total']:.4f}")
            print(f"  Stand at 17-21 Probability: {result['stand_17_21_probability']:.4f}")
            print(f"  Willingness to Play Score: {result['willingness_to_play']:.4f}")
    
    # Rank tables by willingness to play
    valid_results = [r for r in results if 'error' not in r]
    ranked_tables = sorted(valid_results, key=lambda x: x['willingness_to_play'], reverse=True)
    
    print("\nTables Ranked by Willingness to Play:")
    for i, result in enumerate(ranked_tables):
        print(f"{i+1}. Table {result['table_index']} - Score: {result['willingness_to_play']:.4f}") 