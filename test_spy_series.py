#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
from spy_series_solution import MyPlayer

def data_loader(path, table_idx, player_or_dealer):
    """
    Load data from CSV file for evaluation.
    
    Args:
        path (str): Path to the CSV file containing the data
        table_idx (int): Index of the table to load data for
        player_or_dealer (str): Either 'player' or 'dealer'
        
    Returns:
        numpy.ndarray: Array with spy and card values
    
    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    try:
        data = pd.read_csv(path, header=[0,1,2])
        spy = data[(f'table_{table_idx}', player_or_dealer, 'spy')]
        card = data[(f'table_{table_idx}', player_or_dealer, 'card')]
        return np.array([spy, card]).T
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        raise
    except Exception as e:
        print(f"Error loading data for table {table_idx}, {player_or_dealer}: {e}")
        raise

def split_train_test(data, test_ratio=0.2):
    """
    Split data into training and testing sets.
    
    Args:
        data (numpy.ndarray): Data to split
        test_ratio (float): Ratio of data to use for testing
        
    Returns:
        tuple: (train_data, test_data)
    """
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]

def score_mse(test_data, mode, player):
    """
    Calculate Mean Squared Error (MSE) for predictions.
    
    Args:
        test_data (numpy.ndarray): Test data containing spy values
        mode (str): Either 'player' or 'dealer'
        player (MyPlayer): Instance of MyPlayer class to use for predictions
        
    Returns:
        float: Mean Squared Error of predictions
        
    Raises:
        ValueError: If mode is not 'player' or 'dealer'
    """
    test_data = test_data[:, 0]  # Only spy values
    
    preds = []
    trues = []
    
    # We need at least 5 historical values to make a prediction
    if len(test_data) <= 5:
        print(f"Warning: Not enough data points for {mode} (minimum 6 required)")
        return float('inf')
        
    for idx in range(5, len(test_data)):
        inp = test_data[idx-5:idx]
        if mode == 'player':
            op = player.get_player_spy_prediction(inp)
        elif mode == 'dealer':
            op = player.get_dealer_spy_prediction(inp)
        else:
            raise ValueError("Mode must be 'player' or 'dealer'")
        preds.append(op)
        trues.append(test_data[idx])
    
    mse = np.mean((np.array(trues) - np.array(preds)) ** 2)
    return mse

if __name__ == '__main__':
    print("Spy Series Seer - Testing Prediction Performance")
    print("=" * 50)
    
    # Create a data directory for storing training/testing splits
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate train-test split if it doesn't exist
    data_path = "data/train.csv"
    test_split_path = os.path.join(test_dir, "test_split.csv")
    train_split_path = os.path.join(test_dir, "train_split.csv")
    
    if not (os.path.exists(test_split_path) and os.path.exists(train_split_path)):
        print("Creating train-test split...")
        all_data = pd.read_csv(data_path, header=[0,1,2])
        
        # Split into 80% train, 20% test
        split_idx = int(len(all_data) * 0.8)
        train_data = all_data.iloc[:split_idx]
        test_data = all_data.iloc[split_idx:]
        
        # Save splits to disk
        train_data.to_csv(train_split_path, index=False)
        test_data.to_csv(test_split_path, index=False)
        print(f"Created train split ({len(train_data)} rows) and test split ({len(test_data)} rows)")
    
    # Calculate MSE for each table and each mode (player/dealer)
    results = {}
    
    for table_idx in range(5):
        print(f"\nEvaluating Table {table_idx}")
        try:
            # Initialize player with training data
            player = MyPlayer(table_idx)
            
            # Load test data
            player_data = data_loader(test_split_path, table_idx, 'player')
            dealer_data = data_loader(test_split_path, table_idx, 'dealer')
            
            # Calculate MSE on test data
            player_mse = score_mse(player_data, 'player', player)
            dealer_mse = score_mse(dealer_data, 'dealer', player)
            
            print(f"  Player MSE: {player_mse:.6f}")
            print(f"  Dealer MSE: {dealer_mse:.6f}")
            
            results[table_idx] = {
                'player_mse': player_mse,
                'dealer_mse': dealer_mse
            }
        except Exception as e:
            print(f"  Error evaluating Table {table_idx}: {e}")
            results[table_idx] = {
                'player_mse': float('inf'),
                'dealer_mse': float('inf')
            }
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 50)
    total_player_mse = 0
    total_dealer_mse = 0
    valid_tables = 0
    
    for table_idx, scores in results.items():
        player_mse = scores['player_mse']
        dealer_mse = scores['dealer_mse']
        
        # Skip tables with infinite MSE in the average calculation
        if np.isinf(player_mse) or np.isinf(dealer_mse):
            print(f"Table {table_idx} - ERROR: Could not calculate MSE")
            continue
            
        print(f"Table {table_idx} - Player MSE: {player_mse:.6f}, Dealer MSE: {dealer_mse:.6f}")
        total_player_mse += player_mse
        total_dealer_mse += dealer_mse
        valid_tables += 1
    
    if valid_tables > 0:
        print("\nPerformance Metrics:")
        print(f"Average Player MSE: {total_player_mse/valid_tables:.6f}")
        print(f"Average Dealer MSE: {total_dealer_mse/valid_tables:.6f}")
        print(f"Overall Average MSE: {(total_player_mse + total_dealer_mse)/(2*valid_tables):.6f}")
    else:
        print("\nERROR: Could not calculate metrics for any table") 