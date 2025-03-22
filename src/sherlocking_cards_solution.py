#!/usr/bin/env python

"""
Sherlocking the Cards - Solution

This challenge involves creating a deterministic function that maps 
spy values to card values (2-11) for all tables.

The implementation analyzes the relationships between spy and card values
across all tables and creates mappings that can be used to predict
card values from spy values.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict

class CardSherlockSolution:
    """
    A solution for mapping spy values to card values
    """
    
    def __init__(self):
        """
        Initialize the solution and load data
        """
        self.table_mappings = {}  # Store mappings for each table
        self.global_mapping = {}  # Store a global mapping
        self.models = {}  # Store prediction models for complex relationships
        self.card_ranges = {}  # Store ranges of spy values for each card value
        
        # Try to load and analyze data
        try:
            self._load_and_analyze_data()
        except Exception as e:
            print(f"Error initializing Sherlocking Cards solution: {e}")
    
    def _load_and_analyze_data(self):
        """
        Load data from CSV and analyze spy-card relationships for all tables
        """
        try:
            # Load data
            data = pd.read_csv('data/train.csv', header=[0, 1, 2])
            
            # Process each table
            for table_idx in range(5):
                # Extract player data
                player_data = np.array([
                    data[(f'table_{table_idx}', 'player', 'spy')],
                    data[(f'table_{table_idx}', 'player', 'card')]
                ]).T
                
                # Extract dealer data
                dealer_data = np.array([
                    data[(f'table_{table_idx}', 'dealer', 'spy')],
                    data[(f'table_{table_idx}', 'dealer', 'card')]
                ]).T
                
                # Combine player and dealer data for this table
                combined_data = np.vstack([player_data, dealer_data])
                
                # Create mapping for this table
                self._create_mapping(table_idx, combined_data)
                
                # Train a model for this table for complex relationships
                self._train_mapping_model(table_idx, combined_data)
                
                # Analyze ranges of spy values for each card
                self._analyze_card_ranges(table_idx, combined_data)
            
            # Create a global mapping using all data
            all_data = []
            for table_idx in range(5):
                for role in ['player', 'dealer']:
                    table_data = np.array([
                        data[(f'table_{table_idx}', role, 'spy')],
                        data[(f'table_{table_idx}', role, 'card')]
                    ]).T
                    all_data.append(table_data)
            
            all_data_combined = np.vstack(all_data)
            self._create_mapping('global', all_data_combined)
            self._train_mapping_model('global', all_data_combined)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _create_mapping(self, table_idx, data):
        """
        Create a mapping from rounded spy values to most common card values
        
        Args:
            table_idx: Index of the table or 'global'
            data: Array of [spy_value, card_value] pairs
        """
        mapping = defaultdict(lambda: defaultdict(int))
        
        # Round spy values to nearest 0.1 for binning
        for spy_val, card_val in data:
            # Round to nearest 0.1
            rounded_spy = round(spy_val * 10) / 10
            mapping[rounded_spy][card_val] += 1
        
        # Convert to dictionary of most common card values
        result_mapping = {}
        for spy_val, card_counts in mapping.items():
            most_common_card = max(card_counts.items(), key=lambda x: x[1])[0]
            result_mapping[spy_val] = most_common_card
        
        self.table_mappings[table_idx] = result_mapping
    
    def _train_mapping_model(self, table_idx, data):
        """
        Train a model to map spy values to card values for complex relationships
        
        Args:
            table_idx: Index of the table or 'global'
            data: Array of [spy_value, card_value] pairs
        """
        X = data[:, 0].reshape(-1, 1)  # Spy values
        y = data[:, 1]  # Card values
        
        # Train a simple decision tree model
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X, y)
        
        self.models[table_idx] = model
    
    def _analyze_card_ranges(self, table_idx, data):
        """
        Analyze the range of spy values for each card value
        
        Args:
            table_idx: Index of the table
            data: Array of [spy_value, card_value] pairs
        """
        card_ranges = {}
        
        # Group by card value
        for card_val in sorted(set(data[:, 1])):
            card_mask = (data[:, 1] == card_val)
            spy_vals = data[card_mask, 0]
            
            if len(spy_vals) > 0:
                card_ranges[card_val] = {
                    'min': np.min(spy_vals),
                    'max': np.max(spy_vals),
                    'mean': np.mean(spy_vals),
                    'std': np.std(spy_vals)
                }
        
        self.card_ranges[table_idx] = card_ranges
    
    def get_card_value_from_spy_value(self, spy_value, table_idx=None):
        """
        Convert a spy value to a card value using the table-specific or global mapping
        
        Args:
            spy_value: Spy value to convert
            table_idx: Optional table index (0-4). If None, uses the global mapping.
            
        Returns:
            int: Predicted card value (2-11)
        """
        # Use table-specific mapping if available, otherwise use global
        which_table = table_idx if table_idx is not None and table_idx in self.table_mappings else 'global'
        
        # Round to nearest 0.1 for lookup
        rounded_spy = round(spy_value * 10) / 10
        
        # Try direct lookup first
        if rounded_spy in self.table_mappings[which_table]:
            return self.table_mappings[which_table][rounded_spy]
        
        # If no direct match, use the model if available
        if which_table in self.models:
            try:
                # Reshape for prediction
                spy_reshaped = np.array([[spy_value]])
                return self.models[which_table].predict(spy_reshaped)[0]
            except Exception:
                # If model prediction fails, fallback to closest match
                pass
        
        # Fallback to finding closest spy value in the mapping
        if self.table_mappings[which_table]:
            closest_spy = min(self.table_mappings[which_table].keys(), 
                             key=lambda x: abs(x - rounded_spy))
            return self.table_mappings[which_table][closest_spy]
        
        # Final fallback - simple heuristic based on common patterns
        if spy_value <= 2:
            return 2
        elif spy_value <= 4:
            return 3
        elif spy_value <= 6:
            return 5
        elif spy_value <= 8:
            return 7
        elif spy_value <= 10:
            return 9
        else:
            return 10

# Create a global instance
_sherlock_solution = CardSherlockSolution()

def get_card_value_from_spy_value(spy_value, table_idx=None):
    """
    Global function to convert a spy value to a card value
    
    Args:
        spy_value: Spy value to convert
        table_idx: Optional table index (0-4)
        
    Returns:
        int: Predicted card value (2-11)
    """
    return _sherlock_solution.get_card_value_from_spy_value(spy_value, table_idx)

if __name__ == "__main__":
    print("Sherlocking the Cards - Solution Test")
    print("=" * 50)
    
    # Test with some sample spy values
    test_spy_values = [1.2, 3.7, 5.5, 7.8, 9.9, 11.1]
    
    print("Global mapping predictions:")
    for spy_val in test_spy_values:
        card_val = get_card_value_from_spy_value(spy_val)
        print(f"  Spy value {spy_val:.1f} -> Card value {card_val}")
    
    print("\nTable-specific predictions:")
    for table_idx in range(5):
        print(f"  Table {table_idx}:")
        for spy_val in test_spy_values[:3]:  # Just test a few values
            card_val = get_card_value_from_spy_value(spy_val, table_idx)
            print(f"    Spy value {spy_val:.1f} -> Card value {card_val}") 