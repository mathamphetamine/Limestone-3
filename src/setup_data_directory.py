#!/usr/bin/env python

"""
Script to set up the data directory structure required for the solution.
This creates the necessary directories and informs the user
about the file structure required for tests to run properly.
"""

import os
import sys

def main():
    """Create and set up the data directory structure."""
    print("Setting up data directory structure for Limestone solution")
    print("=" * 60)
    
    # Get the repository root directory
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(repo_root, "data")
    if not os.path.exists(data_dir):
        print(f"Creating data directory: {data_dir}")
        os.makedirs(data_dir)
    else:
        print(f"Data directory already exists: {data_dir}")
    
    # Explain the required file structure
    print("\nRequired Data Files:")
    print("-------------------")
    print("Please place the following files in the data directory:")
    print("1. train.csv - Training data with spy values and card data for all tables")
    print("2. test.csv (optional) - Test data for evaluating the solution")
    
    print("\nData file format requirements:")
    print("----------------------------")
    print("- CSV files should have multi-level headers in the format: (table_x, player/dealer, spy/card)")
    print("- For example: (table_0, player, spy), (table_0, player, card), etc.")
    
    print("\nSetup complete!")
    print("\nTo run the solution:")
    print("1. Place the required data files in the data directory")
    print("2. Run the tests with: python src/test_spy_series.py")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 