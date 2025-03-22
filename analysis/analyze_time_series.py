import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def analyze_autocorrelation(data, name):
    """Analyze autocorrelation in the time series"""
    print(f"\n=== Autocorrelation Analysis for {name} ===")
    correlations = []
    
    # Calculate lag-1 to lag-10 autocorrelations
    for lag in range(1, 11):
        correlation = np.corrcoef(data[lag:], data[:-lag])[0, 1]
        correlations.append(correlation)
        print(f"Lag-{lag} autocorrelation: {correlation:.4f}")
    
    return correlations

def test_prediction_model(data, window_size=5):
    """Test a simple linear regression model for prediction"""
    X = []
    y = []
    
    # Create the training data
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    # For comparison, also try using the last value as prediction
    last_value_mse = mean_squared_error(y_test, X_test[:, -1])
    print(f"Last value MSE: {last_value_mse:.4f}")
    
    return model, test_mse

# Load data
data = pd.read_csv('data/train.csv', header=[0,1,2])

# Analyze one table and series type at a time
for table_idx in range(5):
    print(f"\n\n==== Table {table_idx} Analysis ====")
    
    # Player spy data
    player_spy = data[(f'table_{table_idx}', 'player', 'spy')].values
    analyze_autocorrelation(player_spy, f"Table {table_idx} Player Spy")
    print("\nLinear Regression Prediction Model:")
    player_model, player_mse = test_prediction_model(player_spy)
    
    # Dealer spy data
    dealer_spy = data[(f'table_{table_idx}', 'dealer', 'spy')].values
    analyze_autocorrelation(dealer_spy, f"Table {table_idx} Dealer Spy")
    print("\nLinear Regression Prediction Model:")
    dealer_model, dealer_mse = test_prediction_model(dealer_spy)

# Also analyze Sicilian data
print("\n\n==== Sicilian Data Analysis ====")
sicilian_data = pd.read_csv('data/sicilian_train.csv', header=[0,1,2])

special_player_spy = sicilian_data[('special_table', 'player', 'spy')].values
analyze_autocorrelation(special_player_spy, "Sicilian Player Spy")
print("\nLinear Regression Prediction Model:")
special_player_model, special_player_mse = test_prediction_model(special_player_spy)

special_dealer_spy = sicilian_data[('special_table', 'dealer', 'spy')].values
analyze_autocorrelation(special_dealer_spy, "Sicilian Dealer Spy")
print("\nLinear Regression Prediction Model:")
special_dealer_model, special_dealer_mse = test_prediction_model(special_dealer_spy) 