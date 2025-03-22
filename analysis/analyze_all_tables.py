#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def load_data(table_index):
    """Load data for a specific table."""
    filename = f"data_{table_index}.csv"
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

def analyze_autocorrelation(series, max_lag=10, title="Autocorrelation"):
    """Analyze and plot autocorrelation for a time series."""
    autocorr = [series.autocorr(lag=i) for i in range(1, max_lag + 1)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, max_lag + 1), autocorr)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(title)
    plt.show()
    
    print(f"Autocorrelation at lag 1: {autocorr[0]:.4f}")
    print(f"Autocorrelation at lag 2: {autocorr[1]:.4f}")
    
    # Check for alternating pattern
    alternating_ratio = sum(1 for i in range(len(autocorr)-1) if autocorr[i] * autocorr[i+1] < 0) / (len(autocorr)-1)
    print(f"Alternating pattern ratio: {alternating_ratio:.4f}")
    
    return autocorr

def analyze_card_spy_relationship(data, card_col, spy_col, title="Card-Spy Relationship"):
    """Analyze and plot the relationship between cards and spy values."""
    # Group by card and calculate statistics for spy values
    card_spy_stats = data.groupby(card_col)[spy_col].agg(['mean', 'std', 'count'])
    card_spy_stats = card_spy_stats.reset_index()
    
    print(f"\n{title}:")
    print(card_spy_stats)
    
    # Plot mean spy value by card
    plt.figure(figsize=(12, 6))
    
    # Bar plot for means
    plt.subplot(1, 2, 1)
    plt.bar(card_spy_stats[card_col], card_spy_stats['mean'])
    plt.xlabel('Card Value')
    plt.ylabel('Mean Spy Value')
    plt.title(f'Mean {spy_col} by {card_col}')
    
    # Bar plot for standard deviations
    plt.subplot(1, 2, 2)
    plt.bar(card_spy_stats[card_col], card_spy_stats['std'])
    plt.xlabel('Card Value')
    plt.ylabel('Std Dev of Spy Value')
    plt.title(f'Standard Deviation of {spy_col} by {card_col}')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate correlation between card and spy
    correlation = data[card_col].corr(data[spy_col])
    print(f"Correlation between {card_col} and {spy_col}: {correlation:.4f}")
    
    return card_spy_stats

def analyze_spy_transitions(data, spy_col, title="Spy Value Transitions"):
    """Analyze how spy values transition from one point to the next."""
    # Calculate the differences between consecutive spy values
    diff = data[spy_col].diff().dropna()
    
    # Plot histogram of differences
    plt.figure(figsize=(10, 6))
    plt.hist(diff, bins=30, alpha=0.7)
    plt.xlabel('Difference in Consecutive Spy Values')
    plt.ylabel('Frequency')
    plt.title(f'{title} - Distribution of Differences')
    plt.show()
    
    # Statistics on differences
    print(f"\n{title} - Difference Statistics:")
    print(f"Mean difference: {diff.mean():.4f}")
    print(f"Std deviation: {diff.std():.4f}")
    print(f"Min difference: {diff.min():.4f}")
    print(f"Max difference: {diff.max():.4f}")
    
    # Check for alternating pattern
    alternating_count = sum(1 for i in range(len(diff)-1) if diff.iloc[i] * diff.iloc[i+1] < 0)
    alternating_ratio = alternating_count / (len(diff)-1)
    print(f"Alternating pattern ratio in differences: {alternating_ratio:.4f}")
    
    return diff

def test_prediction_models(data, spy_col, window_sizes=[3, 5, 10]):
    """Test various prediction models on the spy series."""
    # Prepare data: use past n values to predict the next
    results = []
    
    # Create lagged features
    max_window = max(window_sizes)
    for i in range(1, max_window + 1):
        data[f'{spy_col}_lag{i}'] = data[spy_col].shift(i)
    
    # Drop rows with NaN values
    lagged_data = data.dropna()
    
    # Test simple models: average of last n values
    for window in window_sizes:
        x = lagged_data[[f'{spy_col}_lag{i}' for i in range(1, window + 1)]]
        y = lagged_data[spy_col]
        
        # Mean of last n
        y_pred_mean = x.mean(axis=1)
        mse_mean = mean_squared_error(y, y_pred_mean)
        results.append({
            'model': f'Mean of Last {window}',
            'window': window,
            'mse': mse_mean
        })
        
        # Median of last n
        y_pred_median = x.median(axis=1)
        mse_median = mean_squared_error(y, y_pred_median)
        results.append({
            'model': f'Median of Last {window}',
            'window': window,
            'mse': mse_median
        })
        
        # Exponential Weighted Average (weight=0.7)
        weights = 0.7 ** np.arange(window)
        weights = weights / weights.sum()  # Normalize
        y_pred_ewa = (x.values * weights.reshape(1, -1)).sum(axis=1)
        mse_ewa = mean_squared_error(y, y_pred_ewa)
        results.append({
            'model': f'EWA(0.7) of Last {window}',
            'window': window,
            'mse': mse_ewa
        })
    
    # Test regression models
    x = lagged_data[[f'{spy_col}_lag{i}' for i in range(1, 6)]]  # Use last 5 values
    y = lagged_data[spy_col]
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(x, y)
    y_pred_lr = lr.predict(x)
    mse_lr = mean_squared_error(y, y_pred_lr)
    results.append({
        'model': 'Linear Regression',
        'window': 5,
        'mse': mse_lr
    })
    
    # Polynomial Regression (degree=2) with Ridge
    poly2_ridge = make_pipeline(
        PolynomialFeatures(degree=2),
        Ridge(alpha=0.01)
    )
    poly2_ridge.fit(x, y)
    y_pred_poly2 = poly2_ridge.predict(x)
    mse_poly2 = mean_squared_error(y, y_pred_poly2)
    results.append({
        'model': 'Poly(2) + Ridge',
        'window': 5,
        'mse': mse_poly2
    })
    
    # Polynomial Regression (degree=3) with Ridge
    poly3_ridge = make_pipeline(
        PolynomialFeatures(degree=3),
        Ridge(alpha=0.01)
    )
    poly3_ridge.fit(x, y)
    y_pred_poly3 = poly3_ridge.predict(x)
    mse_poly3 = mean_squared_error(y, y_pred_poly3)
    results.append({
        'model': 'Poly(3) + Ridge',
        'window': 5,
        'mse': mse_poly3
    })
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x, y)
    y_pred_rf = rf.predict(x)
    mse_rf = mean_squared_error(y, y_pred_rf)
    results.append({
        'model': 'RandomForest',
        'window': 5,
        'mse': mse_rf
    })
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(x, y)
    y_pred_gb = gb.predict(x)
    mse_gb = mean_squared_error(y, y_pred_gb)
    results.append({
        'model': 'GradientBoosting',
        'window': 5,
        'mse': mse_gb
    })
    
    # Sort results by MSE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mse')
    
    print("\nPrediction Model Performance:")
    print(results_df)
    
    # Plot predictions for best model
    best_model = results_df.iloc[0]['model']
    plt.figure(figsize=(12, 6))
    plt.plot(y.values, label='Actual')
    
    if 'Mean of Last' in best_model:
        window = int(best_model.split(' ')[-1])
        y_pred = lagged_data[[f'{spy_col}_lag{i}' for i in range(1, window + 1)]].mean(axis=1)
    elif 'Median of Last' in best_model:
        window = int(best_model.split(' ')[-1])
        y_pred = lagged_data[[f'{spy_col}_lag{i}' for i in range(1, window + 1)]].median(axis=1)
    elif 'EWA' in best_model:
        window = int(best_model.split(' ')[-1])
        weights = 0.7 ** np.arange(window)
        weights = weights / weights.sum()
        y_pred = (lagged_data[[f'{spy_col}_lag{i}' for i in range(1, window + 1)]].values * weights.reshape(1, -1)).sum(axis=1)
    elif best_model == 'Linear Regression':
        y_pred = y_pred_lr
    elif best_model == 'Poly(2) + Ridge':
        y_pred = y_pred_poly2
    elif best_model == 'Poly(3) + Ridge':
        y_pred = y_pred_poly3
    elif best_model == 'RandomForest':
        y_pred = y_pred_rf
    elif best_model == 'GradientBoosting':
        y_pred = y_pred_gb
    
    plt.plot(y_pred, label=f'Predicted ({best_model})')
    plt.legend()
    plt.title(f'Best Model: {best_model}, MSE: {results_df.iloc[0]["mse"]:.4f}')
    plt.show()
    
    return results_df

def analyze_table(table_index):
    """Analyze a specific table."""
    print(f"\n{'='*80}")
    print(f"ANALYZING TABLE {table_index}")
    print(f"{'='*80}\n")
    
    data = load_data(table_index)
    if data is None:
        return
    
    # Basic statistics
    print("Data Shape:", data.shape)
    print("\nBasic Statistics - Player:")
    print(data[['player_card', 'player_spy']].describe())
    print("\nBasic Statistics - Dealer:")
    print(data[['dealer_card', 'dealer_spy']].describe())
    
    # Card distribution
    print("\nCard Distribution:")
    print("Player Cards:", data['player_card'].value_counts().sort_index())
    print("Dealer Cards:", data['dealer_card'].value_counts().sort_index())
    
    # Analyze autocorrelation for player and dealer spy values
    print("\n--- Player Spy Autocorrelation ---")
    player_autocorr = analyze_autocorrelation(data['player_spy'], title=f"Table {table_index}: Player Spy Autocorrelation")
    
    print("\n--- Dealer Spy Autocorrelation ---")
    dealer_autocorr = analyze_autocorrelation(data['dealer_spy'], title=f"Table {table_index}: Dealer Spy Autocorrelation")
    
    # Analyze relationship between card values and spy values
    player_card_spy = analyze_card_spy_relationship(
        data, 'player_card', 'player_spy', 
        title=f"Table {table_index}: Player Card-Spy Relationship"
    )
    
    dealer_card_spy = analyze_card_spy_relationship(
        data, 'dealer_card', 'dealer_spy', 
        title=f"Table {table_index}: Dealer Card-Spy Relationship"
    )
    
    # Analyze transitions in spy values
    player_diff = analyze_spy_transitions(
        data, 'player_spy', 
        title=f"Table {table_index}: Player Spy Transitions"
    )
    
    dealer_diff = analyze_spy_transitions(
        data, 'dealer_spy', 
        title=f"Table {table_index}: Dealer Spy Transitions"
    )
    
    # Test prediction models for player spy values
    print("\n--- Player Spy Prediction Models ---")
    player_model_results = test_prediction_models(
        data.copy(), 'player_spy', 
        window_sizes=[3, 5, 10]
    )
    
    # Test prediction models for dealer spy values
    print("\n--- Dealer Spy Prediction Models ---")
    dealer_model_results = test_prediction_models(
        data.copy(), 'dealer_spy', 
        window_sizes=[3, 5, 10]
    )
    
    return {
        'player_autocorr': player_autocorr,
        'dealer_autocorr': dealer_autocorr,
        'player_card_spy': player_card_spy,
        'dealer_card_spy': dealer_card_spy,
        'player_diff': player_diff,
        'dealer_diff': dealer_diff,
        'player_model_results': player_model_results,
        'dealer_model_results': dealer_model_results
    }

def main():
    """Main function to analyze all tables."""
    results = {}
    
    for table_index in range(5):  # Tables 0-4
        results[table_index] = analyze_table(table_index)
    
    # Compare results across tables
    print("\n\n" + "="*80)
    print("CROSS-TABLE COMPARISON")
    print("="*80 + "\n")
    
    # Compare best models across tables
    best_models = {
        'player': {},
        'dealer': {}
    }
    
    for table_index in range(5):
        if table_index in results and results[table_index] is not None:
            player_best = results[table_index]['player_model_results'].iloc[0]
            dealer_best = results[table_index]['dealer_model_results'].iloc[0]
            
            best_models['player'][table_index] = {
                'model': player_best['model'],
                'mse': player_best['mse']
            }
            
            best_models['dealer'][table_index] = {
                'model': dealer_best['model'],
                'mse': dealer_best['mse']
            }
    
    # Create comparison table
    comparison_table = []
    for table_index in range(5):
        if table_index in best_models['player'] and table_index in best_models['dealer']:
            comparison_table.append({
                'Table': table_index,
                'Player MSE': best_models['player'][table_index]['mse'],
                'Player Best Model': best_models['player'][table_index]['model'],
                'Dealer MSE': best_models['dealer'][table_index]['mse'],
                'Dealer Best Model': best_models['dealer'][table_index]['model']
            })
    
    comparison_df = pd.DataFrame(comparison_table)
    print("\nBest Model Comparison Across Tables:")
    print(comparison_df)
    
    # Plot MSE comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(comparison_df['Table'], comparison_df['Player MSE'])
    plt.xlabel('Table')
    plt.ylabel('Mean Squared Error')
    plt.title('Best Player Model MSE by Table')
    
    plt.subplot(1, 2, 2)
    plt.bar(comparison_df['Table'], comparison_df['Dealer MSE'])
    plt.xlabel('Table')
    plt.ylabel('Mean Squared Error')
    plt.title('Best Dealer Model MSE by Table')
    
    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main() 