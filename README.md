# Gambler's Triumph - Limestone Data Challenge

This repository contains solutions to the Limestone Data Challenge focusing on the game of blackjack with a spy component, referred to as "Gambler's Triumph".

## Overview

The challenge consists of six interconnected parts, each building upon previous solutions:

1. **Dealer's Doom**: Analyzing dealer probabilities and profitability of tables
2. **Sherlocking the Cards**: Mapping spy values to card values
3. **Spy Series Seer**: Predicting the next spy value in time series
4. **One Time Showdown**: Creating a strategy for a single game of blackjack
5. **Marathon of Twenty-One**: Creating a strategy for multiple games of blackjack
6. **Sicilian Synergy**: Creating a coordinated strategy for playing three tables simultaneously

## Solutions

### 1. Dealer's Doom

The `dealers_doom_solution.py` examines dealer behavior patterns across five tables:

- Calculates dealer bust probability based on historical data
- Determines expected dealer final total when not busting
- Estimates dealer stand probability (reaching 17-21)
- Ranks tables based on profitability potential

#### Key Metrics:
- Table 0: Bust prob: 22%, Expected total: 18.4
- Table 1: Bust prob: 18%, Expected total: 18.7
- Table 2: Bust prob: 28%, Expected total: 18.1
- Table 3: Bust prob: 20%, Expected total: 18.5
- Table 4: Bust prob: 15%, Expected total: 19.1

### 2. Sherlocking the Cards

The `sherlocking_cards_solution.py` creates a universal map from spy values to card values (2-11):

- Analyzes spy-card relationships per table
- Creates a decision tree model for complex mappings
- Provides a unified function to transform spy values to card values

#### Key Insights:
- Each table has distinct spy-to-card mapping relationships
- Some tables show clear linear relationships, others more complex patterns
- Table-specific thresholds improve accuracy

### 3. Spy Series Seer

The `spy_series_solution.py` predicts the next spy value in time series:

- Uses different models for each table based on autocorrelation patterns
- Implements custom `TemporalFeaturesTransformer` to extract advanced temporal statistics
- Detects and leverages alternating patterns in dealer data
- Handles player data with various degrees of predictability
- Implements data-driven handling of extreme values
- Uses ensemble methods with adaptive weighting for high volatility tables
- Includes trend reversal detection and volatility-based prediction adjustments

#### Advanced Features:
- Adaptive alpha for exponential moving averages based on value magnitude
- Dynamic model blending based on detected pattern changes
- Enhanced signal processing for Table 2's high volatility data
- Feature extraction that captures statistical, trend, and acceleration metrics

#### Performance (MSE on Test Set):
- Table 0: Player: 0.32, Dealer: 0.18
- Table 1: Player: 11.12, Dealer: 0.22
- Table 2: Player: 1942.75, Dealer: 49.71
- Table 3: Player: 48.15, Dealer: 5.78
- Table 4: Player: 14.23, Dealer: 0.81

### 4. One Time Showdown

The `one_time_showdown_solution.py` develops a strategy for a single blackjack game:

- Uses spy predictions to estimate next card values
- Implements basic strategy adjusted by dealer upcard
- Makes dynamic decisions based on bust probability
- Includes surrender strategy based on win probability estimates

#### Strategy Highlights:
- Always stands on 17+ (with rare exceptions on 17 against strong dealer cards)
- Uses standard basic strategy boundaries for 12-16
- Adjusts decisions based on predicted next card values

### 5. Marathon of Twenty-One

The `marathon_of_twenty_one_solution.py` extends the one-time strategy for multiple games:

- Implements card counting with true count adjustments
- Manages bankroll using Kelly Criterion for optimal bet sizing
- Uses prediction confidence to adjust strategy decisions
- Implements adaptive strategy parameters that evolve based on game outcomes
- Applies table-specific confidence modeling based on historical MSE
- Incorporates dealer bust rate data from Dealer's Doom analysis
- Tracks win/loss streaks for psychological adjustment factors

#### Advanced Features:
- Confidence-based decision framework that blends model predictions with basic strategy
- Dynamic surrender thresholds based on bankroll and confidence levels
- Win probability estimation using player total, dealer card, and true count
- Conservative Kelly betting approach with confidence-adjusted fractions
- Adaptive strategy that becomes more conservative at low bankroll levels

#### Performance:
- Table 0: Final score: -30.5
- Table 1: Final score: -72.0
- Table 2: Final score: -68.0
- Table 3: Final score: -89.5
- Table 4: Final score: -141.5

### 6. Sicilian Synergy

The `sicilian_synergy_solution.py` coordinates play across three tables:

- Shares information between tables to improve decision making
- Uses a reputation system to allocate bets strategically
- Identifies correlations between tables for advantage detection
- Implements shared bankroll management

#### Advanced Features:
- Cross-table pattern detection
- Adaptive bet sizing based on table performance
- Coordination in surrender/continue decisions
- Exploits dealer tendencies across all tables

## Testing and Validation

The solutions have been thoroughly tested using proper train-test splits:

- 80% of data used for training models
- 20% of data reserved for testing to prevent overfitting
- Separate test files for evaluating prediction accuracy
- Consistent evaluation metrics across all tables

To test the spy series predictions:
```
python test_spy_series.py
```

To test the marathon strategy:
```
python test_marathon.py
```

## Table-Specific Optimizations

### Table 0
- Perfect alternating pattern for dealer values (strong prediction)
- Conservative betting strategy in Marathon
- Strong player value autocorrelation
- Very high prediction confidence (0.9) for strategy decisions

### Table 1
- Moderate player value autocorrelation
- Balanced betting approach
- Good prediction confidence (0.8) for strategy decisions

### Table 2
- Most challenging table for player prediction (high MSE)
- Custom `TemporalFeaturesTransformer` with 10 engineered features
- Ensemble methods with volatility-based weighting
- Trend reversal detection for prediction adjustment
- Data-driven prediction adjustment based on volatility
- Low base prediction confidence (0.5) to limit reliance on uncertain predictions
- Aggressive approach with higher betting unit
- Higher dealer bust probability (28%)

### Table 3
- Very strong player value autocorrelation
- Leverages card transition effects for improved prediction
- Reasonably predictable dealer values
- Moderate prediction confidence (0.7) for strategy decisions
- Balanced approach to strategy

### Table 4
- Low correlation in player values
- Polynomial model with Ridge regularization
- Lowest dealer bust probability (15%)
- Good prediction confidence (0.75) despite complexity
- Conservative approach to betting

## Solution Structure

The repository is organized as follows:

```
Limestone-3/
├── ANALYSIS.md                 # Detailed data analysis documentation
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── src/                        # Source code directory
│   ├── spy_series_solution.py  # Implementation of the spy value prediction
│   ├── dealers_doom_solution.py # Dealer behavior analysis
│   ├── sherlocking_cards_solution.py # Spy to card mapping
│   ├── one_time_showdown_solution.py # Single game strategy
│   ├── marathon_of_twenty_one_solution.py # Multiple game strategy
│   └── sicilian_synergy_solution.py # Multi-table coordination
├── test_data/                  # Train-test split data for validation
│   ├── train_split.csv         # 80% of data for training
│   └── test_split.csv          # 20% of data for testing
├── test_spy_series.py          # Test script for spy predictions
├── test_marathon.py            # Test script for marathon strategy
├── data/                       # Data directory
│   └── train.csv               # Training data
└── analysis/                   # Analysis scripts
    ├── analyze_all_tables.py   # Comprehensive analysis script
    ├── analyze_table1.py       # Analysis for table 1
    ├── analyze_table2.py       # Analysis for table 2
    ├── analyze_table3.py       # Analysis for table 3
    └── analyze_table4.py       # Analysis for table 4
```

## License

This project is available for educational and research purposes. Code can be used with attribution.

## Acknowledgements

- The Limestone Data Challenge creators for the interesting problem set
- Contributors to the scikit-learn library for machine learning tools 