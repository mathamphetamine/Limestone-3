# Limestone Data Challenge Solutions

This repository contains solutions to the Limestone data challenge, a series of machine learning problems focused on prediction and strategy in card-based games.

## Overview

The challenge consists of several tasks, with the main focus on:

1. **Spy Series Seer**: Predicting the next value in a time series of "spy values"
2. **Marathon of Twenty-One**: Developing a strategy for a Blackjack-like game

The solutions use a variety of machine learning and statistical techniques including:
- Time series analysis with autocorrelation detection
- Pattern recognition for alternating sequences
- Polynomial regression with regularization
- Ensemble methods (Random Forest, Gradient Boosting)
- Table-specific optimizations based on statistical properties

## Solution Structure

The repository is organized as follows:

```
Limestone-3/
├── ANALYSIS.md                 # Detailed data analysis documentation
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── src/                        # Source code directory
│   ├── spy_series_solution.py  # Implementation of the spy value prediction
│   ├── test_spy_series.py      # Test script for evaluating the solution
│   ├── quick_test.py           # Simple test to verify functionality
│   └── setup_data_directory.py # Script to set up data directory
├── data/                       # Data directory (not included in repo)
│   └── train.csv               # Training data (must be added manually)
└── analysis/                   # Analysis scripts
    ├── analyze_all_tables.py   # Comprehensive analysis script
    ├── analyze_table1.py       # Analysis for table 1
    ├── analyze_table2.py       # Analysis for table 2
    ├── analyze_table3.py       # Analysis for table 3
    └── analyze_table4.py       # Analysis for table 4
```

## Spy Series Seer

This solution predicts spy values for both player and dealer across 5 different tables.

### Key Features

- **Table-specific models**: Each table has unique patterns requiring different approaches
- **Pattern detection**: Algorithms to detect alternating sequences and autocorrelation
- **Range-based prediction**: Special handling for different value ranges in Table 2
- **Card mapping**: Analysis of relationships between spy values and cards
- **Advanced regression**: Polynomial features with Ridge regularization for tables with complex patterns

### Performance

| Table | Player MSE | Dealer MSE | Best Player Model     | Best Dealer Model    |
|-------|------------|------------|----------------------|----------------------|
| 0     | 0.33       | 0.18       | Linear Regression    | Alternating Pattern  |
| 1     | 11.18      | 0.22       | Poly(2) + Ridge      | GradientBoosting     |
| 2     | 1924.63    | 49.56      | Range-based          | Poly(2) + Ridge      |
| 3     | 48.26      | 5.56       | Mean of Last 3       | Poly(2) + Linear     |
| 4     | 14.26      | 0.83       | Poly(3) + Ridge      | Alternating Pattern  |

**Overall Average MSE**: 205.50

## Marathon of Twenty-One

This solution implements a strategy for a Blackjack-like game using card prediction.

### Key Features

- **Dynamic surrender strategy**: Decision to surrender is based on win probability
- **Dealer prediction**: Estimate of dealer's final score based on upcard and spy values
- **Card counting**: Frequency analysis and tracking of seen cards
- **Policy optimization**: Different strategies based on table characteristics

### Performance

| Table | Final Score |
|-------|-------------|
| 0     | -30.5       |
| 1     | -72.0       |
| 2     | -68.0       |
| 3     | -89.5       |
| 4     | -141.5      |

## Getting Started

### Setting Up

1. Clone the repository:
   ```bash
   git clone https://github.com/mathamphetamine/Limestone-3.git
   cd Limestone-3
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the data directory:
   ```bash
   python src/setup_data_directory.py
   ```

4. Add your training data to the `data/` directory
   - The file should be named `train.csv`
   - CSV file should have multi-level headers in the format `(table_x, player/dealer, spy/card)`

### Running Tests

Quick verification:
```bash
python src/quick_test.py
```

Full test suite:
```bash
python src/test_spy_series.py
```

### Running Analysis

Comprehensive analysis:
```bash
python analysis/analyze_all_tables.py
```

Table-specific analysis:
```bash
python analysis/analyze_table1.py  # Replace with table number 1-4
```

## Implementation Details

### Table-Specific Approaches

- **Table 0**: Very consistent patterns with strong correlation between cards and spy values
- **Table 1**: Moderate autocorrelation leveraged with polynomial regression
- **Table 2**: Range-based approach for handling different behavioral segments
- **Table 3**: Extremely high autocorrelation (0.9997) making simple averaging effective
- **Table 4**: Strong card-spy relationship for specific cards (6-10)

For detailed analysis of each table, refer to [ANALYSIS.md](ANALYSIS.md).

## License

This project is available for educational and research purposes. Code can be used with attribution.

## Acknowledgements

- The Limestone Data Challenge creators for the interesting problem set
- Contributors to the scikit-learn library for machine learning tools 