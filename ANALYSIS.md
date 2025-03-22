# Limestone Challenge Data Analysis

## Overview

This document details the analysis performed on the Limestone challenge datasets, with a focus on understanding the patterns and relationships in the spy values and card data across different tables.

## Table 0

### Statistics
- Player Spy: Very low variance, strong predictability
- Dealer Spy: Alternating pattern with high accuracy
- Strong correlation between card values and spy values

### Key Insights
- Dealer spy values follow a clear alternating pattern, making prediction straightforward
- Player spy values can be predicted with extremely high accuracy (MSE = 0.33)
- Linear regression works well for player prediction due to consistent patterns

## Table 1

### Statistics
- Player Spy: Mean = 79.78, Std = 4.03, Range = [70.55, 89.03]
- Moderate autocorrelation at lag 1 (0.4855)
- Clear relationship between card values and spy values
- Card 2: Mean spy = 79.98, Std = 0.29
- Card 3: Mean spy = 80.67, Std = 1.81
- Card transitions show predictable patterns

### Key Insights
- Polynomial features with Ridge regularization (α=0.01) provide the best performance
- Exponential weighted average (0.7 weight) also performs well
- Card transitions have low variance, indicating strong predictability
- Dealer spy values map almost directly to card values

## Table 2

### Statistics
- Player Spy: Mean = 33.96, Std = 43.51, Range = [10.00, 202.07]
- Highly variable data with different segments
- Range-specific behavior: Values in [50, 100) have much lower MSE (326.96)
- Range-specific average changes: [0, 50): +10.74, [50, 100): -17.19, [100, inf): -126.42

### Key Insights
- Range-based prediction dramatically improves performance
- Using median for lower values and range-specific adjustments for higher values
- RandomForest performs better than more complex models like GradientBoosting
- Table 2 remains the most challenging table with highest MSE

## Table 3

### Statistics
- Player Spy: Mean = -155.25, Std = 305.14
- Extremely high autocorrelation at lag 1 (0.9997)
- Strong autocorrelation persists at higher lags
- Card sequences show patterns but have less impact than autocorrelation

### Key Insights
- Mean of last 3 values provides the best prediction (MSE = 48.26)
- Card transition adjustments don't improve performance
- The high autocorrelation makes simple averaging highly effective
- More complex models don't outperform simple averaging

## Table 4

### Statistics
- Player Spy: Mean = 6.02, Std = 3.77, Range = [-1.47, 13.55]
- Low autocorrelation at lag 1 (0.0040)
- High alternating pattern ratio (0.6677)
- Cards 6-10 have extremely low variance in spy values:
  - Card 6: Mean = 4.00, Std = 0.38
  - Card 7: Mean = 5.01, Std = 0.29
  - Card 8: Mean = 6.00, Std = 0.28
  - Card 9: Mean = 7.00, Std = 0.28
  - Card 10: Mean = 8.00, Std = 0.29

### Key Insights
- Mean of last 5 values performs best overall (MSE = 16.96)
- Card-specific predictions are accurate for cards 6-10
- Polynomial features (degree=3) with Ridge regularization work well
- Alternating patterns are present but not as strong as other tables

## Cross-Table Comparison

| Table | Player MSE | Dealer MSE | Best Player Model | Best Dealer Model |
|-------|------------|------------|-------------------|-------------------|
| 0     | 0.33       | 0.18       | Linear Regression | Alternating       |
| 1     | 11.18      | 0.22       | Poly(2) + Ridge   | GradientBoosting  |
| 2     | 1924.63    | 49.56      | Range-based       | Poly(2) + Ridge   |
| 3     | 48.26      | 5.56       | Mean of Last 3    | Poly(2) + Linear  |
| 4     | 14.26      | 0.83       | Poly(3) + Ridge   | Alternating       |

## Marathon of Twenty-One Analysis

The marathon simulation relies on accurate prediction of future cards and spy values. Key findings:

- Dealer behavior is more predictable than player behavior across all tables
- Table 0 and Table 2 have the best performance in the game simulation
- Table 4 has the worst performance, suggesting the need for more tailored strategies
- Surrender strategy is critical for performance, with approximately 30-40% of hands being surrendered in optimal play
- Dealer busting frequency varies significantly by table, from 1% (Table 4) to 36% (Table 2)

## Methodology

1. Statistical analysis of raw data (mean, std, range, distribution)
2. Autocorrelation analysis at different lags
3. Card-spy value relationship analysis
4. Segment-based analysis for detecting pattern changes
5. Comparative testing of multiple prediction models
6. Card transition analysis for sequential patterns
7. Range-based segmentation for targeted modeling

## Conclusion

Each table in the Limestone challenge exhibits distinct patterns requiring tailored approaches. The most effective solutions leverage these table-specific characteristics rather than applying a one-size-fits-all approach. The range-based approach for Table 2 and the high-autocorrelation model for Table 3 demonstrate the value of targeted analysis in time series prediction tasks. 