# REWARD COMPONENT DIAGNOSTIC REPORT
Generated: 2025-08-04 11:52:14

## OVERALL STATISTICS
- Total Steps: 4,825
- Total Episodes: 50
- Mean Episode Length: 96.5
- **Mean Total Reward: 0.020012**

## REWARD COMPONENT BREAKDOWN
### Normalized Pnl
- Mean: -0.001275
- Sum: -6.153
- Contribution: -6.4%
- Std: 0.019309
- Range: [-0.267311, 0.311237]

### Holding Bonus
- Mean: 0.000000
- Sum: 0.000
- Contribution: 0.0%
- Std: 0.000000
- Range: [0.000000, 0.000000]

### Smoothed Penalty
- Mean: -0.018786
- Sum: -90.642
- Contribution: -93.9%
- Std: 0.027876
- Range: [-0.281594, 0.000000]

### Exploration Bonus
- Mean: 0.039666
- Sum: 191.388
- Contribution: 198.2%
- Std: 0.005515
- Range: [0.030864, 0.050000]

### Directional Bonus
- Mean: 0.000408
- Sum: 1.967
- Contribution: 2.0%
- Std: 0.001228
- Range: [0.000000, 0.026731]

## KEY FINDINGS
- **Most Negative Component**: smoothed_penalty (sum: -90.642)
- **Most Positive Component**: exploration_bonus (sum: 191.388)
- **Primary Bottleneck**: smoothed_penalty is dragging down total reward

## RECOMMENDATIONS
1. **Target smoothed_penalty** for immediate tuning
2. Consider adjusting parameters related to smoothed_penalty
3. Boost exploration_bonus if it's providing good signal

## FILES GENERATED
- step_data.csv: Raw step-by-step data
- episode_data.csv: Episode-level summaries
- overall_statistics.csv: Summary statistics
- reward_components_timeseries.png: Time series plots
- reward_components_distributions.png: Distribution box plots
- cumulative_contributions.png: Cumulative contribution analysis
- behavioral_correlations.png: Behavioral correlation heatmap
- performance_correlations.png: Performance correlation heatmap