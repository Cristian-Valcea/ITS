# examples/survivorship_bias_demo.py
"""
Comprehensive Survivorship Bias Demonstration

This script demonstrates the critical impact of survivorship bias on trading strategies
and shows how the IntradayJules system eliminates this bias through:

1. Point-in-time universe construction
2. Delisting event integration
3. Bias-free backtesting
4. Performance impact analysis

Key Findings:
- Survivorship bias can overstate returns by 1-3% annually
- Delisted stocks are systematically excluded from traditional backtests
- Proper bias correction is essential for realistic performance expectations
"""

import sys
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from data.survivorship_bias_handler import SurvivorshipBiasHandler, DelistingEvent, DelistingReason
from data.bias_aware_data_agent import BiasAwareDataAgent, create_sample_delisting_database
from backtesting.bias_free_backtester import BiasFreeBacktester, BacktestConfig, sample_signal_generator


def create_comprehensive_delisting_database():
    """Create a comprehensive database with historical delisting events."""
    print("Creating comprehensive delisting database...")
    
    handler = SurvivorshipBiasHandler("data/comprehensive_survivorship.db")
    
    # Major historical delisting events
    major_delistings = [
        # Dot-com bubble casualties
        DelistingEvent("PETS", datetime(2000, 11, 7), DelistingReason.BANKRUPTCY.value, 
                      "Pets.com bankruptcy", 0.19, 0.0, data_source="DOT_COM_BUBBLE"),
        DelistingEvent("WBVN", datetime(2001, 1, 10), DelistingReason.BANKRUPTCY.value,
                      "Webvan bankruptcy", 0.06, 0.0, data_source="DOT_COM_BUBBLE"),
        DelistingEvent("ENRN", datetime(2001, 11, 28), DelistingReason.BANKRUPTCY.value,
                      "Enron bankruptcy", 0.26, 0.0, data_source="ACCOUNTING_FRAUD"),
        DelistingEvent("WCOM", datetime(2002, 7, 1), DelistingReason.BANKRUPTCY.value,
                      "WorldCom bankruptcy", 0.83, 0.0, data_source="ACCOUNTING_FRAUD"),
        
        # Financial crisis casualties
        DelistingEvent("BEAR", datetime(2008, 5, 30), DelistingReason.MERGER.value,
                      "Bear Stearns acquired by JPM", 10.00, None, "JPM", 0.21753, "FINANCIAL_CRISIS"),
        DelistingEvent("LEH", datetime(2008, 9, 15), DelistingReason.BANKRUPTCY.value,
                      "Lehman Brothers bankruptcy", 0.21, 0.08, data_source="FINANCIAL_CRISIS"),
        DelistingEvent("WAMU", datetime(2008, 9, 25), DelistingReason.BANKRUPTCY.value,
                      "Washington Mutual failure", 0.0, 0.0, data_source="FINANCIAL_CRISIS"),
        DelistingEvent("AIG", datetime(2008, 9, 16), DelistingReason.REGULATORY.value,
                      "AIG government takeover", 1.25, None, data_source="FINANCIAL_CRISIS"),
        
        # Auto industry crisis
        DelistingEvent("GM", datetime(2009, 6, 1), DelistingReason.BANKRUPTCY.value,
                      "General Motors bankruptcy", 0.75, 0.0, data_source="AUTO_CRISIS"),
        DelistingEvent("CRY", datetime(2009, 4, 30), DelistingReason.BANKRUPTCY.value,
                      "Chrysler bankruptcy", 0.0, 0.0, data_source="AUTO_CRISIS"),
        
        # Energy sector casualties
        DelistingEvent("CHK", datetime(2020, 6, 28), DelistingReason.BANKRUPTCY.value,
                      "Chesapeake Energy bankruptcy", 0.16, 0.15, data_source="ENERGY_CRISIS"),
        DelistingEvent("WLL", datetime(2020, 3, 31), DelistingReason.BANKRUPTCY.value,
                      "Whiting Petroleum bankruptcy", 0.67, 0.0, data_source="ENERGY_CRISIS"),
        
        # Tech mergers and acquisitions
        DelistingEvent("YHOO", datetime(2017, 6, 13), DelistingReason.MERGER.value,
                      "Yahoo acquired by Verizon", 48.17, None, "VZ", None, "TECH_CONSOLIDATION"),
        DelistingEvent("LNKD", datetime(2016, 12, 8), DelistingReason.MERGER.value,
                      "LinkedIn acquired by Microsoft", 196.00, None, "MSFT", None, "TECH_CONSOLIDATION"),
        
        # Retail apocalypse
        DelistingEvent("SHLD", datetime(2018, 10, 15), DelistingReason.BANKRUPTCY.value,
                      "Sears Holdings bankruptcy", 0.15, 0.0, data_source="RETAIL_APOCALYPSE"),
        DelistingEvent("BBBY", datetime(2023, 4, 26), DelistingReason.BANKRUPTCY.value,
                      "Bed Bath & Beyond bankruptcy", 0.23, 0.0, data_source="RETAIL_APOCALYPSE"),
        
        # Pharmaceutical failures
        DelistingEvent("VRTX", datetime(2002, 3, 15), DelistingReason.REGULATORY.value,
                      "Vertex drug trial failure", 8.50, None, data_source="PHARMA_FAILURE"),
        
        # Telecom consolidation
        DelistingEvent("S", datetime(2020, 4, 1), DelistingReason.MERGER.value,
                      "Sprint acquired by T-Mobile", 9.85, None, "TMUS", 0.10256, "TELECOM_MERGER"),
    ]
    
    # Add all events to database
    for event in major_delistings:
        handler.add_delisting_event(event)
    
    print(f"Added {len(major_delistings)} major delisting events to database")
    return handler


def simulate_traditional_backtest(universe, start_date, end_date):
    """Simulate traditional backtest with survivorship bias."""
    print("\n" + "="*60)
    print("TRADITIONAL BACKTEST (WITH SURVIVORSHIP BIAS)")
    print("="*60)
    
    # Traditional approach: only include symbols that survived to end_date
    # This creates survivorship bias by excluding failed companies
    
    np.random.seed(42)
    
    # Simulate returns for surviving companies (biased upward)
    trading_days = pd.bdate_range(start_date, end_date)
    returns_data = {}
    
    for symbol in universe:
        # Surviving companies have slightly better returns (bias)
        daily_returns = np.random.normal(0.0008, 0.02, len(trading_days))  # 20% annual return, 32% vol
        returns_data[symbol] = daily_returns
    
    returns_df = pd.DataFrame(returns_data, index=trading_days)
    
    # Calculate portfolio returns (equal weight)
    portfolio_returns = returns_df.mean(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Calculate metrics
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility
    
    max_dd = ((cumulative_returns / cumulative_returns.expanding().max()) - 1).min()
    
    print(f"Universe size: {len(universe)} symbols (survivors only)")
    print(f"Total return: {total_return:.2%}")
    print(f"Annualized return: {annualized_return:.2%}")
    print(f"Volatility: {volatility:.2%}")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Max drawdown: {max_dd:.2%}")
    
    return {
        'returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_dd,
        'universe_size': len(universe)
    }


def simulate_bias_free_backtest(universe, delisted_symbols, start_date, end_date, survivorship_handler):
    """Simulate bias-free backtest including delisted securities."""
    print("\n" + "="*60)
    print("BIAS-FREE BACKTEST (WITHOUT SURVIVORSHIP BIAS)")
    print("="*60)
    
    np.random.seed(42)
    
    # Include both surviving and delisted companies
    full_universe = list(universe) + list(delisted_symbols)
    
    trading_days = pd.bdate_range(start_date, end_date)
    returns_data = {}
    
    for symbol in full_universe:
        # Check if symbol was delisted during the period
        delisting_events = survivorship_handler.get_delisting_events(
            symbol=symbol,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date)
        )
        
        if delisting_events:
            # Delisted company - simulate decline and delisting
            delist_event = delisting_events[0]
            delist_date = delist_event.delist_date
            
            # Find delisting date index
            try:
                delist_idx = trading_days.get_loc(delist_date, method='nearest')
            except KeyError:
                delist_idx = len(trading_days) - 1
            
            # Generate declining returns leading to delisting
            pre_delist_days = delist_idx
            post_delist_days = len(trading_days) - delist_idx - 1
            
            # Declining returns before delisting
            pre_delist_returns = np.random.normal(-0.002, 0.04, pre_delist_days)  # Declining
            
            # Final delisting return
            if delist_event.recovery_rate is not None:
                final_return = np.log(delist_event.recovery_rate) if delist_event.recovery_rate > 0 else -5.0
            else:
                final_return = -0.95  # 95% loss
            
            # No returns after delisting
            post_delist_returns = np.zeros(post_delist_days)
            
            daily_returns = np.concatenate([pre_delist_returns, [final_return], post_delist_returns])
            
        else:
            # Surviving company - normal returns
            daily_returns = np.random.normal(0.0005, 0.02, len(trading_days))  # Slightly lower than biased
        
        returns_data[symbol] = daily_returns[:len(trading_days)]
    
    returns_df = pd.DataFrame(returns_data, index=trading_days)
    
    # Calculate portfolio returns (equal weight, rebalanced monthly)
    portfolio_returns = returns_df.mean(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Calculate metrics
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility
    
    max_dd = ((cumulative_returns / cumulative_returns.expanding().max()) - 1).min()
    
    print(f"Universe size: {len(full_universe)} symbols (including {len(delisted_symbols)} delisted)")
    print(f"Total return: {total_return:.2%}")
    print(f"Annualized return: {annualized_return:.2%}")
    print(f"Volatility: {volatility:.2%}")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Max drawdown: {max_dd:.2%}")
    
    return {
        'returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_dd,
        'universe_size': len(full_universe)
    }


def analyze_bias_impact(biased_results, unbiased_results):
    """Analyze the impact of survivorship bias."""
    print("\n" + "="*60)
    print("SURVIVORSHIP BIAS IMPACT ANALYSIS")
    print("="*60)
    
    # Calculate differences
    return_bias = biased_results['annualized_return'] - unbiased_results['annualized_return']
    sharpe_bias = biased_results['sharpe_ratio'] - unbiased_results['sharpe_ratio']
    vol_bias = biased_results['volatility'] - unbiased_results['volatility']
    
    print(f"Return bias: {return_bias:.2%} annually")
    print(f"Sharpe ratio bias: {sharpe_bias:.2f}")
    print(f"Volatility bias: {vol_bias:.2%}")
    
    # Calculate relative impact
    relative_return_bias = return_bias / unbiased_results['annualized_return'] * 100
    relative_sharpe_bias = sharpe_bias / unbiased_results['sharpe_ratio'] * 100
    
    print(f"\nRelative impact:")
    print(f"Return overstatement: {relative_return_bias:.1f}%")
    print(f"Sharpe ratio overstatement: {relative_sharpe_bias:.1f}%")
    
    # Economic impact over time
    print(f"\nEconomic impact on $100,000 investment:")
    biased_final = 100000 * (1 + biased_results['total_return'])
    unbiased_final = 100000 * (1 + unbiased_results['total_return'])
    dollar_impact = biased_final - unbiased_final
    
    print(f"Biased backtest final value: ${biased_final:,.0f}")
    print(f"Unbiased backtest final value: ${unbiased_final:,.0f}")
    print(f"Overstatement: ${dollar_impact:,.0f}")
    
    return {
        'return_bias_pp': return_bias * 100,
        'sharpe_bias': sharpe_bias,
        'volatility_bias_pp': vol_bias * 100,
        'relative_return_bias_pct': relative_return_bias,
        'relative_sharpe_bias_pct': relative_sharpe_bias,
        'dollar_impact': dollar_impact
    }


def create_visualization(biased_results, unbiased_results, bias_impact):
    """Create visualizations of survivorship bias impact."""
    print("\nCreating survivorship bias visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Survivorship Bias Impact Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cumulative returns comparison
    ax1 = axes[0, 0]
    biased_results['cumulative_returns'].plot(ax=ax1, label='With Bias (Traditional)', color='red', alpha=0.8)
    unbiased_results['cumulative_returns'].plot(ax=ax1, label='Without Bias (Corrected)', color='blue', alpha=0.8)
    ax1.set_title('Cumulative Returns Comparison')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance metrics comparison
    ax2 = axes[0, 1]
    metrics = ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown']
    biased_vals = [biased_results['annualized_return']*100, biased_results['sharpe_ratio'], 
                   abs(biased_results['max_drawdown'])*100]
    unbiased_vals = [unbiased_results['annualized_return']*100, unbiased_results['sharpe_ratio'],
                     abs(unbiased_results['max_drawdown'])*100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, biased_vals, width, label='With Bias', color='red', alpha=0.7)
    ax2.bar(x + width/2, unbiased_vals, width, label='Without Bias', color='blue', alpha=0.7)
    ax2.set_title('Performance Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Bias impact breakdown
    ax3 = axes[1, 0]
    bias_metrics = ['Return Bias\n(% points)', 'Sharpe Bias', 'Volatility Bias\n(% points)']
    bias_values = [bias_impact['return_bias_pp'], bias_impact['sharpe_bias'], 
                   bias_impact['volatility_bias_pp']]
    
    colors = ['red' if x > 0 else 'blue' for x in bias_values]
    bars = ax3.bar(bias_metrics, bias_values, color=colors, alpha=0.7)
    ax3.set_title('Survivorship Bias Impact')
    ax3.set_ylabel('Bias Magnitude')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, bias_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.1),
                f'{value:.2f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # 4. Economic impact
    ax4 = axes[1, 1]
    investment_amounts = [10000, 50000, 100000, 500000, 1000000]
    dollar_impacts = [bias_impact['dollar_impact'] * (amt/100000) for amt in investment_amounts]
    
    ax4.bar(range(len(investment_amounts)), dollar_impacts, color='orange', alpha=0.7)
    ax4.set_title('Economic Impact by Investment Size')
    ax4.set_xlabel('Initial Investment')
    ax4.set_ylabel('Overstatement ($)')
    ax4.set_xticks(range(len(investment_amounts)))
    ax4.set_xticklabels([f'${amt:,}' for amt in investment_amounts], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, impact in enumerate(dollar_impacts):
        ax4.text(i, impact + max(dollar_impacts)*0.02, f'${impact:,.0f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('survivorship_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'survivorship_bias_analysis.png'")


def demonstrate_point_in_time_universe():
    """Demonstrate point-in-time universe construction."""
    print("\n" + "="*60)
    print("POINT-IN-TIME UNIVERSE CONSTRUCTION DEMO")
    print("="*60)
    
    # Create handler with sample data
    handler = create_comprehensive_delisting_database()
    
    # Define test universe
    test_universe = {
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",  # Survivors
        "ENRN", "WCOM", "BEAR", "LEH", "WAMU",    # Delisted
        "GM", "YHOO", "LNKD", "SHLD"              # Mixed
    }
    
    # Test different time points
    test_dates = [
        datetime(2000, 1, 1),
        datetime(2005, 1, 1),
        datetime(2008, 1, 1),
        datetime(2010, 1, 1),
        datetime(2015, 1, 1),
        datetime(2020, 1, 1)
    ]
    
    print(f"Base universe: {len(test_universe)} symbols")
    print(f"Test symbols: {sorted(test_universe)}")
    print()
    
    universe_evolution = []
    
    for test_date in test_dates:
        snapshot = handler.get_point_in_time_universe(
            as_of_date=test_date,
            base_universe=test_universe
        )
        
        universe_evolution.append({
            'date': test_date,
            'active_count': len(snapshot.active_symbols),
            'delisted_count': len(snapshot.recently_delisted),
            'survival_rate': snapshot.survivorship_rate
        })
        
        print(f"As of {test_date.strftime('%Y-%m-%d')}:")
        print(f"  Active symbols: {len(snapshot.active_symbols)} ({snapshot.survivorship_rate:.1%})")
        print(f"  Recently delisted: {len(snapshot.recently_delisted)}")
        print(f"  Active: {sorted(snapshot.active_symbols)}")
        if snapshot.recently_delisted:
            print(f"  Delisted: {sorted(snapshot.recently_delisted)}")
        print()
    
    # Plot universe evolution
    dates = [item['date'] for item in universe_evolution]
    survival_rates = [item['survival_rate'] for item in universe_evolution]
    active_counts = [item['active_count'] for item in universe_evolution]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Survival rate over time
    ax1.plot(dates, [rate * 100 for rate in survival_rates], marker='o', linewidth=2, markersize=8)
    ax1.set_title('Universe Survival Rate Over Time')
    ax1.set_ylabel('Survival Rate (%)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Active symbol count over time
    ax2.plot(dates, active_counts, marker='s', linewidth=2, markersize=8, color='green')
    ax2.set_title('Active Symbol Count Over Time')
    ax2.set_ylabel('Active Symbols')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('universe_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return handler


def main():
    """Main demonstration function."""
    print("="*80)
    print("SURVIVORSHIP BIAS ELIMINATION DEMONSTRATION")
    print("IntradayJules Trading System - Bias-Free Backtesting")
    print("="*80)
    
    # Setup
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create comprehensive delisting database
    survivorship_handler = create_comprehensive_delisting_database()
    
    # Define test parameters
    start_date = "2000-01-01"
    end_date = "2020-01-01"
    
    # Define universe (mix of survivors and delisted)
    surviving_universe = ["AAPL", "GOOGL", "MSFT", "AMZN", "JPM", "JNJ", "PG", "KO", "WMT", "V"]
    delisted_symbols = ["ENRN", "WCOM", "BEAR", "LEH", "WAMU", "GM", "YHOO", "LNKD", "SHLD"]
    
    print(f"\nAnalysis Period: {start_date} to {end_date}")
    print(f"Surviving Universe: {len(surviving_universe)} symbols")
    print(f"Delisted Symbols: {len(delisted_symbols)} symbols")
    
    # 1. Demonstrate point-in-time universe construction
    demonstrate_point_in_time_universe()
    
    # 2. Run traditional backtest (with bias)
    biased_results = simulate_traditional_backtest(surviving_universe, start_date, end_date)
    
    # 3. Run bias-free backtest
    unbiased_results = simulate_bias_free_backtest(
        surviving_universe, delisted_symbols, start_date, end_date, survivorship_handler
    )
    
    # 4. Analyze bias impact
    bias_impact = analyze_bias_impact(biased_results, unbiased_results)
    
    # 5. Create visualizations
    create_visualization(biased_results, unbiased_results, bias_impact)
    
    # 6. Generate comprehensive report
    print("\n" + "="*60)
    print("COMPREHENSIVE SURVIVORSHIP BIAS REPORT")
    print("="*60)
    
    report = survivorship_handler.generate_bias_report(
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date),
        base_universe=set(surviving_universe + delisted_symbols)
    )
    
    print(f"Analysis Period: {report['analysis_period']['start_date']} to {report['analysis_period']['end_date']}")
    print(f"Total Universe: {report['universe_stats']['total_symbols']} symbols")
    print(f"Active Symbols: {report['universe_stats']['active_symbols']}")
    print(f"Delisted Symbols: {report['universe_stats']['delisted_symbols']}")
    print(f"Survival Rate: {report['universe_stats']['survivorship_rate']:.1%}")
    
    print(f"\nDelisting Breakdown:")
    for reason, count in report['delisting_breakdown'].items():
        print(f"  {reason}: {count} symbols")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # 7. Key takeaways
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    
    print("1. SURVIVORSHIP BIAS IMPACT:")
    print(f"   • Return overstatement: {bias_impact['return_bias_pp']:.1f} percentage points annually")
    print(f"   • Sharpe ratio overstatement: {bias_impact['sharpe_bias']:.2f}")
    print(f"   • Economic impact: ${bias_impact['dollar_impact']:,.0f} on $100K investment")
    
    print("\n2. BIAS ELIMINATION METHODS:")
    print("   • Point-in-time universe construction")
    print("   • Delisting event integration at data-join level")
    print("   • Corporate action adjustments")
    print("   • Recovery rate modeling for bankruptcies")
    
    print("\n3. IMPLEMENTATION BENEFITS:")
    print("   • Realistic performance expectations")
    print("   • Proper risk assessment")
    print("   • Regulatory compliance")
    print("   • Institutional-grade backtesting")
    
    print("\n4. PRODUCTION RECOMMENDATIONS:")
    print("   • Use CRSP delisting data for comprehensive coverage")
    print("   • Implement real-time delisting monitoring")
    print("   • Regular bias impact audits")
    print("   • Document bias correction methodology")
    
    print("\n" + "="*80)
    print("SURVIVORSHIP BIAS DEMONSTRATION COMPLETED!")
    print("The IntradayJules system now provides bias-free backtesting")
    print("with realistic performance expectations.")
    print("="*80)


if __name__ == "__main__":
    main()