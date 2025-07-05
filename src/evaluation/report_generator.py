"""
ReportGenerator module for creating evaluation reports.

This module handles:
- Generating human-readable evaluation reports
- Saving trade logs to CSV files
- Creating summary reports in text format
- File I/O operations and error handling
"""

import os
import logging
from typing import Optional, Dict, Any
import pandas as pd


class ReportGenerator:
    """
    Handles generation and saving of evaluation reports and trade logs.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the ReportGenerator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reports_dir = config.get('reports_dir', 'reports/')
        
        # Ensure reports directory exists
        os.makedirs(self.reports_dir, exist_ok=True)
        self.logger.info(f"ReportGenerator initialized. Reports will be saved to: {self.reports_dir}")
    
    def generate_report(
        self, 
        metrics: Dict[str, Any], 
        trade_log_df: Optional[pd.DataFrame], 
        model_name: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Dictionary of calculated performance metrics
            trade_log_df: DataFrame containing trade log
            model_name: Name/identifier for the model
            additional_info: Optional additional information to include
            
        Returns:
            Path to the saved text report
        """
        self.logger.info(f"Generating evaluation report for model: {model_name}")
        
        # Generate file names
        report_name_base = f"eval_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        report_txt_path = os.path.join(self.reports_dir, f"{report_name_base}_summary.txt")
        report_trades_csv_path = os.path.join(self.reports_dir, f"{report_name_base}_trades.csv")
        
        # Generate and save text report
        self._save_text_report(report_txt_path, metrics, model_name, additional_info)
        
        # Save trade log if available
        self._save_trade_log(report_trades_csv_path, trade_log_df)
        
        return report_txt_path
    
    def _save_text_report(
        self, 
        report_path: str, 
        metrics: Dict[str, Any], 
        model_name: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save the text summary report.
        
        Args:
            report_path: Path to save the report
            metrics: Performance metrics dictionary
            model_name: Name of the model
            additional_info: Additional information to include
        """
        try:
            report_content = self._generate_report_content(metrics, model_name, additional_info)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            self.logger.info(f"Evaluation summary report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation report to {report_path}: {e}", exc_info=True)
    
    def _generate_report_content(
        self, 
        metrics: Dict[str, Any], 
        model_name: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate the content for the text report.
        
        Args:
            metrics: Performance metrics dictionary
            model_name: Name of the model
            additional_info: Additional information to include
            
        Returns:
            Formatted report content as string
        """
        # Header
        report_content = f"Evaluation Report for Model: {model_name}\n"
        report_content += f"Report Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += "=" * 60 + "\n\n"
        
        # Performance Metrics Section
        report_content += "PERFORMANCE METRICS:\n"
        report_content += "-" * 30 + "\n"
        
        # Group metrics by category for better readability
        return_metrics = {}
        risk_metrics = {}
        trading_metrics = {}
        other_metrics = {}
        
        for key, value in metrics.items():
            if 'return' in key.lower() or 'capital' in key.lower():
                return_metrics[key] = value
            elif any(risk_term in key.lower() for risk_term in ['sharpe', 'drawdown', 'sortino', 'volatility']):
                risk_metrics[key] = value
            elif any(trade_term in key.lower() for trade_term in ['trades', 'turnover', 'win_rate']):
                trading_metrics[key] = value
            else:
                other_metrics[key] = value
        
        # Add return metrics
        if return_metrics:
            report_content += "\nReturn Metrics:\n"
            for key, value in return_metrics.items():
                formatted_key = key.replace('_', ' ').title()
                report_content += f"  {formatted_key}: {self._format_metric_value(key, value)}\n"
        
        # Add risk metrics
        if risk_metrics:
            report_content += "\nRisk Metrics:\n"
            for key, value in risk_metrics.items():
                formatted_key = key.replace('_', ' ').title()
                report_content += f"  {formatted_key}: {self._format_metric_value(key, value)}\n"
        
        # Add trading metrics
        if trading_metrics:
            report_content += "\nTrading Metrics:\n"
            for key, value in trading_metrics.items():
                formatted_key = key.replace('_', ' ').title()
                report_content += f"  {formatted_key}: {self._format_metric_value(key, value)}\n"
        
        # Add other metrics
        if other_metrics:
            report_content += "\nOther Metrics:\n"
            for key, value in other_metrics.items():
                formatted_key = key.replace('_', ' ').title()
                report_content += f"  {formatted_key}: {self._format_metric_value(key, value)}\n"
        
        # Additional information section
        if additional_info:
            report_content += "\n" + "=" * 60 + "\n"
            report_content += "ADDITIONAL INFORMATION:\n"
            report_content += "-" * 30 + "\n"
            for key, value in additional_info.items():
                formatted_key = key.replace('_', ' ').title()
                report_content += f"  {formatted_key}: {value}\n"
        
        # Footer
        report_content += "\n" + "=" * 60 + "\n"
        report_content += "Report generated by IntradayJules Trading System\n"
        
        return report_content
    
    def _format_metric_value(self, key: str, value: Any) -> str:
        """
        Format metric values for display.
        
        Args:
            key: Metric key name
            value: Metric value
            
        Returns:
            Formatted value string
        """
        if isinstance(value, (int, float)):
            if 'pct' in key.lower() or 'ratio' in key.lower():
                return f"{value:.4f}"
            elif 'capital' in key.lower() or 'value' in key.lower():
                return f"${value:,.2f}"
            else:
                return f"{value:.4f}"
        else:
            return str(value)
    
    def _save_trade_log(self, csv_path: str, trade_log_df: Optional[pd.DataFrame]) -> None:
        """
        Save the trade log to a CSV file.
        
        Args:
            csv_path: Path to save the CSV file
            trade_log_df: DataFrame containing trade log
        """
        if trade_log_df is not None and not trade_log_df.empty:
            try:
                trade_log_df.to_csv(csv_path, index=False)
                self.logger.info(f"Trade log saved to: {csv_path}")
            except Exception as e:
                self.logger.error(f"Error saving trade log CSV to {csv_path}: {e}", exc_info=True)
        else:
            self.logger.info("No trades to save for this evaluation run.")
    
    def generate_comparison_report(
        self, 
        results_list: list, 
        comparison_name: str = "model_comparison"
    ) -> str:
        """
        Generate a comparison report for multiple model evaluations.
        
        Args:
            results_list: List of dictionaries containing model results
            comparison_name: Name for the comparison report
            
        Returns:
            Path to the saved comparison report
        """
        self.logger.info(f"Generating comparison report: {comparison_name}")
        
        report_name = f"{comparison_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(self.reports_dir, report_name)
        
        try:
            report_content = self._generate_comparison_content(results_list, comparison_name)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            self.logger.info(f"Comparison report saved to: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating comparison report: {e}", exc_info=True)
            return ""
    
    def _generate_comparison_content(self, results_list: list, comparison_name: str) -> str:
        """
        Generate content for comparison report.
        
        Args:
            results_list: List of model results
            comparison_name: Name of the comparison
            
        Returns:
            Formatted comparison report content
        """
        content = f"Model Comparison Report: {comparison_name}\n"
        content += f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += "=" * 80 + "\n\n"
        
        if not results_list:
            content += "No results to compare.\n"
            return content
        
        # Create comparison table
        content += "PERFORMANCE COMPARISON:\n"
        content += "-" * 40 + "\n"
        
        # Extract common metrics
        all_metrics = set()
        for result in results_list:
            if 'metrics' in result:
                all_metrics.update(result['metrics'].keys())
        
        # Table header
        content += f"{'Model':<20}"
        for metric in sorted(all_metrics):
            content += f"{metric.replace('_', ' ').title():<15}"
        content += "\n" + "-" * (20 + len(all_metrics) * 15) + "\n"
        
        # Table rows
        for result in results_list:
            model_name = result.get('model_name', 'Unknown')[:19]
            content += f"{model_name:<20}"
            
            metrics = result.get('metrics', {})
            for metric in sorted(all_metrics):
                value = metrics.get(metric, 'N/A')
                if isinstance(value, (int, float)):
                    content += f"{value:<15.4f}"
                else:
                    content += f"{str(value):<15}"
            content += "\n"
        
        content += "\n" + "=" * 80 + "\n"
        return content