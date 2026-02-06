"""
Visualization and Monitoring Dashboard for RTB System
Real-time performance monitoring and analytics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import json


class RTBDashboard:
    """
    Real-time dashboard for monitoring RTB system performance
    """
    
    def __init__(self):
        self.metrics_history = {
            'timestamp': deque(maxlen=1000),
            'win_rate': deque(maxlen=1000),
            'ctr': deque(maxlen=1000),
            'cpm': deque(maxlen=1000),
            'latency': deque(maxlen=1000),
            'budget_spent': deque(maxlen=1000),
            'roi': deque(maxlen=1000)
        }
        
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (15, 10)
    
    def update_metrics(self, metrics: Dict):
        """Update dashboard metrics"""
        self.metrics_history['timestamp'].append(datetime.now())
        
        for key in ['win_rate', 'ctr', 'cpm', 'latency', 'budget_spent', 'roi']:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
    
    def plot_realtime_metrics(self, save_path: Optional[str] = None):
        """Plot real-time metrics"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('RTB System Real-Time Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Win Rate
        ax = axes[0, 0]
        if len(self.metrics_history['win_rate']) > 0:
            ax.plot(list(self.metrics_history['win_rate']), color='#2E86AB', linewidth=2)
            ax.fill_between(
                range(len(self.metrics_history['win_rate'])),
                list(self.metrics_history['win_rate']),
                alpha=0.3,
                color='#2E86AB'
            )
            ax.set_title('Win Rate (%)', fontweight='bold')
            ax.set_ylabel('Win Rate')
            ax.set_xlabel('Time')
            ax.grid(True, alpha=0.3)
        
        # CTR
        ax = axes[0, 1]
        if len(self.metrics_history['ctr']) > 0:
            ax.plot(list(self.metrics_history['ctr']), color='#A23B72', linewidth=2)
            ax.fill_between(
                range(len(self.metrics_history['ctr'])),
                list(self.metrics_history['ctr']),
                alpha=0.3,
                color='#A23B72'
            )
            ax.set_title('Click-Through Rate (CTR)', fontweight='bold')
            ax.set_ylabel('CTR')
            ax.set_xlabel('Time')
            ax.grid(True, alpha=0.3)
        
        # CPM
        ax = axes[1, 0]
        if len(self.metrics_history['cpm']) > 0:
            ax.plot(list(self.metrics_history['cpm']), color='#F18F01', linewidth=2)
            ax.fill_between(
                range(len(self.metrics_history['cpm'])),
                list(self.metrics_history['cpm']),
                alpha=0.3,
                color='#F18F01'
            )
            ax.set_title('Cost Per Mille (CPM) - $', fontweight='bold')
            ax.set_ylabel('CPM ($)')
            ax.set_xlabel('Time')
            ax.grid(True, alpha=0.3)
        
        # Latency
        ax = axes[1, 1]
        if len(self.metrics_history['latency']) > 0:
            latencies = list(self.metrics_history['latency'])
            ax.plot(latencies, color='#C73E1D', linewidth=2)
            ax.axhline(y=100, color='red', linestyle='--', label='100ms SLA', linewidth=2)
            ax.fill_between(
                range(len(latencies)),
                latencies,
                alpha=0.3,
                color='#C73E1D'
            )
            ax.set_title('Response Latency (ms)', fontweight='bold')
            ax.set_ylabel('Latency (ms)')
            ax.set_xlabel('Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Budget Spent
        ax = axes[2, 0]
        if len(self.metrics_history['budget_spent']) > 0:
            ax.plot(list(self.metrics_history['budget_spent']), color='#6A994E', linewidth=2)
            ax.fill_between(
                range(len(self.metrics_history['budget_spent'])),
                list(self.metrics_history['budget_spent']),
                alpha=0.3,
                color='#6A994E'
            )
            ax.set_title('Cumulative Budget Spent ($)', fontweight='bold')
            ax.set_ylabel('Budget Spent ($)')
            ax.set_xlabel('Time')
            ax.grid(True, alpha=0.3)
        
        # ROI
        ax = axes[2, 1]
        if len(self.metrics_history['roi']) > 0:
            roi_values = list(self.metrics_history['roi'])
            colors = ['green' if r > 0 else 'red' for r in roi_values]
            ax.bar(range(len(roi_values)), roi_values, color=colors, alpha=0.6)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_title('Return on Investment (ROI)', fontweight='bold')
            ax.set_ylabel('ROI (%)')
            ax.set_xlabel('Time')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        return fig
    
    def plot_latency_distribution(
        self,
        latencies: List[float],
        save_path: Optional[str] = None
    ):
        """Plot latency distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax = axes[0]
        ax.hist(latencies, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax.axvline(x=100, color='red', linestyle='--', linewidth=2, label='100ms SLA')
        ax.axvline(x=np.mean(latencies), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(latencies):.1f}ms')
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Latency Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box plot
        ax = axes[1]
        bp = ax.boxplot(latencies, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('#2E86AB')
        bp['boxes'][0].set_alpha(0.7)
        ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100ms SLA')
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Latency Box Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"""
        Mean: {np.mean(latencies):.2f}ms
        Median: {np.median(latencies):.2f}ms
        P95: {np.percentile(latencies, 95):.2f}ms
        P99: {np.percentile(latencies, 99):.2f}ms
        Max: {np.max(latencies):.2f}ms
        """
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Latency distribution saved to {save_path}")
        
        return fig
    
    def plot_budget_pacing(
        self,
        budget_history: List[Dict],
        save_path: Optional[str] = None
    ):
        """Plot budget pacing over time"""
        df = pd.DataFrame(budget_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Budget Pacing Analysis', fontsize=16, fontweight='bold')
        
        # Spend over time
        ax = axes[0, 0]
        ax.plot(df['hour'], df['actual_spend'], label='Actual', linewidth=2, color='#2E86AB')
        ax.plot(df['hour'], df['ideal_spend'], label='Ideal', linewidth=2, linestyle='--', color='green')
        ax.fill_between(df['hour'], df['actual_spend'], df['ideal_spend'], alpha=0.3)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Cumulative Spend ($)')
        ax.set_title('Budget Spend vs. Ideal Pace')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Spend rate
        ax = axes[0, 1]
        ax.plot(df['hour'], df['spend_rate'], linewidth=2, color='#F18F01')
        ax.axhline(y=df['target_rate'].iloc[0], color='red', linestyle='--', label='Target Rate')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Spend Rate ($/hour)')
        ax.set_title('Spend Rate Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Pacing multiplier
        ax = axes[1, 0]
        ax.plot(df['hour'], df['pacing_multiplier'], linewidth=2, color='#A23B72')
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Pacing Multiplier')
        ax.set_title('Budget Pacing Multiplier')
        ax.grid(True, alpha=0.3)
        
        # Budget remaining
        ax = axes[1, 1]
        ax.fill_between(df['hour'], df['budget_remaining'], alpha=0.6, color='#6A994E')
        ax.plot(df['hour'], df['budget_remaining'], linewidth=2, color='#6A994E')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Budget Remaining ($)')
        ax.set_title('Remaining Budget')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Budget pacing plot saved to {save_path}")
        
        return fig
    
    def plot_ab_test_results(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot A/B test results comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('A/B Test Results Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['win_rate', 'ctr', 'cvr', 'cpm', 'roi', 'profit']
        metric_names = ['Win Rate', 'CTR', 'CVR', 'CPM ($)', 'ROI (%)', 'Profit ($)']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 3, idx % 3]
            
            if metric in results_df.columns:
                bars = ax.bar(
                    results_df['variant'],
                    results_df[metric],
                    color=['#2E86AB', '#F18F01', '#A23B72'][:len(results_df)],
                    alpha=0.7,
                    edgecolor='black'
                )
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{height:.4f}',
                        ha='center',
                        va='bottom',
                        fontweight='bold'
                    )
                
                ax.set_ylabel(name)
                ax.set_title(name, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"A/B test results saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: List[float],
        save_path: Optional[str] = None
    ):
        """Plot feature importance for CTR model"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)
        pos = np.arange(len(sorted_idx))
        
        ax.barh(
            pos,
            np.array(importance_scores)[sorted_idx],
            color='#2E86AB',
            alpha=0.7,
            edgecolor='black'
        )
        ax.set_yticks(pos)
        ax.set_yticklabels(np.array(feature_names)[sorted_idx])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('CTR Model Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def generate_summary_report(
        self,
        metrics: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """Generate text summary report"""
        report = []
        report.append("=" * 80)
        report.append("RTB SYSTEM PERFORMANCE SUMMARY")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("KEY PERFORMANCE INDICATORS:")
        report.append("-" * 80)
        
        kpis = {
            'Total Requests': metrics.get('total_requests', 0),
            'Total Wins': metrics.get('total_wins', 0),
            'Win Rate': f"{metrics.get('win_rate', 0)*100:.2f}%",
            'Average CTR': f"{metrics.get('avg_ctr', 0)*100:.2f}%",
            'Average CPM': f"${metrics.get('avg_cpm', 0):.2f}",
            'Total Spent': f"${metrics.get('total_spent', 0):.2f}",
            'Total Revenue': f"${metrics.get('total_revenue', 0):.2f}",
            'ROI': f"{metrics.get('roi', 0)*100:.2f}%",
        }
        
        for key, value in kpis.items():
            report.append(f"  {key:20s}: {value}")
        
        report.append("")
        report.append("LATENCY METRICS:")
        report.append("-" * 80)
        
        latency_metrics = {
            'Average Latency': f"{metrics.get('avg_latency_ms', 0):.2f}ms",
            'Median Latency': f"{metrics.get('median_latency_ms', 0):.2f}ms",
            'P95 Latency': f"{metrics.get('p95_latency_ms', 0):.2f}ms",
            'P99 Latency': f"{metrics.get('p99_latency_ms', 0):.2f}ms",
            'Max Latency': f"{metrics.get('max_latency_ms', 0):.2f}ms",
            '% Meeting SLA': f"{metrics.get('sla_compliance', 0)*100:.2f}%",
        }
        
        for key, value in latency_metrics.items():
            report.append(f"  {key:20s}: {value}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Summary report saved to {save_path}")
        
        return report_text
    
    def export_metrics_json(self, metrics: Dict, save_path: str):
        """Export metrics to JSON"""
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics exported to {save_path}")


def create_performance_dashboard(
    rtb_metrics: Dict,
    latency_data: List[float],
    budget_history: List[Dict],
    ab_test_results: pd.DataFrame,
    output_dir: str = '/home/claude/rtb_system/output'
):
    """
    Create complete performance dashboard with all visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    dashboard = RTBDashboard()
    
    print("\nGenerating performance dashboard...")
    
    # 1. Real-time metrics
    print("  - Creating real-time metrics dashboard...")
    dashboard.plot_realtime_metrics(
        save_path=f'{output_dir}/realtime_dashboard.png'
    )
    
    # 2. Latency distribution
    print("  - Creating latency distribution plots...")
    dashboard.plot_latency_distribution(
        latency_data,
        save_path=f'{output_dir}/latency_distribution.png'
    )
    
    # 3. Budget pacing
    print("  - Creating budget pacing analysis...")
    dashboard.plot_budget_pacing(
        budget_history,
        save_path=f'{output_dir}/budget_pacing.png'
    )
    
    # 4. A/B test results
    print("  - Creating A/B test comparison...")
    dashboard.plot_ab_test_results(
        ab_test_results,
        save_path=f'{output_dir}/ab_test_results.png'
    )
    
    # 5. Summary report
    print("  - Generating summary report...")
    report = dashboard.generate_summary_report(
        rtb_metrics,
        save_path=f'{output_dir}/summary_report.txt'
    )
    print(report)
    
    # 6. Export metrics
    print("  - Exporting metrics to JSON...")
    dashboard.export_metrics_json(
        rtb_metrics,
        save_path=f'{output_dir}/metrics.json'
    )
    
    print(f"\n✅ Dashboard created successfully in {output_dir}/")
