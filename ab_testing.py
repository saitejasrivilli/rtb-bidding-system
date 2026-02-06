"""
A/B Testing Framework for RTB Systems
Evaluate and compare different bidding strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import scipy.stats as stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class Experiment:
    """Represents an A/B test experiment"""
    experiment_id: str
    name: str
    variants: List[str]
    traffic_split: Dict[str, float]  # variant -> traffic percentage
    start_time: datetime
    end_time: Optional[datetime] = None
    is_active: bool = True
    primary_metric: str = 'revenue'
    secondary_metrics: List[str] = field(default_factory=list)


@dataclass
class VariantMetrics:
    """Metrics for a single variant"""
    variant_name: str
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    cost: float = 0.0
    
    bid_requests: int = 0
    wins: int = 0
    avg_bid: float = 0.0
    avg_ctr: float = 0.0
    avg_cvr: float = 0.0
    avg_cpm: float = 0.0
    avg_cpc: float = 0.0
    avg_cpa: float = 0.0
    roi: float = 0.0
    
    def update(
        self,
        impression: bool = False,
        click: bool = False,
        conversion: bool = False,
        revenue_delta: float = 0.0,
        cost_delta: float = 0.0,
        won_bid: bool = False,
        bid_amount: float = 0.0
    ):
        """Update metrics"""
        self.bid_requests += 1
        
        if won_bid:
            self.wins += 1
            self.cost += cost_delta
        
        if impression:
            self.impressions += 1
        
        if click:
            self.clicks += 1
        
        if conversion:
            self.conversions += 1
            self.revenue += revenue_delta
        
        # Update bid average
        if bid_amount > 0:
            total_bids = self.avg_bid * (self.bid_requests - 1) + bid_amount
            self.avg_bid = total_bids / self.bid_requests
        
        # Calculate rates
        if self.impressions > 0:
            self.avg_ctr = self.clicks / self.impressions
            self.avg_cpm = (self.cost / self.impressions) * 1000
        
        if self.clicks > 0:
            self.avg_cvr = self.conversions / self.clicks
            self.avg_cpc = self.cost / self.clicks
        
        if self.conversions > 0:
            self.avg_cpa = self.cost / self.conversions
        
        if self.cost > 0:
            self.roi = (self.revenue - self.cost) / self.cost
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'variant': self.variant_name,
            'impressions': self.impressions,
            'clicks': self.clicks,
            'conversions': self.conversions,
            'revenue': self.revenue,
            'cost': self.cost,
            'win_rate': self.wins / self.bid_requests if self.bid_requests > 0 else 0,
            'ctr': self.avg_ctr,
            'cvr': self.avg_cvr,
            'cpm': self.avg_cpm,
            'cpc': self.avg_cpc,
            'cpa': self.avg_cpa,
            'roi': self.roi,
            'profit': self.revenue - self.cost
        }


class ABTesting:
    """
    A/B Testing framework for RTB systems
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.experiments: Dict[str, Experiment] = {}
        self.variant_metrics: Dict[str, Dict[str, VariantMetrics]] = defaultdict(dict)
        self.assignment_history: Dict[str, str] = {}  # user_id -> variant
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        variants: List[str],
        traffic_split: Optional[Dict[str, float]] = None,
        primary_metric: str = 'revenue',
        secondary_metrics: Optional[List[str]] = None
    ) -> Experiment:
        """
        Create a new A/B test experiment
        
        Args:
            experiment_id: Unique identifier
            name: Human-readable name
            variants: List of variant names (e.g., ['control', 'treatment'])
            traffic_split: Traffic allocation (must sum to 1.0)
            primary_metric: Main metric to optimize
            secondary_metrics: Additional metrics to track
        """
        if traffic_split is None:
            # Equal split
            split = 1.0 / len(variants)
            traffic_split = {v: split for v in variants}
        
        # Validate traffic split
        if abs(sum(traffic_split.values()) - 1.0) > 1e-6:
            raise ValueError("Traffic split must sum to 1.0")
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            variants=variants,
            traffic_split=traffic_split,
            start_time=datetime.now(),
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics or []
        )
        
        self.experiments[experiment_id] = experiment
        
        # Initialize metrics for each variant
        for variant in variants:
            self.variant_metrics[experiment_id][variant] = VariantMetrics(variant)
        
        print(f"Created experiment '{name}' with variants: {variants}")
        print(f"Traffic split: {traffic_split}")
        
        return experiment
    
    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """
        Assign user to a variant
        Uses consistent hashing for stable assignments
        
        Args:
            experiment_id: Experiment ID
            user_id: User identifier
        
        Returns:
            Assigned variant name
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Check if user already assigned
        assignment_key = f"{experiment_id}:{user_id}"
        if assignment_key in self.assignment_history:
            return self.assignment_history[assignment_key]
        
        experiment = self.experiments[experiment_id]
        
        # Consistent hashing
        hash_value = hash(assignment_key) % 10000 / 10000.0
        
        # Assign based on traffic split
        cumulative = 0.0
        for variant, percentage in experiment.traffic_split.items():
            cumulative += percentage
            if hash_value < cumulative:
                self.assignment_history[assignment_key] = variant
                return variant
        
        # Fallback
        variant = experiment.variants[0]
        self.assignment_history[assignment_key] = variant
        return variant
    
    def record_event(
        self,
        experiment_id: str,
        variant: str,
        impression: bool = False,
        click: bool = False,
        conversion: bool = False,
        revenue: float = 0.0,
        cost: float = 0.0,
        won_bid: bool = False,
        bid_amount: float = 0.0
    ):
        """Record an event for a variant"""
        if experiment_id not in self.variant_metrics:
            return
        
        if variant not in self.variant_metrics[experiment_id]:
            return
        
        metrics = self.variant_metrics[experiment_id][variant]
        metrics.update(
            impression=impression,
            click=click,
            conversion=conversion,
            revenue_delta=revenue,
            cost_delta=cost,
            won_bid=won_bid,
            bid_amount=bid_amount
        )
    
    def get_results(self, experiment_id: str) -> pd.DataFrame:
        """Get experiment results as DataFrame"""
        if experiment_id not in self.variant_metrics:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        results = []
        for variant, metrics in self.variant_metrics[experiment_id].items():
            results.append(metrics.to_dict())
        
        return pd.DataFrame(results)
    
    def calculate_statistical_significance(
        self,
        experiment_id: str,
        metric: str = 'ctr',
        control_variant: str = 'control',
        alpha: float = 0.05
    ) -> Dict:
        """
        Calculate statistical significance using two-sample t-test
        
        Args:
            experiment_id: Experiment ID
            metric: Metric to test ('ctr', 'cvr', 'roi', etc.)
            control_variant: Name of control variant
            alpha: Significance level (default 0.05)
        
        Returns:
            Dictionary with test results
        """
        results = {}
        
        control_metrics = self.variant_metrics[experiment_id][control_variant]
        
        for variant_name, variant_metrics in self.variant_metrics[experiment_id].items():
            if variant_name == control_variant:
                continue
            
            # Get metric values
            if metric == 'ctr':
                control_successes = control_metrics.clicks
                control_trials = control_metrics.impressions
                variant_successes = variant_metrics.clicks
                variant_trials = variant_metrics.impressions
            elif metric == 'cvr':
                control_successes = control_metrics.conversions
                control_trials = control_metrics.clicks
                variant_successes = variant_metrics.conversions
                variant_trials = variant_metrics.clicks
            else:
                # For continuous metrics, use t-test
                # (This is simplified; in practice, store individual samples)
                continue
            
            # Two-proportion z-test
            if control_trials == 0 or variant_trials == 0:
                results[variant_name] = {
                    'p_value': 1.0,
                    'significant': False,
                    'lift': 0.0,
                    'confidence_interval': (0.0, 0.0)
                }
                continue
            
            control_rate = control_successes / control_trials
            variant_rate = variant_successes / variant_trials
            
            # Pooled proportion
            pooled_p = (control_successes + variant_successes) / (control_trials + variant_trials)
            
            # Standard error
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1/control_trials + 1/variant_trials))
            
            if se == 0:
                results[variant_name] = {
                    'p_value': 1.0,
                    'significant': False,
                    'lift': 0.0,
                    'confidence_interval': (0.0, 0.0)
                }
                continue
            
            # Z-statistic
            z = (variant_rate - control_rate) / se
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            # Lift
            lift = (variant_rate - control_rate) / control_rate if control_rate > 0 else 0
            
            # Confidence interval for difference
            z_critical = stats.norm.ppf(1 - alpha/2)
            margin = z_critical * se
            ci = (variant_rate - control_rate - margin, variant_rate - control_rate + margin)
            
            results[variant_name] = {
                'control_rate': control_rate,
                'variant_rate': variant_rate,
                'p_value': p_value,
                'significant': p_value < alpha,
                'lift': lift,
                'confidence_interval': ci,
                'z_statistic': z
            }
        
        return results
    
    def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        Calculate required sample size for experiment
        
        Args:
            baseline_rate: Current conversion/click rate
            minimum_detectable_effect: Minimum effect to detect (e.g., 0.05 for 5%)
            alpha: Significance level
            power: Statistical power (1 - beta)
        
        Returns:
            Required sample size per variant
        """
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        pooled_p = (p1 + p2) / 2
        
        n = (
            (z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) +
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 /
            (p1 - p2) ** 2
        )
        
        return int(np.ceil(n))
    
    def plot_results(
        self,
        experiment_id: str,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot experiment results
        
        Args:
            experiment_id: Experiment ID
            metrics: Metrics to plot (default: primary + secondary)
            save_path: Path to save figure
        """
        if metrics is None:
            experiment = self.experiments[experiment_id]
            metrics = [experiment.primary_metric] + experiment.secondary_metrics
        
        results_df = self.get_results(experiment_id)
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            if metric in results_df.columns:
                results_df.plot(
                    x='variant',
                    y=metric,
                    kind='bar',
                    ax=ax,
                    legend=False
                )
                ax.set_title(f'{metric.upper()}')
                ax.set_xlabel('Variant')
                ax.set_ylabel(metric.upper())
                
                # Add value labels on bars
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.4f')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
    
    def generate_report(self, experiment_id: str) -> str:
        """
        Generate a text report of experiment results
        """
        if experiment_id not in self.experiments:
            return f"Experiment {experiment_id} not found"
        
        experiment = self.experiments[experiment_id]
        results_df = self.get_results(experiment_id)
        
        report = []
        report.append("=" * 80)
        report.append(f"Experiment Report: {experiment.name}")
        report.append("=" * 80)
        report.append(f"Experiment ID: {experiment_id}")
        report.append(f"Start Time: {experiment.start_time}")
        report.append(f"Status: {'Active' if experiment.is_active else 'Completed'}")
        report.append(f"Primary Metric: {experiment.primary_metric}")
        report.append("")
        
        report.append("Results by Variant:")
        report.append("-" * 80)
        report.append(results_df.to_string(index=False))
        report.append("")
        
        # Statistical significance
        if len(experiment.variants) > 1:
            report.append("Statistical Significance (vs Control):")
            report.append("-" * 80)
            
            control_variant = experiment.variants[0]
            sig_results = self.calculate_statistical_significance(
                experiment_id,
                metric='ctr',
                control_variant=control_variant
            )
            
            for variant, result in sig_results.items():
                report.append(f"\n{variant}:")
                report.append(f"  Lift: {result['lift']*100:.2f}%")
                report.append(f"  P-value: {result['p_value']:.4f}")
                report.append(f"  Significant: {result['significant']}")
                report.append(f"  95% CI: ({result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


class MultiArmedBandit:
    """
    Multi-Armed Bandit for dynamic traffic allocation
    Alternative to fixed A/B testing
    """
    
    def __init__(
        self,
        variants: List[str],
        algorithm: str = 'thompson_sampling'
    ):
        """
        Args:
            variants: List of variant names
            algorithm: 'epsilon_greedy', 'ucb', or 'thompson_sampling'
        """
        self.variants = variants
        self.algorithm = algorithm
        
        # Track metrics
        self.successes = {v: 0 for v in variants}
        self.failures = {v: 0 for v in variants}
        self.total_trials = {v: 0 for v in variants}
        
        # Algorithm parameters
        self.epsilon = 0.1  # for epsilon-greedy
        self.c = 2.0  # for UCB
    
    def select_variant(self) -> str:
        """Select variant based on algorithm"""
        if self.algorithm == 'epsilon_greedy':
            return self._epsilon_greedy()
        elif self.algorithm == 'ucb':
            return self._ucb()
        elif self.algorithm == 'thompson_sampling':
            return self._thompson_sampling()
        else:
            return np.random.choice(self.variants)
    
    def _epsilon_greedy(self) -> str:
        """Epsilon-greedy selection"""
        if np.random.random() < self.epsilon:
            # Explore
            return np.random.choice(self.variants)
        else:
            # Exploit
            rates = {}
            for v in self.variants:
                if self.total_trials[v] == 0:
                    rates[v] = 0.0
                else:
                    rates[v] = self.successes[v] / self.total_trials[v]
            
            return max(rates, key=rates.get)
    
    def _ucb(self) -> str:
        """Upper Confidence Bound selection"""
        total_trials_sum = sum(self.total_trials.values())
        
        if total_trials_sum == 0:
            return np.random.choice(self.variants)
        
        ucb_scores = {}
        for v in self.variants:
            if self.total_trials[v] == 0:
                ucb_scores[v] = float('inf')
            else:
                mean = self.successes[v] / self.total_trials[v]
                exploration = np.sqrt(
                    self.c * np.log(total_trials_sum) / self.total_trials[v]
                )
                ucb_scores[v] = mean + exploration
        
        return max(ucb_scores, key=ucb_scores.get)
    
    def _thompson_sampling(self) -> str:
        """Thompson Sampling (Bayesian)"""
        samples = {}
        for v in self.variants:
            # Beta distribution
            alpha = self.successes[v] + 1
            beta = self.failures[v] + 1
            samples[v] = np.random.beta(alpha, beta)
        
        return max(samples, key=samples.get)
    
    def update(self, variant: str, success: bool):
        """Update after observing result"""
        self.total_trials[variant] += 1
        if success:
            self.successes[variant] += 1
        else:
            self.failures[variant] += 1
    
    def get_performance(self) -> Dict:
        """Get current performance"""
        results = {}
        for v in self.variants:
            if self.total_trials[v] > 0:
                rate = self.successes[v] / self.total_trials[v]
            else:
                rate = 0.0
            
            results[v] = {
                'trials': self.total_trials[v],
                'successes': self.successes[v],
                'rate': rate
            }
        
        return results
