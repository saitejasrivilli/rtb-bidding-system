"""
Budget Pacing Algorithm for Real-Time Bidding
Implements sophisticated pacing strategies to spend budget evenly over time
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import math


@dataclass
class BudgetState:
    """Current state of advertiser budget"""
    total_budget: float
    remaining_budget: float
    time_start: datetime
    time_end: datetime
    current_time: datetime
    total_spent: float
    total_impressions: int
    total_clicks: int
    current_spend_rate: float  # $ per hour
    target_spend_rate: float  # $ per hour


class BudgetPacer:
    """
    Budget pacing algorithm to control spending rate
    Implements multiple pacing strategies
    """
    
    def __init__(self, strategy: str = 'proportional'):
        """
        Args:
            strategy: Pacing strategy ('proportional', 'adaptive', 'exponential', 'threshold')
        """
        self.strategy = strategy
        self.alpha = 0.1  # Smoothing factor for adaptive pacing
    
    def calculate_pacing_multiplier(self, budget_state: BudgetState) -> float:
        """
        Calculate pacing multiplier to adjust bid prices
        
        Returns:
            Multiplier between 0 and 2 (typically)
        """
        if self.strategy == 'proportional':
            return self._proportional_pacing(budget_state)
        elif self.strategy == 'adaptive':
            return self._adaptive_pacing(budget_state)
        elif self.strategy == 'exponential':
            return self._exponential_pacing(budget_state)
        elif self.strategy == 'threshold':
            return self._threshold_pacing(budget_state)
        else:
            return 1.0
    
    def _proportional_pacing(self, state: BudgetState) -> float:
        """
        Proportional pacing: multiply bids based on budget vs. time remaining ratio
        """
        time_elapsed = (state.current_time - state.time_start).total_seconds()
        time_total = (state.time_end - state.time_start).total_seconds()
        time_remaining = time_total - time_elapsed
        
        if time_remaining <= 0:
            return 0.0  # Campaign ended
        
        # Expected position
        time_fraction = time_elapsed / time_total
        budget_fraction = state.total_spent / state.total_budget
        
        # If we're ahead of schedule (spent more than time elapsed), reduce bids
        # If we're behind schedule (spent less than time elapsed), increase bids
        if budget_fraction < 0.01:  # Just started
            return 1.0
        
        pacing_ratio = (1 - budget_fraction) / (time_remaining / time_total)
        
        # Clip to reasonable range
        return np.clip(pacing_ratio, 0.1, 2.0)
    
    def _adaptive_pacing(self, state: BudgetState) -> float:
        """
        Adaptive pacing: adjust based on current spend rate vs. target rate
        Uses exponential smoothing
        """
        if state.target_spend_rate == 0:
            return 0.0
        
        # Calculate ratio of current to target spend rate
        rate_ratio = state.current_spend_rate / state.target_spend_rate
        
        # Apply exponential smoothing
        # If spending too fast (rate_ratio > 1), reduce multiplier
        # If spending too slow (rate_ratio < 1), increase multiplier
        multiplier = math.exp(-self.alpha * (rate_ratio - 1))
        
        return np.clip(multiplier, 0.1, 2.0)
    
    def _exponential_pacing(self, state: BudgetState) -> float:
        """
        Exponential pacing: more aggressive adjustment based on budget deviation
        """
        time_elapsed = (state.current_time - state.time_start).total_seconds()
        time_total = (state.time_end - state.time_start).total_seconds()
        
        time_fraction = time_elapsed / time_total
        budget_fraction = state.total_spent / state.total_budget
        
        # Deviation from ideal pacing line
        deviation = budget_fraction - time_fraction
        
        # Exponential adjustment
        multiplier = math.exp(-2 * deviation)
        
        return np.clip(multiplier, 0.1, 2.0)
    
    def _threshold_pacing(self, state: BudgetState, threshold: float = 0.1) -> float:
        """
        Threshold pacing: aggressive changes only when deviation exceeds threshold
        """
        time_elapsed = (state.current_time - state.time_start).total_seconds()
        time_total = (state.time_end - state.time_start).total_seconds()
        
        time_fraction = time_elapsed / time_total
        budget_fraction = state.total_spent / state.total_budget
        
        deviation = budget_fraction - time_fraction
        
        if abs(deviation) < threshold:
            return 1.0  # Within acceptable range
        elif deviation > threshold:
            return 0.5  # Spending too fast, slow down
        else:
            return 1.5  # Spending too slow, speed up
    
    def should_bid(self, budget_state: BudgetState, estimated_cost: float) -> bool:
        """
        Decide whether to bid at all based on budget state
        
        Args:
            budget_state: Current budget state
            estimated_cost: Expected cost if we win
        
        Returns:
            True if we should participate in auction
        """
        # Don't bid if budget exhausted
        if budget_state.remaining_budget < estimated_cost:
            return False
        
        # Don't bid if campaign ended
        if budget_state.current_time >= budget_state.time_end:
            return False
        
        # Calculate how much budget we need per remaining hour
        time_remaining_hours = (
            budget_state.time_end - budget_state.current_time
        ).total_seconds() / 3600
        
        if time_remaining_hours <= 0:
            return False
        
        required_rate = budget_state.remaining_budget / time_remaining_hours
        
        # If current spend rate is way above required rate, throttle
        if budget_state.current_spend_rate > required_rate * 2:
            # Probabilistic throttling
            throttle_prob = required_rate / budget_state.current_spend_rate
            return np.random.random() < throttle_prob
        
        return True


class ThrottlingPacer:
    """
    Throttling-based pacing: randomly skip auctions to control spend rate
    Simpler alternative to bid adjustment
    """
    
    def __init__(self):
        self.participation_history = []
    
    def calculate_participation_rate(self, budget_state: BudgetState) -> float:
        """
        Calculate what fraction of auctions we should participate in
        
        Returns:
            Participation rate between 0 and 1
        """
        time_elapsed = (budget_state.current_time - budget_state.time_start).total_seconds()
        time_total = (budget_state.time_end - budget_state.time_start).total_seconds()
        time_remaining = time_total - time_elapsed
        
        if time_remaining <= 0 or budget_state.remaining_budget <= 0:
            return 0.0
        
        time_fraction = time_elapsed / time_total
        budget_fraction = state.total_spent / budget_state.total_budget
        
        if time_fraction == 0:
            return 1.0
        
        # Ideal participation rate
        ideal_rate = (1 - budget_fraction) / (1 - time_fraction)
        
        return np.clip(ideal_rate, 0.0, 1.0)
    
    def should_participate(self, budget_state: BudgetState) -> bool:
        """
        Decide whether to participate in this auction
        """
        participation_rate = self.calculate_participation_rate(budget_state)
        return np.random.random() < participation_rate


class BudgetAllocator:
    """
    Allocate budget across multiple campaigns/ad groups
    """
    
    def __init__(self, total_budget: float):
        self.total_budget = total_budget
        self.allocations = {}
    
    def allocate_proportional(self, campaign_priorities: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate budget proportionally based on campaign priorities
        
        Args:
            campaign_priorities: Dictionary of campaign_id -> priority score
        
        Returns:
            Dictionary of campaign_id -> allocated budget
        """
        total_priority = sum(campaign_priorities.values())
        
        allocations = {}
        for campaign_id, priority in campaign_priorities.items():
            allocations[campaign_id] = (priority / total_priority) * self.total_budget
        
        self.allocations = allocations
        return allocations
    
    def allocate_performance_based(
        self,
        campaign_performance: Dict[str, Dict[str, float]],
        metric: str = 'roi'
    ) -> Dict[str, float]:
        """
        Allocate budget based on past performance
        
        Args:
            campaign_performance: Dict of campaign_id -> performance metrics
            metric: Metric to optimize ('roi', 'ctr', 'conversions')
        
        Returns:
            Dictionary of campaign_id -> allocated budget
        """
        # Calculate performance scores
        scores = {}
        for campaign_id, perf in campaign_performance.items():
            if metric in perf:
                scores[campaign_id] = perf[metric]
            else:
                scores[campaign_id] = 0.0
        
        # Softmax allocation (gives more to better performers)
        exp_scores = {k: math.exp(v) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        
        allocations = {}
        for campaign_id, exp_score in exp_scores.items():
            allocations[campaign_id] = (exp_score / total_exp) * self.total_budget
        
        self.allocations = allocations
        return allocations
    
    def reallocate_dynamic(
        self,
        current_allocations: Dict[str, float],
        current_spending: Dict[str, float],
        current_performance: Dict[str, float],
        reallocation_threshold: float = 0.2
    ) -> Dict[str, float]:
        """
        Dynamically reallocate budget based on real-time performance
        
        Args:
            current_allocations: Current budget allocations
            current_spending: How much each campaign has spent
            current_performance: Current ROI or other metric
            reallocation_threshold: Minimum performance difference to trigger reallocation
        
        Returns:
            Updated allocations
        """
        # Calculate remaining budgets
        remaining = {
            cid: current_allocations[cid] - current_spending.get(cid, 0)
            for cid in current_allocations
        }
        
        total_remaining = sum(remaining.values())
        
        if total_remaining <= 0:
            return current_allocations
        
        # Identify over and under performers
        avg_performance = np.mean(list(current_performance.values()))
        
        new_allocations = current_allocations.copy()
        
        for campaign_id in current_allocations:
            perf = current_performance.get(campaign_id, 0)
            spent = current_spending.get(campaign_id, 0)
            
            # If significantly outperforming, allocate more
            if perf > avg_performance * (1 + reallocation_threshold):
                bonus = remaining[campaign_id] * 0.2
                new_allocations[campaign_id] += bonus
            
            # If significantly underperforming, reduce allocation
            elif perf < avg_performance * (1 - reallocation_threshold):
                penalty = remaining[campaign_id] * 0.2
                new_allocations[campaign_id] -= penalty
        
        # Ensure non-negative and sum to total
        new_allocations = {k: max(0, v) for k, v in new_allocations.items()}
        
        return new_allocations
