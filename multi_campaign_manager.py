"""
Multi-Campaign Manager
Manages multiple advertising campaigns with budget allocation and optimization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd


@dataclass
class Campaign:
    """Single advertising campaign"""
    campaign_id: str
    name: str
    total_budget: float
    daily_budget: float
    start_date: datetime
    end_date: datetime
    bidding_strategy: str  # 'cpc', 'cpm', 'cpa'
    base_bid: float
    target_ctr: Optional[float] = None
    target_roi: Optional[float] = None
    priority: int = 1
    
    # Performance tracking
    total_spent: float = 0.0
    total_impressions: int = 0
    total_clicks: int = 0
    total_conversions: int = 0
    total_revenue: float = 0.0
    
    # Status
    is_active: bool = True
    pause_reason: Optional[str] = None
    
    def get_metrics(self) -> Dict:
        """Get campaign performance metrics"""
        ctr = self.total_clicks / max(self.total_impressions, 1)
        cvr = self.total_conversions / max(self.total_clicks, 1)
        cpc = self.total_spent / max(self.total_clicks, 1)
        cpm = (self.total_spent / max(self.total_impressions, 1)) * 1000
        cpa = self.total_spent / max(self.total_conversions, 1)
        roi = (self.total_revenue - self.total_spent) / max(self.total_spent, 1)
        
        return {
            'campaign_id': self.campaign_id,
            'total_spent': self.total_spent,
            'total_impressions': self.total_impressions,
            'total_clicks': self.total_clicks,
            'total_conversions': self.total_conversions,
            'total_revenue': self.total_revenue,
            'ctr': ctr,
            'cvr': cvr,
            'cpc': cpc,
            'cpm': cpm,
            'cpa': cpa,
            'roi': roi,
            'budget_utilization': self.total_spent / self.total_budget,
            'is_active': self.is_active
        }


class MultiCampaignManager:
    """
    Manages multiple campaigns with intelligent budget allocation
    """
    
    def __init__(self, total_budget: float):
        """
        Args:
            total_budget: Total budget across all campaigns
        """
        self.total_budget = total_budget
        self.campaigns: Dict[str, Campaign] = {}
        self.budget_allocations: Dict[str, float] = {}
        
        self.history = []
    
    def add_campaign(self, campaign: Campaign):
        """Add a new campaign"""
        self.campaigns[campaign.campaign_id] = campaign
        print(f"✅ Added campaign: {campaign.name} ({campaign.campaign_id})")
    
    def remove_campaign(self, campaign_id: str):
        """Remove a campaign"""
        if campaign_id in self.campaigns:
            del self.campaigns[campaign_id]
            if campaign_id in self.budget_allocations:
                del self.budget_allocations[campaign_id]
            print(f"❌ Removed campaign: {campaign_id}")
    
    def pause_campaign(self, campaign_id: str, reason: str = "Manual pause"):
        """Pause a campaign"""
        if campaign_id in self.campaigns:
            self.campaigns[campaign_id].is_active = False
            self.campaigns[campaign_id].pause_reason = reason
            print(f"⏸️  Paused campaign: {campaign_id} - {reason}")
    
    def resume_campaign(self, campaign_id: str):
        """Resume a paused campaign"""
        if campaign_id in self.campaigns:
            self.campaigns[campaign_id].is_active = True
            self.campaigns[campaign_id].pause_reason = None
            print(f"▶️  Resumed campaign: {campaign_id}")
    
    def allocate_budget_proportional(self) -> Dict[str, float]:
        """
        Allocate budget proportionally based on campaign priority and daily budget
        """
        active_campaigns = {
            cid: c for cid, c in self.campaigns.items()
            if c.is_active
        }
        
        if not active_campaigns:
            return {}
        
        # Calculate allocation weights
        weights = {}
        for cid, campaign in active_campaigns.items():
            # Weight by priority and daily budget
            weights[cid] = campaign.priority * campaign.daily_budget
        
        total_weight = sum(weights.values())
        
        # Allocate budget
        allocations = {}
        for cid, weight in weights.items():
            allocation = (weight / total_weight) * self.total_budget
            # Don't exceed campaign's remaining budget
            remaining = self.campaigns[cid].total_budget - self.campaigns[cid].total_spent
            allocations[cid] = min(allocation, remaining)
        
        self.budget_allocations = allocations
        return allocations
    
    def allocate_budget_performance_based(
        self,
        metric: str = 'roi',
        lookback_window: int = 24  # hours
    ) -> Dict[str, float]:
        """
        Allocate budget based on recent performance
        
        Args:
            metric: Metric to optimize ('roi', 'ctr', 'conversions')
            lookback_window: Hours of history to consider
        """
        active_campaigns = {
            cid: c for cid, c in self.campaigns.items()
            if c.is_active
        }
        
        if not active_campaigns:
            return {}
        
        # Calculate performance scores
        scores = {}
        for cid, campaign in active_campaigns.items():
            metrics = campaign.get_metrics()
            
            if metric == 'roi':
                score = metrics['roi']
            elif metric == 'ctr':
                score = metrics['ctr']
            elif metric == 'conversions':
                score = campaign.total_conversions
            else:
                score = metrics['roi']
            
            # Apply softmax to ensure positive scores
            scores[cid] = max(0.01, score)
        
        # Softmax allocation (more to better performers)
        import math
        exp_scores = {cid: math.exp(score) for cid, score in scores.items()}
        total_exp = sum(exp_scores.values())
        
        allocations = {}
        for cid, exp_score in exp_scores.items():
            allocation = (exp_score / total_exp) * self.total_budget
            remaining = self.campaigns[cid].total_budget - self.campaigns[cid].total_spent
            allocations[cid] = min(allocation, remaining)
        
        self.budget_allocations = allocations
        return allocations
    
    def allocate_budget_mab(self, explore_rate: float = 0.1) -> Dict[str, float]:
        """
        Multi-Armed Bandit budget allocation
        
        Args:
            explore_rate: Fraction of budget for exploration
        """
        active_campaigns = list(self.campaigns.values())
        active_campaigns = [c for c in active_campaigns if c.is_active]
        
        if not active_campaigns:
            return {}
        
        # Exploration budget
        explore_budget = self.total_budget * explore_rate
        explore_per_campaign = explore_budget / len(active_campaigns)
        
        # Exploitation budget
        exploit_budget = self.total_budget * (1 - explore_rate)
        
        # Calculate Thompson Sampling scores
        allocations = {}
        scores = []
        
        for campaign in active_campaigns:
            # Beta distribution parameters (Bayesian)
            alpha = campaign.total_conversions + 1
            beta = (campaign.total_clicks - campaign.total_conversions) + 1
            
            # Sample from distribution
            score = np.random.beta(alpha, beta)
            scores.append((campaign.campaign_id, score))
            
            # Give everyone exploration budget
            allocations[campaign.campaign_id] = explore_per_campaign
        
        # Sort by score for exploitation
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate exploitation budget proportionally to scores
        total_score = sum(s for _, s in scores)
        for cid, score in scores:
            exploit_allocation = (score / total_score) * exploit_budget
            allocations[cid] += exploit_allocation
            
            # Don't exceed remaining budget
            remaining = self.campaigns[cid].total_budget - self.campaigns[cid].total_spent
            allocations[cid] = min(allocations[cid], remaining)
        
        self.budget_allocations = allocations
        return allocations
    
    def optimize_bids(self, method: str = 'performance'):
        """
        Optimize bids for all campaigns
        
        Args:
            method: 'performance' or 'competitive'
        """
        for campaign in self.campaigns.values():
            if not campaign.is_active:
                continue
            
            metrics = campaign.get_metrics()
            
            if method == 'performance':
                # Increase bid if performing well
                if metrics['roi'] > 1.0 and metrics['ctr'] > campaign.target_ctr:
                    campaign.base_bid *= 1.1
                elif metrics['roi'] < 0.5:
                    campaign.base_bid *= 0.9
            
            elif method == 'competitive':
                # Adjust based on win rate (would need auction data)
                # Simplified version
                if metrics['budget_utilization'] < 0.8:
                    campaign.base_bid *= 1.05
            
            # Ensure bid stays within reasonable bounds
            campaign.base_bid = np.clip(campaign.base_bid, 0.1, 10.0)
    
    def rebalance_budgets(self, threshold: float = 0.2):
        """
        Dynamically rebalance budgets based on performance
        
        Args:
            threshold: Minimum performance difference to trigger rebalance
        """
        if len(self.campaigns) < 2:
            return
        
        # Calculate average ROI
        active_campaigns = [c for c in self.campaigns.values() if c.is_active]
        avg_roi = np.mean([c.get_metrics()['roi'] for c in active_campaigns])
        
        # Identify over and under performers
        reallocation_pool = 0.0
        winners = []
        
        for campaign in active_campaigns:
            metrics = campaign.get_metrics()
            roi_diff = (metrics['roi'] - avg_roi) / max(avg_roi, 0.01)
            
            if roi_diff < -threshold:
                # Underperformer: reduce budget
                if campaign.campaign_id in self.budget_allocations:
                    reduction = self.budget_allocations[campaign.campaign_id] * 0.2
                    self.budget_allocations[campaign.campaign_id] -= reduction
                    reallocation_pool += reduction
            
            elif roi_diff > threshold:
                # Overperformer: candidate for more budget
                winners.append(campaign.campaign_id)
        
        # Distribute reallocation pool to winners
        if winners and reallocation_pool > 0:
            bonus_per_winner = reallocation_pool / len(winners)
            for winner_id in winners:
                if winner_id in self.budget_allocations:
                    self.budget_allocations[winner_id] += bonus_per_winner
        
        print(f"🔄 Rebalanced budgets. Reallocated: ${reallocation_pool:.2f}")
    
    def get_dashboard_data(self) -> pd.DataFrame:
        """Get campaign performance data for dashboard"""
        data = []
        for campaign in self.campaigns.values():
            metrics = campaign.get_metrics()
            metrics['name'] = campaign.name
            metrics['allocated_budget'] = self.budget_allocations.get(campaign.campaign_id, 0.0)
            data.append(metrics)
        
        return pd.DataFrame(data)
    
    def generate_report(self) -> str:
        """Generate a text report of all campaigns"""
        report = []
        report.append("=" * 80)
        report.append("MULTI-CAMPAIGN PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Budget: ${self.total_budget:.2f}")
        report.append(f"Active Campaigns: {sum(1 for c in self.campaigns.values() if c.is_active)}")
        report.append("")
        
        # Campaign details
        for campaign in self.campaigns.values():
            report.append("-" * 80)
            report.append(f"Campaign: {campaign.name} ({campaign.campaign_id})")
            report.append(f"Status: {'🟢 Active' if campaign.is_active else '🔴 Paused'}")
            
            metrics = campaign.get_metrics()
            
            report.append(f"  Budget:")
            report.append(f"    Allocated: ${self.budget_allocations.get(campaign.campaign_id, 0.0):.2f}")
            report.append(f"    Spent: ${metrics['total_spent']:.2f}")
            report.append(f"    Utilization: {metrics['budget_utilization']*100:.1f}%")
            
            report.append(f"  Performance:")
            report.append(f"    Impressions: {metrics['total_impressions']:,}")
            report.append(f"    Clicks: {metrics['total_clicks']:,}")
            report.append(f"    CTR: {metrics['ctr']*100:.2f}%")
            report.append(f"    CPM: ${metrics['cpm']:.2f}")
            report.append(f"    CPC: ${metrics['cpc']:.2f}")
            report.append(f"    Revenue: ${metrics['total_revenue']:.2f}")
            report.append(f"    ROI: {metrics['roi']*100:.1f}%")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of using MultiCampaignManager"""
    
    print("\n" + "="*80)
    print("MULTI-CAMPAIGN MANAGER DEMO")
    print("="*80)
    
    # Create manager
    manager = MultiCampaignManager(total_budget=10000.0)
    
    # Add campaigns
    campaigns = [
        Campaign(
            campaign_id="CAMP_001",
            name="Summer Sale - Electronics",
            total_budget=5000.0,
            daily_budget=500.0,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            bidding_strategy="cpc",
            base_bid=1.5,
            target_ctr=0.02,
            priority=2
        ),
        Campaign(
            campaign_id="CAMP_002",
            name="Back to School - Apparel",
            total_budget=3000.0,
            daily_budget=300.0,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            bidding_strategy="cpm",
            base_bid=2.0,
            target_ctr=0.015,
            priority=1
        ),
        Campaign(
            campaign_id="CAMP_003",
            name="Holiday Promo - Home Goods",
            total_budget=2000.0,
            daily_budget=200.0,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            bidding_strategy="cpa",
            base_bid=5.0,
            target_roi=1.5,
            priority=1
        )
    ]
    
    for campaign in campaigns:
        manager.add_campaign(campaign)
    
    # Simulate some performance
    campaigns[0].total_spent = 1200.0
    campaigns[0].total_impressions = 50000
    campaigns[0].total_clicks = 1200
    campaigns[0].total_conversions = 60
    campaigns[0].total_revenue = 1800.0
    
    campaigns[1].total_spent = 800.0
    campaigns[1].total_impressions = 40000
    campaigns[1].total_clicks = 600
    campaigns[1].total_conversions = 20
    campaigns[1].total_revenue = 600.0
    
    campaigns[2].total_spent = 500.0
    campaigns[2].total_impressions = 10000
    campaigns[2].total_clicks = 200
    campaigns[2].total_conversions = 15
    campaigns[2].total_revenue = 900.0
    
    print("\n📊 Proportional Budget Allocation:")
    allocations = manager.allocate_budget_proportional()
    for cid, amount in allocations.items():
        print(f"  {cid}: ${amount:.2f}")
    
    print("\n📈 Performance-Based Allocation:")
    allocations = manager.allocate_budget_performance_based(metric='roi')
    for cid, amount in allocations.items():
        print(f"  {cid}: ${amount:.2f}")
    
    print("\n🎰 Multi-Armed Bandit Allocation:")
    allocations = manager.allocate_budget_mab(explore_rate=0.2)
    for cid, amount in allocations.items():
        print(f"  {cid}: ${amount:.2f}")
    
    # Optimize bids
    print("\n🎯 Optimizing Bids...")
    manager.optimize_bids(method='performance')
    for campaign in campaigns:
        print(f"  {campaign.campaign_id}: ${campaign.base_bid:.2f}")
    
    # Rebalance
    print("\n⚖️  Rebalancing Budgets...")
    manager.rebalance_budgets(threshold=0.3)
    
    # Generate report
    print("\n" + manager.generate_report())


if __name__ == "__main__":
    example_usage()
