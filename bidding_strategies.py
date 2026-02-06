"""
Bidding Strategies for Real-Time Bidding
Implements various bidding algorithms (CPC, CPM, CPA)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class BidStrategy(Enum):
    """Supported bidding strategies"""
    CPM = "cpm"  # Cost Per Mille (1000 impressions)
    CPC = "cpc"  # Cost Per Click
    CPA = "cpa"  # Cost Per Acquisition
    VCPM = "vcpm"  # Viewable CPM
    DYNAMIC = "dynamic"  # Dynamic bidding based on value


@dataclass
class BidRequest:
    """Information about a bid request"""
    request_id: str
    user_id: int
    ad_id: int
    context_id: int
    timestamp: float
    floor_price: float  # Minimum bid price
    ad_position: int
    device_type: str
    additional_features: Dict


@dataclass
class BidResponse:
    """Response to a bid request"""
    bid_price: float
    expected_ctr: float
    expected_value: float
    bidding_strategy: str
    pacing_multiplier: float
    should_bid: bool
    metadata: Dict


class BiddingEngine:
    """
    Core bidding engine that combines CTR prediction, budget pacing, and bidding strategy
    """
    
    def __init__(
        self,
        ctr_model,
        budget_pacer,
        strategy: BidStrategy = BidStrategy.CPC,
        advertiser_bid: float = 1.0,
        min_ctr_threshold: float = 0.001
    ):
        """
        Args:
            ctr_model: Trained CTR prediction model
            budget_pacer: Budget pacing algorithm
            strategy: Bidding strategy to use
            advertiser_bid: Base bid amount from advertiser
            min_ctr_threshold: Minimum CTR to consider bidding
        """
        self.ctr_model = ctr_model
        self.budget_pacer = budget_pacer
        self.strategy = strategy
        self.advertiser_bid = advertiser_bid
        self.min_ctr_threshold = min_ctr_threshold
    
    def compute_bid(
        self,
        bid_request: BidRequest,
        budget_state,
        predicted_ctr: Optional[float] = None
    ) -> BidResponse:
        """
        Compute bid for a given request
        
        Args:
            bid_request: The bid request
            budget_state: Current budget state
            predicted_ctr: Pre-computed CTR (optional)
        
        Returns:
            BidResponse with bid price and metadata
        """
        # Predict CTR if not provided
        if predicted_ctr is None:
            predicted_ctr = self._predict_ctr(bid_request)
        
        # Check if we should bid at all
        if predicted_ctr < self.min_ctr_threshold:
            return BidResponse(
                bid_price=0.0,
                expected_ctr=predicted_ctr,
                expected_value=0.0,
                bidding_strategy=self.strategy.value,
                pacing_multiplier=0.0,
                should_bid=False,
                metadata={'reason': 'low_ctr'}
            )
        
        # Calculate base expected value
        if self.strategy == BidStrategy.CPM:
            base_value = self.advertiser_bid  # Direct CPM bid
        elif self.strategy == BidStrategy.CPC:
            # Convert CPC to CPM: CPC * CTR * 1000
            base_value = self.advertiser_bid * predicted_ctr * 1000
        elif self.strategy == BidStrategy.CPA:
            # CPA with conversion rate estimation
            conversion_rate = self._estimate_conversion_rate(bid_request)
            base_value = self.advertiser_bid * predicted_ctr * conversion_rate * 1000
        elif self.strategy == BidStrategy.DYNAMIC:
            base_value = self._dynamic_valuation(bid_request, predicted_ctr)
        else:
            base_value = self.advertiser_bid * predicted_ctr * 1000
        
        # Apply budget pacing
        pacing_multiplier = self.budget_pacer.calculate_pacing_multiplier(budget_state)
        
        # Final bid
        final_bid = base_value * pacing_multiplier
        
        # Apply floor price
        if final_bid < bid_request.floor_price:
            return BidResponse(
                bid_price=0.0,
                expected_ctr=predicted_ctr,
                expected_value=base_value,
                bidding_strategy=self.strategy.value,
                pacing_multiplier=pacing_multiplier,
                should_bid=False,
                metadata={'reason': 'below_floor'}
            )
        
        # Check budget availability
        should_bid = self.budget_pacer.should_bid(budget_state, final_bid / 1000)
        
        if not should_bid:
            return BidResponse(
                bid_price=0.0,
                expected_ctr=predicted_ctr,
                expected_value=base_value,
                bidding_strategy=self.strategy.value,
                pacing_multiplier=pacing_multiplier,
                should_bid=False,
                metadata={'reason': 'budget_throttle'}
            )
        
        return BidResponse(
            bid_price=final_bid,
            expected_ctr=predicted_ctr,
            expected_value=base_value,
            bidding_strategy=self.strategy.value,
            pacing_multiplier=pacing_multiplier,
            should_bid=True,
            metadata={
                'base_value': base_value,
                'floor_price': bid_request.floor_price
            }
        )
    
    def _predict_ctr(self, bid_request: BidRequest) -> float:
        """Predict CTR for the bid request"""
        import torch
        
        features = {
            'user_id': torch.tensor([bid_request.user_id]),
            'ad_id': torch.tensor([bid_request.ad_id]),
            'context_id': torch.tensor([bid_request.context_id])
        }
        
        self.ctr_model.eval()
        with torch.no_grad():
            ctr = self.ctr_model(features)
        
        return float(ctr.item())
    
    def _estimate_conversion_rate(self, bid_request: BidRequest) -> float:
        """
        Estimate conversion rate
        In practice, this would use a separate model
        """
        # Simplified: assume 5% of clicks convert
        base_cvr = 0.05
        
        # Adjust based on device type
        if bid_request.device_type == 'mobile':
            return base_cvr * 0.8
        elif bid_request.device_type == 'desktop':
            return base_cvr * 1.2
        
        return base_cvr
    
    def _dynamic_valuation(self, bid_request: BidRequest, predicted_ctr: float) -> float:
        """
        Dynamic valuation based on multiple factors
        """
        base_value = self.advertiser_bid * predicted_ctr * 1000
        
        # Adjust for ad position
        position_multiplier = {
            1: 1.5,  # Premium position
            2: 1.2,
            3: 1.0,
            4: 0.8,
            5: 0.6
        }.get(bid_request.ad_position, 0.5)
        
        # Adjust for device type
        device_multiplier = {
            'desktop': 1.2,
            'mobile': 1.0,
            'tablet': 0.9
        }.get(bid_request.device_type, 1.0)
        
        return base_value * position_multiplier * device_multiplier


class TruthfulBidder:
    """
    Truthful bidding (bid your true value)
    Optimal for second-price auctions
    """
    
    def __init__(self, ctr_model, advertiser_value: float):
        self.ctr_model = ctr_model
        self.advertiser_value = advertiser_value  # Value per click
    
    def compute_bid(self, bid_request: BidRequest) -> float:
        """
        Compute truthful bid = expected value
        """
        predicted_ctr = self._predict_ctr(bid_request)
        
        # Expected value = CTR * value per click * 1000 (CPM)
        expected_value = predicted_ctr * self.advertiser_value * 1000
        
        return expected_value
    
    def _predict_ctr(self, bid_request: BidRequest) -> float:
        """Predict CTR"""
        import torch
        
        features = {
            'user_id': torch.tensor([bid_request.user_id]),
            'ad_id': torch.tensor([bid_request.ad_id]),
            'context_id': torch.tensor([bid_request.context_id])
        }
        
        self.ctr_model.eval()
        with torch.no_grad():
            ctr = self.ctr_model(features)
        
        return float(ctr.item())


class OptimalBidder:
    """
    Optimal bidding with win rate and profit optimization
    """
    
    def __init__(
        self,
        ctr_model,
        advertiser_value: float,
        target_win_rate: float = 0.3
    ):
        self.ctr_model = ctr_model
        self.advertiser_value = advertiser_value
        self.target_win_rate = target_win_rate
        
        # Learn bid landscape over time
        self.bid_history = []
        self.win_history = []
    
    def compute_bid(
        self,
        bid_request: BidRequest,
        market_price_distribution: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute optimal bid considering win rate and profit
        """
        predicted_ctr = self._predict_ctr(bid_request)
        expected_value = predicted_ctr * self.advertiser_value * 1000
        
        if market_price_distribution is None:
            # Without market info, bid conservatively
            return expected_value * 0.8
        
        # Find bid that achieves target win rate
        sorted_prices = np.sort(market_price_distribution)
        target_index = int(len(sorted_prices) * self.target_win_rate)
        market_price = sorted_prices[target_index]
        
        # Bid up to expected value, but consider market
        optimal_bid = min(expected_value * 0.9, market_price * 1.1)
        
        return optimal_bid
    
    def _predict_ctr(self, bid_request: BidRequest) -> float:
        """Predict CTR"""
        import torch
        
        features = {
            'user_id': torch.tensor([bid_request.user_id]),
            'ad_id': torch.tensor([bid_request.ad_id]),
            'context_id': torch.tensor([bid_request.context_id])
        }
        
        self.ctr_model.eval()
        with torch.no_grad():
            ctr = self.ctr_model(features)
        
        return float(ctr.item())
    
    def update_history(self, bid: float, won: bool):
        """Update bid history for learning"""
        self.bid_history.append(bid)
        self.win_history.append(won)
        
        # Keep only recent history
        if len(self.bid_history) > 10000:
            self.bid_history = self.bid_history[-10000:]
            self.win_history = self.win_history[-10000:]


class LinearBidder:
    """
    Linear bidding function: bid = base_bid + ctr_coefficient * predicted_ctr
    """
    
    def __init__(
        self,
        ctr_model,
        base_bid: float = 0.5,
        ctr_coefficient: float = 10.0
    ):
        self.ctr_model = ctr_model
        self.base_bid = base_bid
        self.ctr_coefficient = ctr_coefficient
    
    def compute_bid(self, bid_request: BidRequest) -> float:
        """Compute linear bid"""
        predicted_ctr = self._predict_ctr(bid_request)
        
        bid = self.base_bid + self.ctr_coefficient * predicted_ctr
        
        return max(0, bid)
    
    def _predict_ctr(self, bid_request: BidRequest) -> float:
        """Predict CTR"""
        import torch
        
        features = {
            'user_id': torch.tensor([bid_request.user_id]),
            'ad_id': torch.tensor([bid_request.ad_id]),
            'context_id': torch.tensor([bid_request.context_id])
        }
        
        self.ctr_model.eval()
        with torch.no_grad():
            ctr = self.ctr_model(features)
        
        return float(ctr.item())
    
    def optimize_parameters(
        self,
        historical_data: list,
        optimization_metric: str = 'profit'
    ):
        """
        Optimize base_bid and ctr_coefficient based on historical performance
        
        Args:
            historical_data: List of (bid_request, won, price_paid, clicked)
            optimization_metric: 'profit', 'clicks', or 'win_rate'
        """
        # This would use optimization techniques (gradient descent, etc.)
        # Simplified implementation
        
        total_profit = 0
        total_clicks = 0
        total_wins = 0
        
        for bid_request, won, price_paid, clicked in historical_data:
            predicted_ctr = self._predict_ctr(bid_request)
            bid = self.base_bid + self.ctr_coefficient * predicted_ctr
            
            if won:
                total_wins += 1
                if clicked:
                    total_clicks += 1
                    # Assume value per click is $1
                    total_profit += (1.0 - price_paid / 1000)
                else:
                    total_profit -= price_paid / 1000
        
        print(f"Current parameters: base_bid={self.base_bid}, ctr_coefficient={self.ctr_coefficient}")
        print(f"Performance: profit=${total_profit:.2f}, clicks={total_clicks}, wins={total_wins}")
        
        # Simple grid search (in practice, use more sophisticated optimization)
        best_profit = total_profit
        best_params = (self.base_bid, self.ctr_coefficient)
        
        for base in np.arange(0.1, 2.0, 0.2):
            for coef in np.arange(5.0, 20.0, 2.0):
                test_profit = self._evaluate_params(base, coef, historical_data)
                if test_profit > best_profit:
                    best_profit = test_profit
                    best_params = (base, coef)
        
        self.base_bid, self.ctr_coefficient = best_params
        print(f"Optimized parameters: base_bid={self.base_bid}, ctr_coefficient={self.ctr_coefficient}")
        print(f"Optimized profit: ${best_profit:.2f}")
    
    def _evaluate_params(self, base_bid: float, ctr_coef: float, historical_data: list) -> float:
        """Evaluate profit with given parameters"""
        total_profit = 0
        
        for bid_request, won, price_paid, clicked in historical_data:
            predicted_ctr = self._predict_ctr(bid_request)
            bid = base_bid + ctr_coef * predicted_ctr
            
            # Simulate if we would have won with this bid
            # (simplified: assume we win if our bid > price_paid)
            would_win = bid >= price_paid
            
            if would_win:
                if clicked:
                    total_profit += (1.0 - price_paid / 1000)
                else:
                    total_profit -= price_paid / 1000
        
        return total_profit
