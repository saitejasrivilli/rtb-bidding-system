"""
Auction Simulator for Real-Time Bidding
Implements second-price auction (Vickrey auction) with quality scores
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import heapq
from collections import defaultdict


@dataclass
class Bidder:
    """Represents a bidder in the auction"""
    bidder_id: str
    bid_amount: float
    ad_quality_score: float
    predicted_ctr: float
    advertiser_id: str
    creative_id: str


@dataclass
class AuctionResult:
    """Result of an auction"""
    winner_id: str
    winning_bid: float
    price_paid: float
    effective_bid: float
    runner_up_id: Optional[str]
    runner_up_bid: float
    all_bids: List[Tuple[str, float, float]]  # (bidder_id, bid, effective_bid)
    auction_id: str


class SecondPriceAuction:
    """
    Second-price auction (Vickrey auction) implementation
    Winner pays the second-highest price
    """
    
    def __init__(self, use_quality_score: bool = True, quality_weight: float = 1.0):
        """
        Args:
            use_quality_score: Whether to use quality scores in ranking
            quality_weight: Weight given to quality vs. bid amount
        """
        self.use_quality_score = use_quality_score
        self.quality_weight = quality_weight
        self.auction_history = []
    
    def run_auction(
        self,
        bidders: List[Bidder],
        auction_id: str,
        reserve_price: float = 0.0
    ) -> Optional[AuctionResult]:
        """
        Run a second-price auction
        
        Args:
            bidders: List of bidders participating
            auction_id: Unique auction identifier
            reserve_price: Minimum acceptable price
        
        Returns:
            AuctionResult if there's a winner, None otherwise
        """
        if len(bidders) == 0:
            return None
        
        # Calculate effective bids (bid * quality_score)
        effective_bids = []
        for bidder in bidders:
            if self.use_quality_score:
                effective_bid = bidder.bid_amount * (
                    bidder.ad_quality_score ** self.quality_weight
                )
            else:
                effective_bid = bidder.bid_amount
            
            effective_bids.append((bidder, effective_bid))
        
        # Sort by effective bid (descending)
        effective_bids.sort(key=lambda x: x[1], reverse=True)
        
        # Winner is highest effective bid
        winner, winning_effective_bid = effective_bids[0]
        
        # Check reserve price
        if winner.bid_amount < reserve_price:
            return None
        
        # Calculate price paid (second-price)
        if len(effective_bids) == 1:
            # Only one bidder, pays reserve price
            price_paid = reserve_price
            runner_up_id = None
            runner_up_bid = 0.0
        else:
            # Winner pays what they need to beat second place
            runner_up, runner_up_effective_bid = effective_bids[1]
            runner_up_id = runner_up.bidder_id
            runner_up_bid = runner_up.bid_amount
            
            # Price calculation with quality scores:
            # winner_bid * quality_winner = runner_up_bid * quality_runner_up
            # winner_bid = (runner_up_bid * quality_runner_up) / quality_winner
            if self.use_quality_score and winner.ad_quality_score > 0:
                price_paid = (
                    runner_up.bid_amount *
                    (runner_up.ad_quality_score ** self.quality_weight) /
                    (winner.ad_quality_score ** self.quality_weight)
                )
            else:
                price_paid = runner_up.bid_amount
            
            # Add small increment (e.g., $0.01)
            price_paid += 0.01
            
            # Ensure price doesn't exceed winner's bid
            price_paid = min(price_paid, winner.bid_amount)
            
            # Ensure price meets reserve
            price_paid = max(price_paid, reserve_price)
        
        result = AuctionResult(
            winner_id=winner.bidder_id,
            winning_bid=winner.bid_amount,
            price_paid=price_paid,
            effective_bid=winning_effective_bid,
            runner_up_id=runner_up_id,
            runner_up_bid=runner_up_bid,
            all_bids=[
                (b.bidder_id, b.bid_amount, eff_bid)
                for b, eff_bid in effective_bids
            ],
            auction_id=auction_id
        )
        
        self.auction_history.append(result)
        return result
    
    def get_statistics(self) -> Dict:
        """Get auction statistics"""
        if not self.auction_history:
            return {}
        
        win_rates = defaultdict(int)
        total_spent = defaultdict(float)
        
        for result in self.auction_history:
            win_rates[result.winner_id] += 1
            total_spent[result.winner_id] += result.price_paid
        
        total_auctions = len(self.auction_history)
        
        stats = {
            'total_auctions': total_auctions,
            'unique_winners': len(win_rates),
            'win_rates': {k: v / total_auctions for k, v in win_rates.items()},
            'total_spent': dict(total_spent),
            'avg_price_paid': np.mean([r.price_paid for r in self.auction_history]),
            'avg_winning_bid': np.mean([r.winning_bid for r in self.auction_history])
        }
        
        return stats


class GSPAuction:
    """
    Generalized Second Price (GSP) Auction
    Used for multiple ad slots (e.g., search ads)
    """
    
    def __init__(self, num_slots: int = 3, use_quality_score: bool = True):
        """
        Args:
            num_slots: Number of ad slots available
            use_quality_score: Whether to use quality scores
        """
        self.num_slots = num_slots
        self.use_quality_score = use_quality_score
        self.auction_history = []
    
    def run_auction(
        self,
        bidders: List[Bidder],
        slot_ctrs: List[float],  # CTR for each slot position
        auction_id: str
    ) -> List[AuctionResult]:
        """
        Run GSP auction for multiple slots
        
        Args:
            bidders: List of bidders
            slot_ctrs: Click-through rates for each slot position
            auction_id: Unique identifier
        
        Returns:
            List of AuctionResults, one per filled slot
        """
        if len(bidders) == 0:
            return []
        
        # Calculate effective bids
        effective_bids = []
        for bidder in bidders:
            if self.use_quality_score:
                effective_bid = bidder.bid_amount * bidder.ad_quality_score
            else:
                effective_bid = bidder.bid_amount
            
            effective_bids.append((bidder, effective_bid))
        
        # Sort by effective bid
        effective_bids.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate slots
        results = []
        num_filled_slots = min(self.num_slots, len(effective_bids))
        
        for slot_idx in range(num_filled_slots):
            winner, winning_effective_bid = effective_bids[slot_idx]
            
            # Calculate price (pays next highest bid)
            if slot_idx < len(effective_bids) - 1:
                next_bidder, next_effective_bid = effective_bids[slot_idx + 1]
                
                if self.use_quality_score and winner.ad_quality_score > 0:
                    price_paid = (
                        next_bidder.bid_amount * next_bidder.ad_quality_score /
                        winner.ad_quality_score
                    )
                else:
                    price_paid = next_bidder.bid_amount
                
                price_paid += 0.01
                price_paid = min(price_paid, winner.bid_amount)
                
                runner_up_id = next_bidder.bidder_id
                runner_up_bid = next_bidder.bid_amount
            else:
                price_paid = 0.01  # Minimum bid
                runner_up_id = None
                runner_up_bid = 0.0
            
            result = AuctionResult(
                winner_id=winner.bidder_id,
                winning_bid=winner.bid_amount,
                price_paid=price_paid * slot_ctrs[slot_idx],  # Adjust by slot CTR
                effective_bid=winning_effective_bid,
                runner_up_id=runner_up_id,
                runner_up_bid=runner_up_bid,
                all_bids=[(b.bidder_id, b.bid_amount, eff) for b, eff in effective_bids],
                auction_id=f"{auction_id}_slot_{slot_idx}"
            )
            
            results.append(result)
        
        self.auction_history.extend(results)
        return results


class VCGAuction:
    """
    Vickrey-Clarke-Groves (VCG) Auction
    Truthful mechanism for multiple items
    """
    
    def __init__(self, use_quality_score: bool = True):
        self.use_quality_score = use_quality_score
        self.auction_history = []
    
    def run_auction(
        self,
        bidders: List[Bidder],
        num_slots: int,
        auction_id: str
    ) -> List[AuctionResult]:
        """
        Run VCG auction
        Each winner pays their externality (harm to others)
        """
        if len(bidders) == 0:
            return []
        
        # Calculate values (effective bids)
        values = []
        for bidder in bidders:
            if self.use_quality_score:
                value = bidder.bid_amount * bidder.ad_quality_score
            else:
                value = bidder.bid_amount
            values.append((bidder, value))
        
        values.sort(key=lambda x: x[1], reverse=True)
        
        # Winners are top num_slots bidders
        winners = values[:min(num_slots, len(values))]
        
        # Calculate VCG prices
        results = []
        
        for i, (winner, winner_value) in enumerate(winners):
            # Calculate total value without this winner
            others_with_winner = sum(v for _, v in winners if v != winner_value)
            
            # Calculate total value if this slot went to next bidder
            if len(values) > num_slots:
                next_bidder = values[num_slots]
                others_without_winner = others_with_winner + next_bidder[1] - winner_value
            else:
                others_without_winner = others_with_winner - winner_value
            
            # VCG price is the externality
            vcg_price = max(0, others_without_winner - others_with_winner)
            
            # Convert back to bid amount
            if self.use_quality_score and winner.ad_quality_score > 0:
                price_paid = vcg_price / winner.ad_quality_score
            else:
                price_paid = vcg_price
            
            price_paid = min(price_paid, winner.bid_amount)
            
            result = AuctionResult(
                winner_id=winner.bidder_id,
                winning_bid=winner.bid_amount,
                price_paid=price_paid,
                effective_bid=winner_value,
                runner_up_id=None,
                runner_up_bid=0.0,
                all_bids=[(b.bidder_id, b.bid_amount, v) for b, v in values],
                auction_id=f"{auction_id}_vcg_{i}"
            )
            
            results.append(result)
        
        self.auction_history.extend(results)
        return results


class AuctionSimulator:
    """
    Complete auction simulator with multiple bidders and auction types
    """
    
    def __init__(
        self,
        auction_type: str = 'second_price',
        use_quality_score: bool = True
    ):
        """
        Args:
            auction_type: 'second_price', 'gsp', or 'vcg'
            use_quality_score: Whether to use ad quality scores
        """
        self.auction_type = auction_type
        
        if auction_type == 'second_price':
            self.auction = SecondPriceAuction(use_quality_score=use_quality_score)
        elif auction_type == 'gsp':
            self.auction = GSPAuction(use_quality_score=use_quality_score)
        elif auction_type == 'vcg':
            self.auction = VCGAuction(use_quality_score=use_quality_score)
        else:
            raise ValueError(f"Unknown auction type: {auction_type}")
        
        self.metrics = {
            'total_revenue': 0.0,
            'total_auctions': 0,
            'avg_bids_per_auction': 0.0,
            'fill_rate': 0.0
        }
    
    def simulate_auction(
        self,
        bidders: List[Bidder],
        auction_id: str,
        **kwargs
    ) -> Optional[AuctionResult]:
        """
        Simulate a single auction
        
        Returns:
            AuctionResult or None if no winner
        """
        self.metrics['total_auctions'] += 1
        
        if len(bidders) > 0:
            self.metrics['avg_bids_per_auction'] = (
                (self.metrics['avg_bids_per_auction'] * (self.metrics['total_auctions'] - 1) +
                 len(bidders)) / self.metrics['total_auctions']
            )
        
        if self.auction_type == 'second_price':
            result = self.auction.run_auction(bidders, auction_id, **kwargs)
            if result:
                self.metrics['total_revenue'] += result.price_paid
                self.metrics['fill_rate'] = (
                    len(self.auction.auction_history) / self.metrics['total_auctions']
                )
            return result
        else:
            results = self.auction.run_auction(bidders, auction_id, **kwargs)
            if results:
                self.metrics['total_revenue'] += sum(r.price_paid for r in results)
            return results[0] if results else None
    
    def get_metrics(self) -> Dict:
        """Get simulation metrics"""
        return {
            **self.metrics,
            'avg_revenue_per_auction': (
                self.metrics['total_revenue'] / self.metrics['total_auctions']
                if self.metrics['total_auctions'] > 0 else 0
            )
        }
    
    def analyze_bidder_performance(self, bidder_id: str) -> Dict:
        """Analyze performance of a specific bidder"""
        wins = 0
        total_spent = 0.0
        total_bids = 0
        bid_values = []
        
        for result in self.auction.auction_history:
            for bid_id, bid_amount, _ in result.all_bids:
                if bid_id == bidder_id:
                    total_bids += 1
                    bid_values.append(bid_amount)
                    
                    if result.winner_id == bidder_id:
                        wins += 1
                        total_spent += result.price_paid
        
        return {
            'bidder_id': bidder_id,
            'total_bids': total_bids,
            'wins': wins,
            'win_rate': wins / total_bids if total_bids > 0 else 0,
            'total_spent': total_spent,
            'avg_bid': np.mean(bid_values) if bid_values else 0,
            'avg_price_paid': total_spent / wins if wins > 0 else 0
        }
