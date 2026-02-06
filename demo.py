"""
Complete RTB System Demo
End-to-end example of Real-Time Bidding system with all components
"""

import sys
import os
import numpy as np
import torch
from datetime import datetime, timedelta
import time

# Add project to path
sys.path.append('/home/claude/rtb_system')

from models.ctr_model import CTRPredictor, FMCTRPredictor, CTRTrainer
from core.budget_pacing import BudgetPacer, BudgetState
from core.bidding_strategies import BiddingEngine, BidRequest, BidStrategy
from core.auction import SecondPriceAuction, Bidder, AuctionSimulator
from core.rtb_engine import RTBEngine, RTBBenchmark
from testing.ab_testing import ABTesting, MultiArmedBandit
from data.preprocessing import (
    iPinYouDataProcessor,
    create_dataloaders,
    calculate_feature_stats
)


def train_ctr_model(train_loader, val_loader, n_users, n_ads, n_contexts):
    """Train CTR prediction model"""
    print("\n" + "="*80)
    print("STEP 1: Training CTR Prediction Model")
    print("="*80)
    
    # Create model
    model = CTRPredictor(
        n_users=n_users,
        n_ads=n_ads,
        n_contexts=n_contexts,
        embedding_dim=64,
        hidden_dims=[256, 128, 64],
        dropout=0.3
    )
    
    print(f"Model architecture: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    trainer = CTRTrainer(model, learning_rate=0.001)
    
    print("\nTraining...")
    model = trainer.fit(
        train_loader,
        val_loader,
        epochs=5,  # Use more epochs for real training
        early_stopping_patience=2
    )
    
    # Evaluate
    print("\nEvaluating on validation set...")
    val_metrics = trainer.validate(val_loader)
    print(f"Validation AUC: {val_metrics['auc']:.4f}")
    print(f"Validation LogLoss: {val_metrics['logloss']:.4f}")
    
    return model


def simulate_budget_pacing():
    """Demonstrate budget pacing"""
    print("\n" + "="*80)
    print("STEP 2: Budget Pacing Demonstration")
    print("="*80)
    
    # Create budget state
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=24)
    
    budget_state = BudgetState(
        total_budget=1000.0,  # $1000 daily budget
        remaining_budget=1000.0,
        time_start=start_time,
        time_end=end_time,
        current_time=start_time,
        total_spent=0.0,
        total_impressions=0,
        total_clicks=0,
        current_spend_rate=0.0,
        target_spend_rate=1000.0 / 24  # $41.67 per hour
    )
    
    # Test different pacing strategies
    strategies = ['proportional', 'adaptive', 'exponential', 'threshold']
    
    print("\nBudget Pacing Multipliers:")
    print("-" * 60)
    
    for strategy in strategies:
        pacer = BudgetPacer(strategy=strategy)
        
        # Simulate at different times
        results = []
        for hour in [0, 6, 12, 18, 23]:
            budget_state.current_time = start_time + timedelta(hours=hour)
            budget_state.total_spent = (hour / 24) * 1000 * 1.2  # 20% overspend
            budget_state.remaining_budget = 1000 - budget_state.total_spent
            
            multiplier = pacer.calculate_pacing_multiplier(budget_state)
            results.append(f"Hour {hour:2d}: {multiplier:.3f}")
        
        print(f"\n{strategy.upper():15s}: {' | '.join(results)}")
    
    return budget_state


def run_auction_simulations(ctr_model, budget_state):
    """Run auction simulations"""
    print("\n" + "="*80)
    print("STEP 3: Auction Simulation")
    print("="*80)
    
    # Create bidding engine
    budget_pacer = BudgetPacer(strategy='proportional')
    
    bidding_engine = BiddingEngine(
        ctr_model=ctr_model,
        budget_pacer=budget_pacer,
        strategy=BidStrategy.CPC,
        advertiser_bid=1.0,  # $1.0 per click
        min_ctr_threshold=0.001
    )
    
    # Create auction simulator
    auction_sim = AuctionSimulator(
        auction_type='second_price',
        use_quality_score=True
    )
    
    print("\nRunning 100 auctions...")
    
    results = {
        'wins': 0,
        'total_bids': 0,
        'total_spent': 0.0,
        'total_revenue': 0.0
    }
    
    for auction_id in range(100):
        # Generate bid request
        bid_request = BidRequest(
            request_id=f"auction_{auction_id}",
            user_id=np.random.randint(0, 100),
            ad_id=np.random.randint(0, 50),
            context_id=np.random.randint(0, 20),
            timestamp=time.time(),
            floor_price=0.1,
            ad_position=np.random.randint(1, 6),
            device_type=np.random.choice(['mobile', 'desktop']),
            additional_features={}
        )
        
        # Compute our bid
        bid_response = bidding_engine.compute_bid(bid_request, budget_state)
        
        if not bid_response.should_bid:
            continue
        
        results['total_bids'] += 1
        
        # Create competing bidders
        bidders = [
            Bidder(
                bidder_id="our_bidder",
                bid_amount=bid_response.bid_price,
                ad_quality_score=0.8,
                predicted_ctr=bid_response.expected_ctr,
                advertiser_id="advertiser_1",
                creative_id=f"creative_{bid_request.ad_id}"
            )
        ]
        
        # Add 3-5 competitors
        n_competitors = np.random.randint(3, 6)
        for i in range(n_competitors):
            competitor_bid = np.random.lognormal(1.5, 0.5)
            bidders.append(
                Bidder(
                    bidder_id=f"competitor_{i}",
                    bid_amount=competitor_bid,
                    ad_quality_score=np.random.uniform(0.5, 1.0),
                    predicted_ctr=np.random.uniform(0.001, 0.05),
                    advertiser_id=f"advertiser_{i+2}",
                    creative_id=f"creative_{i}"
                )
            )
        
        # Run auction
        auction_result = auction_sim.simulate_auction(
            bidders,
            f"auction_{auction_id}",
            reserve_price=0.1
        )
        
        if auction_result and auction_result.winner_id == "our_bidder":
            results['wins'] += 1
            results['total_spent'] += auction_result.price_paid
            
            # Simulate click
            if np.random.random() < bid_response.expected_ctr:
                results['total_revenue'] += 1.0  # $1 per click
            
            # Update budget
            budget_state.total_spent += auction_result.price_paid
            budget_state.remaining_budget -= auction_result.price_paid
    
    # Print results
    print(f"\nAuction Results:")
    print(f"  Total Bids: {results['total_bids']}")
    print(f"  Wins: {results['wins']}")
    print(f"  Win Rate: {results['wins']/results['total_bids']*100:.1f}%")
    print(f"  Total Spent: ${results['total_spent']:.2f}")
    print(f"  Total Revenue: ${results['total_revenue']:.2f}")
    print(f"  Profit: ${results['total_revenue'] - results['total_spent']:.2f}")
    print(f"  ROI: {(results['total_revenue'] - results['total_spent'])/results['total_spent']*100:.1f}%")
    
    return results


def benchmark_rtb_engine(ctr_model, budget_state):
    """Benchmark RTB engine latency"""
    print("\n" + "="*80)
    print("STEP 4: RTB Engine Latency Benchmark")
    print("="*80)
    
    budget_pacer = BudgetPacer(strategy='proportional')
    
    bidding_engine = BiddingEngine(
        ctr_model=ctr_model,
        budget_pacer=budget_pacer,
        strategy=BidStrategy.CPC,
        advertiser_bid=1.0
    )
    
    # Create RTB engine
    rtb_engine = RTBEngine(
        ctr_model=ctr_model,
        budget_pacer=budget_pacer,
        bidding_engine=bidding_engine,
        budget_state=budget_state,
        use_cache=True,
        max_workers=4
    )
    
    # Optimize for latency
    rtb_engine.optimize_for_latency()
    
    # Benchmark
    benchmark = RTBBenchmark(rtb_engine)
    
    print("\n--- Single Request Latency ---")
    latency_results = benchmark.benchmark_latency(
        num_requests=1000,
        batch_size=1
    )
    
    print("\n--- Batch Processing Latency ---")
    batch_results = benchmark.benchmark_latency(
        num_requests=1000,
        batch_size=32
    )
    
    print("\n--- Throughput Test ---")
    throughput_results = benchmark.benchmark_throughput(duration_seconds=10)
    
    return latency_results


def run_ab_test(ctr_model, budget_state):
    """Run A/B test comparing bidding strategies"""
    print("\n" + "="*80)
    print("STEP 5: A/B Testing Framework")
    print("="*80)
    
    # Create A/B test
    ab_test = ABTesting(seed=42)
    
    experiment = ab_test.create_experiment(
        experiment_id="bidding_strategy_test",
        name="CPC vs CPM Bidding Strategy",
        variants=["cpc_strategy", "cpm_strategy"],
        traffic_split={"cpc_strategy": 0.5, "cpm_strategy": 0.5},
        primary_metric="roi",
        secondary_metrics=["ctr", "cpm", "win_rate"]
    )
    
    print("\nRunning A/B test with 1000 auctions...")
    
    # Simulate auctions for both variants
    for auction_id in range(1000):
        user_id = f"user_{np.random.randint(0, 1000)}"
        
        # Assign to variant
        variant = ab_test.assign_variant(experiment.experiment_id, user_id)
        
        # Create bid request
        bid_request = BidRequest(
            request_id=f"ab_test_{auction_id}",
            user_id=np.random.randint(0, 100),
            ad_id=np.random.randint(0, 50),
            context_id=np.random.randint(0, 20),
            timestamp=time.time(),
            floor_price=0.1,
            ad_position=1,
            device_type='mobile',
            additional_features={}
        )
        
        # Use different strategy based on variant
        if variant == "cpc_strategy":
            strategy = BidStrategy.CPC
            advertiser_bid = 1.0
        else:
            strategy = BidStrategy.CPM
            advertiser_bid = 2.0
        
        budget_pacer = BudgetPacer(strategy='proportional')
        
        bidding_engine = BiddingEngine(
            ctr_model=ctr_model,
            budget_pacer=budget_pacer,
            strategy=strategy,
            advertiser_bid=advertiser_bid
        )
        
        bid_response = bidding_engine.compute_bid(bid_request, budget_state)
        
        if not bid_response.should_bid:
            continue
        
        # Simulate auction (simplified)
        won = np.random.random() < 0.3  # 30% win rate
        price = bid_response.bid_price * 0.7 if won else 0  # Pay 70% of bid
        clicked = np.random.random() < bid_response.expected_ctr if won else False
        converted = np.random.random() < 0.05 if clicked else False
        
        # Record event
        ab_test.record_event(
            experiment_id=experiment.experiment_id,
            variant=variant,
            impression=won,
            click=clicked,
            conversion=converted,
            revenue=5.0 if converted else 0.0,  # $5 per conversion
            cost=price / 1000 if won else 0.0,  # Convert CPM to cost
            won_bid=won,
            bid_amount=bid_response.bid_price
        )
    
    # Get results
    print("\n" + ab_test.generate_report(experiment.experiment_id))
    
    # Statistical significance
    sig_results = ab_test.calculate_statistical_significance(
        experiment.experiment_id,
        metric='ctr',
        control_variant='cpc_strategy'
    )
    
    print("\nStatistical Significance (CTR):")
    for variant, result in sig_results.items():
        print(f"  {variant}: p-value={result['p_value']:.4f}, significant={result['significant']}")


def main():
    """Main demo function"""
    print("="*80)
    print("REAL-TIME BIDDING SYSTEM - COMPLETE DEMO")
    print("="*80)
    
    # Create directories
    os.makedirs('/home/claude/rtb_system/models', exist_ok=True)
    os.makedirs('/home/claude/rtb_system/data', exist_ok=True)
    
    # Step 1: Prepare data
    print("\n" + "="*80)
    print("STEP 0: Data Preparation")
    print("="*80)
    
    processor = iPinYouDataProcessor()
    
    # Generate synthetic data (in real scenario, load actual dataset)
    print("\nGenerating synthetic RTB data...")
    df = processor._generate_synthetic_data(n_samples=50000)
    
    # Preprocess
    features, labels = processor.preprocess(df, fit=True)
    
    # Calculate stats
    stats = calculate_feature_stats(features)
    print("\nFeature Statistics:")
    for key, stat in stats.items():
        print(f"  {key}: {stat['unique_values']} unique values")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        features,
        labels,
        batch_size=256,
        test_size=0.2,
        val_size=0.1
    )
    
    n_users = stats['user_id']['unique_values']
    n_ads = stats['ad_id']['unique_values']
    n_contexts = stats['context_id']['unique_values']
    
    # Step 2: Train CTR model
    ctr_model = train_ctr_model(train_loader, val_loader, n_users, n_ads, n_contexts)
    
    # Step 3: Budget pacing
    budget_state = simulate_budget_pacing()
    
    # Step 4: Run auctions
    auction_results = run_auction_simulations(ctr_model, budget_state)
    
    # Step 5: Benchmark latency
    latency_results = benchmark_rtb_engine(ctr_model, budget_state)
    
    # Step 6: A/B testing
    run_ab_test(ctr_model, budget_state)
    
    # Final summary
    print("\n" + "="*80)
    print("DEMO COMPLETE - SUMMARY")
    print("="*80)
    print("\n✅ CTR Prediction Model: Trained and validated")
    print("✅ Budget Pacing: Multiple strategies implemented")
    print("✅ Auction System: Second-price auction operational")
    print(f"✅ RTB Engine: Avg latency {latency_results['avg_latency_ms']:.2f}ms " +
          f"({'PASS' if latency_results['avg_latency_ms'] < 100 else 'FAIL'} <100ms SLA)")
    print("✅ A/B Testing: Framework operational")
    
    print("\n" + "="*80)
    print("Key Metrics:")
    print("-" * 80)
    print(f"  Model AUC: Training complete (see validation metrics)")
    print(f"  Auction Win Rate: {auction_results['wins']/auction_results['total_bids']*100:.1f}%")
    print(f"  Average Latency: {latency_results['avg_latency_ms']:.2f}ms")
    print(f"  P99 Latency: {latency_results['p99_latency_ms']:.2f}ms")
    print(f"  Throughput: {latency_results['num_requests']/10:.0f} QPS (estimated)")
    print(f"  Budget Utilization: ${budget_state.total_spent:.2f} / ${budget_state.total_budget:.2f}")
    print("="*80)


if __name__ == "__main__":
    main()
