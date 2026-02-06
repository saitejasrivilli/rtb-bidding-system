"""
Real-Time Bidding Engine
High-performance bidding system with <100ms latency requirement
"""

import time
import asyncio
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import torch

from .bidding_strategies import BiddingEngine, BidRequest, BidResponse
from .auction import AuctionSimulator, Bidder, AuctionResult
from .budget_pacing import BudgetPacer, BudgetState


@dataclass
class RTBMetrics:
    """Real-time bidding performance metrics"""
    total_requests: int = 0
    total_responses: int = 0
    total_wins: int = 0
    total_spent: float = 0.0
    total_impressions: int = 0
    total_clicks: int = 0
    
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    win_rate: float = 0.0
    avg_cpm: float = 0.0
    avg_ctr: float = 0.0
    
    def update(self, response_time_ms: float, won: bool, price: float = 0.0, clicked: bool = False):
        """Update metrics"""
        self.total_requests += 1
        self.response_times.append(response_time_ms)
        
        if won:
            self.total_responses += 1
            self.total_wins += 1
            self.total_spent += price
            self.total_impressions += 1
            
            if clicked:
                self.total_clicks += 1
        
        # Update rates
        if self.total_responses > 0:
            self.win_rate = self.total_wins / self.total_responses
        
        if self.total_impressions > 0:
            self.avg_cpm = (self.total_spent / self.total_impressions) * 1000
            self.avg_ctr = self.total_clicks / self.total_impressions
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        return {
            'total_requests': self.total_requests,
            'total_responses': self.total_responses,
            'total_wins': self.total_wins,
            'win_rate': self.win_rate,
            'total_spent': self.total_spent,
            'total_impressions': self.total_impressions,
            'total_clicks': self.total_clicks,
            'avg_cpm': self.avg_cpm,
            'avg_ctr': self.avg_ctr,
            'avg_response_time_ms': np.mean(list(self.response_times)) if self.response_times else 0,
            'p95_response_time_ms': np.percentile(list(self.response_times), 95) if self.response_times else 0,
            'p99_response_time_ms': np.percentile(list(self.response_times), 99) if self.response_times else 0,
        }


class ModelCache:
    """
    Cache for fast CTR predictions
    Implements LRU cache for frequently seen feature combinations
    """
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        self.current_time = 0
    
    def get(self, key: tuple) -> Optional[float]:
        """Get cached prediction"""
        if key in self.cache:
            self.access_times[key] = self.current_time
            self.current_time += 1
            return self.cache[key]
        return None
    
    def put(self, key: tuple, value: float):
        """Put prediction in cache"""
        if len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = value
        self.access_times[key] = self.current_time
        self.current_time += 1
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_times.clear()


class RTBEngine:
    """
    Real-Time Bidding Engine with <100ms latency
    """
    
    def __init__(
        self,
        ctr_model,
        budget_pacer: BudgetPacer,
        bidding_engine: BiddingEngine,
        budget_state: BudgetState,
        use_cache: bool = True,
        max_workers: int = 4
    ):
        """
        Args:
            ctr_model: Trained CTR prediction model
            budget_pacer: Budget pacing algorithm
            bidding_engine: Bidding strategy engine
            budget_state: Current budget state
            use_cache: Whether to use prediction cache
            max_workers: Number of worker threads
        """
        self.ctr_model = ctr_model
        self.budget_pacer = budget_pacer
        self.bidding_engine = bidding_engine
        self.budget_state = budget_state
        
        # Performance optimizations
        self.use_cache = use_cache
        if use_cache:
            self.prediction_cache = ModelCache(max_size=10000)
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Set model to eval mode and optimize
        self.ctr_model.eval()
        if torch.cuda.is_available():
            self.ctr_model = self.ctr_model.cuda()
        
        # Metrics
        self.metrics = RTBMetrics()
        
        # Thread safety
        self.lock = threading.Lock()
        
        print(f"RTB Engine initialized with {max_workers} workers")
    
    def process_bid_request(
        self,
        bid_request: BidRequest,
        timeout_ms: float = 100.0
    ) -> Optional[BidResponse]:
        """
        Process a single bid request with timeout
        
        Args:
            bid_request: The bid request to process
            timeout_ms: Maximum processing time in milliseconds
        
        Returns:
            BidResponse or None if timeout/error
        """
        start_time = time.time()
        
        try:
            # Fast path: check cache
            if self.use_cache:
                cache_key = (
                    bid_request.user_id,
                    bid_request.ad_id,
                    bid_request.context_id
                )
                cached_ctr = self.prediction_cache.get(cache_key)
                
                if cached_ctr is not None:
                    # Use cached prediction
                    bid_response = self.bidding_engine.compute_bid(
                        bid_request,
                        self.budget_state,
                        predicted_ctr=cached_ctr
                    )
                else:
                    # Predict and cache
                    predicted_ctr = self._fast_predict_ctr(bid_request)
                    self.prediction_cache.put(cache_key, predicted_ctr)
                    
                    bid_response = self.bidding_engine.compute_bid(
                        bid_request,
                        self.budget_state,
                        predicted_ctr=predicted_ctr
                    )
            else:
                # No cache
                bid_response = self.bidding_engine.compute_bid(
                    bid_request,
                    self.budget_state
                )
            
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                print(f"Warning: Request timeout ({elapsed_ms:.2f}ms > {timeout_ms}ms)")
                return None
            
            # Update metrics
            self.metrics.update(elapsed_ms, False)  # Win status updated later
            
            return bid_response
            
        except Exception as e:
            print(f"Error processing bid request: {e}")
            return None
    
    def _fast_predict_ctr(self, bid_request: BidRequest) -> float:
        """
        Fast CTR prediction with optimizations
        """
        features = {
            'user_id': torch.tensor([bid_request.user_id]),
            'ad_id': torch.tensor([bid_request.ad_id]),
            'context_id': torch.tensor([bid_request.context_id])
        }
        
        if torch.cuda.is_available():
            features = {k: v.cuda() for k, v in features.items()}
        
        with torch.no_grad():
            ctr = self.ctr_model(features)
        
        return float(ctr.cpu().item())
    
    async def process_bid_request_async(
        self,
        bid_request: BidRequest,
        timeout_ms: float = 100.0
    ) -> Optional[BidResponse]:
        """
        Async version of process_bid_request
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.process_bid_request,
            bid_request,
            timeout_ms
        )
        return result
    
    def process_batch(
        self,
        bid_requests: List[BidRequest],
        timeout_ms: float = 100.0
    ) -> List[Optional[BidResponse]]:
        """
        Process multiple bid requests in batch for efficiency
        """
        start_time = time.time()
        
        # Batch predict CTRs
        user_ids = torch.tensor([br.user_id for br in bid_requests])
        ad_ids = torch.tensor([br.ad_id for br in bid_requests])
        context_ids = torch.tensor([br.context_id for br in bid_requests])
        
        features = {
            'user_id': user_ids,
            'ad_id': ad_ids,
            'context_id': context_ids
        }
        
        if torch.cuda.is_available():
            features = {k: v.cuda() for k, v in features.items()}
        
        with torch.no_grad():
            predicted_ctrs = self.ctr_model(features).cpu().numpy()
        
        # Generate bid responses
        responses = []
        for bid_request, predicted_ctr in zip(bid_requests, predicted_ctrs):
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                responses.append(None)
                continue
            
            bid_response = self.bidding_engine.compute_bid(
                bid_request,
                self.budget_state,
                predicted_ctr=float(predicted_ctr)
            )
            responses.append(bid_response)
            
            # Update cache
            if self.use_cache:
                cache_key = (bid_request.user_id, bid_request.ad_id, bid_request.context_id)
                self.prediction_cache.put(cache_key, float(predicted_ctr))
        
        return responses
    
    def update_budget_state(
        self,
        spent: float,
        impressions: int = 1,
        clicks: int = 0
    ):
        """Update budget state after auction result"""
        with self.lock:
            self.budget_state.total_spent += spent
            self.budget_state.remaining_budget -= spent
            self.budget_state.total_impressions += impressions
            self.budget_state.total_clicks += clicks
            
            # Update spend rate (moving average)
            time_elapsed_hours = (
                self.budget_state.current_time - self.budget_state.time_start
            ).total_seconds() / 3600
            
            if time_elapsed_hours > 0:
                self.budget_state.current_spend_rate = (
                    self.budget_state.total_spent / time_elapsed_hours
                )
    
    def record_auction_result(
        self,
        bid_response: BidResponse,
        won: bool,
        price_paid: float = 0.0,
        clicked: bool = False
    ):
        """Record auction result and update metrics"""
        if won:
            self.update_budget_state(price_paid, impressions=1, clicks=1 if clicked else 0)
        
        # Update metrics (response time was recorded during bid)
        self.metrics.update(0, won, price_paid, clicked)  # 0 for time since already recorded
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.metrics.get_summary()
    
    def optimize_for_latency(self):
        """
        Apply optimizations to reduce latency
        """
        # Compile model (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.ctr_model = torch.compile(self.ctr_model, mode='reduce-overhead')
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"Could not compile model: {e}")
        
        # Warm up cache
        print("Warming up prediction cache...")
        for i in range(1000):
            dummy_request = BidRequest(
                request_id=f"warmup_{i}",
                user_id=i % 100,
                ad_id=i % 50,
                context_id=i % 20,
                timestamp=time.time(),
                floor_price=0.1,
                ad_position=1,
                device_type='mobile',
                additional_features={}
            )
            self.process_bid_request(dummy_request, timeout_ms=1000)
        
        print("Warm-up complete")


class RTBBenchmark:
    """
    Benchmark RTB engine performance
    """
    
    def __init__(self, rtb_engine: RTBEngine):
        self.rtb_engine = rtb_engine
    
    def benchmark_latency(
        self,
        num_requests: int = 10000,
        batch_size: int = 1
    ) -> Dict:
        """
        Benchmark latency performance
        
        Args:
            num_requests: Number of requests to test
            batch_size: Batch size for processing
        
        Returns:
            Latency statistics
        """
        print(f"Benchmarking with {num_requests} requests...")
        
        latencies = []
        
        for i in range(0, num_requests, batch_size):
            # Generate random bid requests
            requests = []
            for j in range(batch_size):
                request = BidRequest(
                    request_id=f"bench_{i}_{j}",
                    user_id=np.random.randint(0, 1000),
                    ad_id=np.random.randint(0, 500),
                    context_id=np.random.randint(0, 100),
                    timestamp=time.time(),
                    floor_price=0.1,
                    ad_position=np.random.randint(1, 6),
                    device_type=np.random.choice(['mobile', 'desktop', 'tablet']),
                    additional_features={}
                )
                requests.append(request)
            
            # Process and measure time
            start = time.time()
            
            if batch_size == 1:
                self.rtb_engine.process_bid_request(requests[0])
            else:
                self.rtb_engine.process_batch(requests)
            
            elapsed_ms = (time.time() - start) * 1000 / batch_size
            latencies.append(elapsed_ms)
        
        latencies = np.array(latencies)
        
        results = {
            'num_requests': num_requests,
            'batch_size': batch_size,
            'avg_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies),
            'std_latency_ms': np.std(latencies),
            'requests_meeting_100ms_sla': np.sum(latencies < 100) / len(latencies) * 100
        }
        
        print("\n=== Latency Benchmark Results ===")
        print(f"Average Latency: {results['avg_latency_ms']:.2f}ms")
        print(f"Median Latency: {results['median_latency_ms']:.2f}ms")
        print(f"P95 Latency: {results['p95_latency_ms']:.2f}ms")
        print(f"P99 Latency: {results['p99_latency_ms']:.2f}ms")
        print(f"Max Latency: {results['max_latency_ms']:.2f}ms")
        print(f"Requests meeting 100ms SLA: {results['requests_meeting_100ms_sla']:.1f}%")
        
        return results
    
    def benchmark_throughput(self, duration_seconds: int = 60) -> Dict:
        """
        Benchmark throughput (requests per second)
        """
        print(f"Benchmarking throughput for {duration_seconds} seconds...")
        
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration_seconds:
            request = BidRequest(
                request_id=f"throughput_{request_count}",
                user_id=np.random.randint(0, 1000),
                ad_id=np.random.randint(0, 500),
                context_id=np.random.randint(0, 100),
                timestamp=time.time(),
                floor_price=0.1,
                ad_position=1,
                device_type='mobile',
                additional_features={}
            )
            
            self.rtb_engine.process_bid_request(request)
            request_count += 1
        
        elapsed = time.time() - start_time
        qps = request_count / elapsed
        
        results = {
            'duration_seconds': elapsed,
            'total_requests': request_count,
            'queries_per_second': qps,
            'avg_time_per_request_ms': (elapsed / request_count) * 1000
        }
        
        print(f"\n=== Throughput Benchmark Results ===")
        print(f"Total Requests: {request_count}")
        print(f"Duration: {elapsed:.2f}s")
        print(f"QPS: {qps:.2f} requests/second")
        
        return results
