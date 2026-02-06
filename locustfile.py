"""
Locust Load Testing for RTB System API
Tests system performance under various load conditions
"""

from locust import HttpUser, task, between, events
import random
import json
import time
from datetime import datetime


class RTBUser(HttpUser):
    """
    Simulates a user making bid requests to the RTB system
    """
    
    wait_time = between(0.1, 0.5)  # Wait 100-500ms between requests
    
    def on_start(self):
        """Initialize user session"""
        self.api_key = "test_key_12345"
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Track metrics
        self.successful_bids = 0
        self.total_bids = 0
        self.wins = 0
    
    @task(10)
    def submit_bid_request(self):
        """
        Submit a single bid request
        Weight: 10 (most common operation)
        """
        payload = {
            "request_id": f"req_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            "user_id": random.randint(0, 1000),
            "ad_id": random.randint(0, 500),
            "context_id": random.randint(0, 100),
            "floor_price": round(random.uniform(0.1, 1.0), 2),
            "ad_position": random.randint(1, 5),
            "device_type": random.choice(["mobile", "desktop", "tablet"])
        }
        
        with self.client.post(
            "/bid",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/bid (single)"
        ) as response:
            self.total_bids += 1
            
            if response.status_code == 200:
                data = response.json()
                self.successful_bids += 1
                
                # Check latency
                if response.elapsed.total_seconds() * 1000 > 100:
                    response.failure(f"Latency too high: {response.elapsed.total_seconds() * 1000:.1f}ms")
                else:
                    response.success()
                
                # Check if we should bid
                if data.get('should_bid'):
                    # Simulate auction result (30% win rate)
                    if random.random() < 0.3:
                        self.wins += 1
                        self._report_auction_result(
                            payload['request_id'],
                            won=True,
                            price_paid=data['bid_price'] * 0.7
                        )
            else:
                response.failure(f"Failed with status {response.status_code}")
    
    @task(3)
    def submit_batch_bid_requests(self):
        """
        Submit batch of bid requests
        Weight: 3 (less common than single requests)
        """
        batch_size = random.randint(10, 50)
        
        requests = []
        for i in range(batch_size):
            requests.append({
                "request_id": f"batch_{int(time.time() * 1000)}_{i}",
                "user_id": random.randint(0, 1000),
                "ad_id": random.randint(0, 500),
                "context_id": random.randint(0, 100),
                "floor_price": round(random.uniform(0.1, 1.0), 2),
                "ad_position": random.randint(1, 5),
                "device_type": random.choice(["mobile", "desktop", "tablet"])
            })
        
        with self.client.post(
            "/bid/batch",
            json=requests,
            headers=self.headers,
            catch_response=True,
            name=f"/bid/batch (size={batch_size})"
        ) as response:
            if response.status_code == 200:
                # Check average latency per request
                avg_latency = response.elapsed.total_seconds() * 1000 / batch_size
                if avg_latency > 100:
                    response.failure(f"Avg latency too high: {avg_latency:.1f}ms")
                else:
                    response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")
    
    @task(1)
    def get_campaign_details(self):
        """
        Get campaign details
        Weight: 1 (occasional operation)
        """
        campaign_id = f"Campaign_{random.randint(1, 3):03d}"
        
        with self.client.get(
            f"/campaign/{campaign_id}",
            headers=self.headers,
            catch_response=True,
            name="/campaign/{id}"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")
    
    @task(1)
    def get_stats(self):
        """
        Get system statistics
        Weight: 1 (occasional operation)
        """
        with self.client.get(
            "/stats",
            headers=self.headers,
            catch_response=True,
            name="/stats"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")
    
    @task(1)
    def health_check(self):
        """
        Health check endpoint
        Weight: 1 (monitoring)
        """
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")
    
    def _report_auction_result(self, request_id: str, won: bool, price_paid: float = None):
        """Report auction result back to system"""
        payload = {
            "request_id": request_id,
            "won": won,
            "price_paid": price_paid if won else None,
            "impression_served": won,
            "clicked": won and random.random() < 0.02,  # 2% CTR
            "converted": False,
            "revenue": 0.0
        }
        
        self.client.post(
            "/auction-result",
            json=payload,
            headers=self.headers,
            name="/auction-result"
        )


class HighLoadUser(RTBUser):
    """
    High-load user simulation for stress testing
    """
    wait_time = between(0.01, 0.1)  # Very short wait time
    
    @task(20)
    def rapid_fire_requests(self):
        """Submit requests rapidly"""
        self.submit_bid_request()


# ============================================================================
# EVENT HANDLERS
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print("=" * 80)
    print("🚀 RTB System Load Test Started")
    print("=" * 80)
    print(f"Target: {environment.host}")
    print(f"Users: {environment.runner.user_count if hasattr(environment.runner, 'user_count') else 'N/A'}")
    print("=" * 80)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops"""
    print("\n" + "=" * 80)
    print("🏁 RTB System Load Test Completed")
    print("=" * 80)
    
    # Print summary
    stats = environment.stats
    
    print("\n📊 Summary Statistics:")
    print(f"  Total Requests: {stats.total.num_requests:,}")
    print(f"  Total Failures: {stats.total.num_failures:,}")
    print(f"  Failure Rate: {stats.total.fail_ratio * 100:.2f}%")
    print(f"  Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"  Median Response Time: {stats.total.median_response_time:.2f}ms")
    print(f"  95th Percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"  99th Percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"  Requests/sec: {stats.total.total_rps:.2f}")
    
    # Check if SLA is met
    p99_latency = stats.total.get_response_time_percentile(0.99)
    sla_met = p99_latency < 100
    
    print(f"\n🎯 SLA Status (<100ms P99): {'✅ PASS' if sla_met else '❌ FAIL'}")
    print("=" * 80)


# ============================================================================
# CUSTOM LOAD SHAPES
# ============================================================================

from locust import LoadTestShape

class StepLoadShape(LoadTestShape):
    """
    Step load pattern - gradually increase load
    """
    
    step_time = 60  # seconds per step
    step_load = 50  # users per step
    spawn_rate = 10
    time_limit = 600  # 10 minutes total
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = (run_time // self.step_time) + 1
        user_count = current_step * self.step_load
        
        return (user_count, self.spawn_rate)


class SpikeLoadShape(LoadTestShape):
    """
    Spike load pattern - sudden traffic spikes
    """
    
    time_limit = 600
    spawn_rate = 50
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        # Create spike every 2 minutes
        if (run_time % 120) < 30:
            # Spike: 500 users
            return (500, self.spawn_rate)
        else:
            # Normal: 100 users
            return (100, self.spawn_rate)


# ============================================================================
# USAGE
# ============================================================================

"""
Run load tests:

# Basic load test
locust -f locustfile.py --host=http://localhost:8000 \\
  --users=100 --spawn-rate=10 --run-time=5m

# Step load test
locust -f locustfile.py --host=http://localhost:8000 \\
  --users=500 --spawn-rate=10 --run-time=10m \\
  --load-shape=StepLoadShape

# Stress test (high load)
locust -f locustfile.py --host=http://localhost:8000 \\
  --users=1000 --spawn-rate=50 --run-time=10m \\
  --user-classes=HighLoadUser

# With results export
locust -f locustfile.py --host=http://localhost:8000 \\
  --users=200 --spawn-rate=20 --run-time=10m \\
  --csv=results --html=report.html --headless

# Distributed load test (master)
locust -f locustfile.py --host=http://localhost:8000 \\
  --master --expect-workers=4

# Distributed load test (worker)
locust -f locustfile.py --worker --master-host=localhost
"""
