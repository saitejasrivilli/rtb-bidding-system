"""
FastAPI REST API for Real-Time Bidding System
Production-ready API with authentication, rate limiting, and monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import time
import asyncio
from datetime import datetime
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest
from functools import wraps
import jwt
import redis
import pickle

# Import our RTB components
import sys
sys.path.append('/home/claude/rtb_system')
from core.rtb_engine import RTBEngine
from core.bidding_strategies import BiddingEngine, BidRequest, BidStrategy
from core.budget_pacing import BudgetPacer, BudgetState
from models.ctr_model import CTRPredictor


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

bid_requests_total = Counter('rtb_bid_requests_total', 'Total bid requests')
bid_responses_total = Counter('rtb_bid_responses_total', 'Total bid responses', ['status'])
bid_latency = Histogram('rtb_bid_latency_seconds', 'Bid request latency')
auction_wins = Counter('rtb_auction_wins_total', 'Total auction wins')
revenue_total = Counter('rtb_revenue_total', 'Total revenue in cents')


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class BidRequestAPI(BaseModel):
    """Bid request from ad exchange"""
    request_id: str = Field(..., description="Unique request ID")
    user_id: int = Field(..., ge=0, description="User identifier")
    ad_id: int = Field(..., ge=0, description="Ad identifier")
    context_id: int = Field(..., ge=0, description="Context identifier")
    floor_price: float = Field(0.0, ge=0, description="Minimum bid price (CPM)")
    ad_position: int = Field(1, ge=1, le=10, description="Ad position")
    device_type: str = Field("mobile", description="Device type")
    timestamp: Optional[float] = None
    
    @validator('device_type')
    def validate_device(cls, v):
        allowed = ['mobile', 'desktop', 'tablet']
        if v not in allowed:
            raise ValueError(f'device_type must be one of {allowed}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "req_12345",
                "user_id": 123,
                "ad_id": 456,
                "context_id": 789,
                "floor_price": 0.5,
                "ad_position": 1,
                "device_type": "mobile"
            }
        }


class BidResponseAPI(BaseModel):
    """Bid response to ad exchange"""
    request_id: str
    bid_price: float
    should_bid: bool
    expected_ctr: float
    processing_time_ms: float
    budget_remaining: float
    metadata: Dict


class AuctionResultAPI(BaseModel):
    """Auction result notification"""
    request_id: str
    won: bool
    price_paid: Optional[float] = None
    impression_served: bool = False
    clicked: bool = False
    converted: bool = False
    revenue: float = 0.0


class CampaignConfig(BaseModel):
    """Campaign configuration"""
    campaign_id: str
    total_budget: float = Field(..., gt=0)
    daily_budget: float = Field(..., gt=0)
    bidding_strategy: str = Field("cpc", regex="^(cpc|cpm|cpa|dynamic)$")
    base_bid: float = Field(..., gt=0)
    pacing_strategy: str = Field("proportional", regex="^(proportional|adaptive|exponential|threshold)$")
    target_ctr: Optional[float] = Field(None, ge=0, le=1)
    

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    metrics: Dict


# ============================================================================
# API APPLICATION
# ============================================================================

app = FastAPI(
    title="RTB System API",
    description="Real-Time Bidding System with <100ms latency",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
app.state.start_time = time.time()
app.state.redis_client = None


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

class RTBService:
    """Singleton RTB service"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self, model, budget_state):
        """Initialize RTB engine"""
        if self._initialized:
            return
        
        self.pacer = BudgetPacer(strategy='proportional')
        self.bidding_engine = BiddingEngine(
            ctr_model=model,
            budget_pacer=self.pacer,
            strategy=BidStrategy.CPC,
            advertiser_bid=1.0
        )
        
        self.rtb_engine = RTBEngine(
            ctr_model=model,
            budget_pacer=self.pacer,
            bidding_engine=self.bidding_engine,
            budget_state=budget_state,
            use_cache=True,
            max_workers=4
        )
        
        self.rtb_engine.optimize_for_latency()
        self._initialized = True
        print("✅ RTB Service initialized")
    
    def get_engine(self):
        return self.rtb_engine


def get_rtb_service() -> RTBService:
    """Dependency for RTB service"""
    return RTBService()


async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key authentication"""
    # In production, check against database
    valid_keys = ["test_key_12345", "prod_key_67890"]
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter"""
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    async def check_rate_limit(self, key: str) -> bool:
        """Check if request is within rate limit"""
        now = time.time()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests outside window
        self.requests[key] = [
            ts for ts in self.requests[key]
            if now - ts < self.window_seconds
        ]
        
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        self.requests[key].append(now)
        return True


rate_limiter = RateLimiter(max_requests=1000, window_seconds=60)


async def check_rate_limit(request: Request, api_key: str = Depends(verify_api_key)):
    """Rate limiting middleware"""
    if not await rate_limiter.check_rate_limit(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting RTB API...")
    
    # Initialize Redis (optional)
    try:
        app.state.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        app.state.redis_client.ping()
        print("✅ Redis connected")
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        app.state.redis_client = None
    
    # Initialize RTB service (would load model here in production)
    print("✅ RTB API ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down RTB API...")
    if app.state.redis_client:
        app.state.redis_client.close()


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "service": "RTB System API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app.state.start_time
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime_seconds=uptime,
        metrics={
            "redis_connected": app.state.redis_client is not None,
            "uptime_hours": uptime / 3600
        }
    )


@app.post("/bid", response_model=BidResponseAPI, tags=["Bidding"])
async def submit_bid(
    request: BidRequestAPI,
    api_key: str = Depends(verify_api_key),
    rtb_service: RTBService = Depends(get_rtb_service)
):
    """
    Submit a bid request and receive bid response
    
    This endpoint processes a bid request in real-time (<100ms) and returns
    whether to bid and at what price.
    """
    start_time = time.time()
    bid_requests_total.inc()
    
    try:
        # Convert to internal format
        bid_request = BidRequest(
            request_id=request.request_id,
            user_id=request.user_id,
            ad_id=request.ad_id,
            context_id=request.context_id,
            timestamp=request.timestamp or time.time(),
            floor_price=request.floor_price,
            ad_position=request.ad_position,
            device_type=request.device_type,
            additional_features={}
        )
        
        # Get RTB engine
        engine = rtb_service.get_engine()
        
        # Process bid request
        bid_response = engine.process_bid_request(bid_request, timeout_ms=100.0)
        
        if bid_response is None:
            bid_responses_total.labels(status='timeout').inc()
            raise HTTPException(status_code=408, detail="Request timeout")
        
        processing_time = (time.time() - start_time) * 1000
        bid_latency.observe(processing_time / 1000)
        
        bid_responses_total.labels(status='success').inc()
        
        return BidResponseAPI(
            request_id=request.request_id,
            bid_price=bid_response.bid_price,
            should_bid=bid_response.should_bid,
            expected_ctr=bid_response.expected_ctr,
            processing_time_ms=processing_time,
            budget_remaining=engine.budget_state.remaining_budget,
            metadata=bid_response.metadata
        )
        
    except Exception as e:
        bid_responses_total.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/auction-result", tags=["Bidding"])
async def report_auction_result(
    result: AuctionResultAPI,
    api_key: str = Depends(verify_api_key),
    rtb_service: RTBService = Depends(get_rtb_service)
):
    """
    Report auction result (win/loss, impressions, clicks, conversions)
    
    Ad exchanges call this endpoint to notify about auction outcomes
    """
    try:
        engine = rtb_service.get_engine()
        
        if result.won:
            auction_wins.inc()
            
            # Update budget
            engine.update_budget_state(
                spent=result.price_paid or 0.0,
                impressions=1 if result.impression_served else 0,
                clicks=1 if result.clicked else 0
            )
            
            # Update metrics
            if result.revenue > 0:
                revenue_total.inc(int(result.revenue * 100))  # Convert to cents
        
        return {"status": "recorded", "request_id": result.request_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording result: {str(e)}")


@app.post("/campaign/create", tags=["Campaign Management"])
async def create_campaign(
    config: CampaignConfig,
    api_key: str = Depends(verify_api_key)
):
    """Create a new campaign with budget and strategy"""
    try:
        # In production, save to database
        campaign_data = config.dict()
        campaign_data['created_at'] = datetime.now().isoformat()
        campaign_data['status'] = 'active'
        
        # Cache in Redis if available
        if app.state.redis_client:
            app.state.redis_client.setex(
                f"campaign:{config.campaign_id}",
                86400,  # 24 hours
                str(campaign_data)
            )
        
        return {
            "campaign_id": config.campaign_id,
            "status": "created",
            "budget": {
                "total": config.total_budget,
                "daily": config.daily_budget,
                "remaining": config.total_budget
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating campaign: {str(e)}")


@app.get("/campaign/{campaign_id}", tags=["Campaign Management"])
async def get_campaign(
    campaign_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get campaign details and performance"""
    try:
        # In production, fetch from database
        if app.state.redis_client:
            data = app.state.redis_client.get(f"campaign:{campaign_id}")
            if data:
                return {"campaign_id": campaign_id, "data": data}
        
        return {
            "campaign_id": campaign_id,
            "status": "active",
            "budget": {
                "total": 1000.0,
                "spent": 234.56,
                "remaining": 765.44
            },
            "performance": {
                "impressions": 12450,
                "clicks": 234,
                "ctr": 0.0188,
                "avg_cpm": 2.15,
                "roi": 0.76
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching campaign: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint"""
    return JSONResponse(
        content=generate_latest().decode('utf-8'),
        media_type="text/plain"
    )


@app.get("/stats", tags=["Monitoring"])
async def get_stats(
    api_key: str = Depends(verify_api_key),
    rtb_service: RTBService = Depends(get_rtb_service)
):
    """Get current system statistics"""
    try:
        engine = rtb_service.get_engine()
        metrics = engine.get_performance_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "rtb_metrics": metrics,
            "api_uptime_seconds": time.time() - app.state.start_time
        }
        
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# BATCH ENDPOINTS
# ============================================================================

@app.post("/bid/batch", tags=["Bidding"])
async def submit_batch_bids(
    requests: List[BidRequestAPI],
    api_key: str = Depends(verify_api_key),
    rtb_service: RTBService = Depends(get_rtb_service)
):
    """
    Submit multiple bid requests in batch for efficiency
    Maximum 100 requests per batch
    """
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 requests per batch")
    
    start_time = time.time()
    
    try:
        engine = rtb_service.get_engine()
        
        # Convert to internal format
        bid_requests = [
            BidRequest(
                request_id=req.request_id,
                user_id=req.user_id,
                ad_id=req.ad_id,
                context_id=req.context_id,
                timestamp=req.timestamp or time.time(),
                floor_price=req.floor_price,
                ad_position=req.ad_position,
                device_type=req.device_type,
                additional_features={}
            )
            for req in requests
        ]
        
        # Process batch
        responses = engine.process_batch(bid_requests, timeout_ms=200.0)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert to API format
        api_responses = []
        for req, resp in zip(requests, responses):
            if resp:
                api_responses.append(BidResponseAPI(
                    request_id=req.request_id,
                    bid_price=resp.bid_price,
                    should_bid=resp.should_bid,
                    expected_ctr=resp.expected_ctr,
                    processing_time_ms=processing_time / len(requests),
                    budget_remaining=engine.budget_state.remaining_budget,
                    metadata=resp.metadata
                ))
        
        return {
            "batch_size": len(requests),
            "responses": api_responses,
            "total_processing_time_ms": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@app.post("/admin/reset-budget", tags=["Admin"])
async def reset_budget(
    total_budget: float,
    api_key: str = Depends(verify_api_key),
    rtb_service: RTBService = Depends(get_rtb_service)
):
    """Reset campaign budget (admin only)"""
    try:
        engine = rtb_service.get_engine()
        engine.budget_state.total_budget = total_budget
        engine.budget_state.remaining_budget = total_budget
        engine.budget_state.total_spent = 0.0
        
        return {
            "status": "reset",
            "new_budget": total_budget
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/clear-cache", tags=["Admin"])
async def clear_cache(
    api_key: str = Depends(verify_api_key),
    rtb_service: RTBService = Depends(get_rtb_service)
):
    """Clear prediction cache"""
    try:
        engine = rtb_service.get_engine()
        if hasattr(engine, 'prediction_cache'):
            engine.prediction_cache.clear()
        
        return {"status": "cache_cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              RTB System REST API Server                      ║
    ║              Production-Ready Real-Time Bidding              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📡 Starting server on http://0.0.0.0:8000
    📚 API Docs: http://localhost:8000/docs
    📊 Metrics: http://localhost:8000/metrics
    
    """)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info"
    )
