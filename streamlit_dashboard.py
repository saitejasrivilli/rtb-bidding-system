"""
Interactive Streamlit Dashboard for RTB System
Real-time monitoring and analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sys

sys.path.append('/home/claude/rtb_system')

# Page config
st.set_page_config(
    page_title="RTB System Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .status-good {
        color: #00C851;
    }
    .status-warning {
        color: #ffbb33;
    }
    .status-bad {
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/200x80.png?text=RTB+System", use_column_width=True)
    st.title("⚙️ Configuration")
    
    # Refresh rate
    refresh_rate = st.slider("Refresh Rate (seconds)", 1, 60, 5)
    auto_refresh = st.checkbox("Auto Refresh", value=True)
    
    st.markdown("---")
    
    # Campaign selection
    st.subheader("Campaign")
    campaign_id = st.selectbox(
        "Select Campaign",
        ["Campaign_001", "Campaign_002", "Campaign_003"]
    )
    
    # Date range
    st.subheader("Date Range")
    date_range = st.date_input(
        "Select Range",
        value=(datetime.now() - timedelta(days=7), datetime.now())
    )
    
    st.markdown("---")
    
    # System status
    st.subheader("🔴 System Status")
    st.write("**API:** 🟢 Online")
    st.write("**Database:** 🟢 Connected")
    st.write("**Cache:** 🟢 Active")
    st.write("**ML Model:** 🟢 Loaded")


# ============================================================================
# GENERATE MOCK DATA
# ============================================================================

@st.cache_data(ttl=5)
def generate_realtime_data():
    """Generate mock real-time data"""
    now = datetime.now()
    
    # Time series data (last 24 hours)
    hours = pd.date_range(end=now, periods=24, freq='H')
    
    data = {
        'timestamp': hours,
        'bid_requests': np.random.randint(5000, 8000, 24),
        'wins': np.random.randint(1500, 2500, 24),
        'impressions': np.random.randint(1500, 2500, 24),
        'clicks': np.random.randint(30, 80, 24),
        'spend': np.random.uniform(40, 80, 24),
        'revenue': np.random.uniform(50, 100, 24),
        'latency_ms': np.random.uniform(20, 40, 24),
    }
    
    df = pd.DataFrame(data)
    df['win_rate'] = df['wins'] / df['bid_requests'] * 100
    df['ctr'] = df['clicks'] / df['impressions'] * 100
    df['roi'] = (df['revenue'] - df['spend']) / df['spend'] * 100
    
    return df


@st.cache_data(ttl=5)
def get_current_metrics():
    """Get current system metrics"""
    return {
        'bid_requests': np.random.randint(150000, 180000),
        'wins': np.random.randint(45000, 55000),
        'win_rate': np.random.uniform(28, 35),
        'impressions': np.random.randint(45000, 55000),
        'clicks': np.random.randint(1200, 1800),
        'ctr': np.random.uniform(2.5, 3.5),
        'total_spend': np.random.uniform(1200, 1500),
        'total_revenue': np.random.uniform(1800, 2200),
        'roi': np.random.uniform(45, 65),
        'avg_latency': np.random.uniform(22, 28),
        'p99_latency': np.random.uniform(75, 90),
        'budget_remaining': np.random.uniform(400, 600),
        'budget_utilization': np.random.uniform(92, 98)
    }


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("📊 Real-Time Bidding System Dashboard")
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Get data
df = generate_realtime_data()
metrics = get_current_metrics()

# ============================================================================
# TOP-LEVEL METRICS
# ============================================================================

st.markdown("## 📈 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Bid Requests",
        f"{metrics['bid_requests']:,}",
        delta="+5.2%"
    )

with col2:
    win_rate = metrics['win_rate']
    status = "🟢" if win_rate > 30 else "🟡" if win_rate > 25 else "🔴"
    st.metric(
        f"{status} Win Rate",
        f"{win_rate:.1f}%",
        delta="+2.1%"
    )

with col3:
    ctr = metrics['ctr']
    status = "🟢" if ctr > 3 else "🟡" if ctr > 2 else "🔴"
    st.metric(
        f"{status} CTR",
        f"{ctr:.2f}%",
        delta="+0.3%"
    )

with col4:
    roi = metrics['roi']
    status = "🟢" if roi > 50 else "🟡" if roi > 30 else "🔴"
    st.metric(
        f"{status} ROI",
        f"{roi:.1f}%",
        delta="+3.5%"
    )

with col5:
    latency = metrics['avg_latency']
    status = "🟢" if latency < 30 else "🟡" if latency < 50 else "🔴"
    st.metric(
        f"{status} Avg Latency",
        f"{latency:.1f}ms",
        delta="-1.2ms",
        delta_color="inverse"
    )

st.markdown("---")

# ============================================================================
# TIME SERIES CHARTS
# ============================================================================

st.markdown("## 📉 Performance Over Time (Last 24 Hours)")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💰 Financial", "⚡ Latency", "🎯 Targeting"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Win Rate Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['win_rate'],
            mode='lines+markers',
            name='Win Rate',
            line=dict(color='#00C851', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 200, 81, 0.1)'
        ))
        fig.update_layout(
            title="Win Rate (%)",
            xaxis_title="Time",
            yaxis_title="Win Rate (%)",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CTR Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['ctr'],
            mode='lines+markers',
            name='CTR',
            line=dict(color='#4285F4', width=3),
            fill='tozeroy',
            fillcolor='rgba(66, 133, 244, 0.1)'
        ))
        fig.update_layout(
            title="Click-Through Rate (%)",
            xaxis_title="Time",
            yaxis_title="CTR (%)",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Bid Volume Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['bid_requests'],
        name='Bid Requests',
        marker_color='#ffbb33'
    ))
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['wins'],
        name='Wins',
        marker_color='#00C851'
    ))
    fig.update_layout(
        title="Bid Volume",
        xaxis_title="Time",
        yaxis_title="Count",
        barmode='group',
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Spend vs Revenue
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['spend'],
            mode='lines+markers',
            name='Spend',
            line=dict(color='#ff4444', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#00C851', width=2)
        ))
        fig.update_layout(
            title="Spend vs Revenue ($)",
            xaxis_title="Time",
            yaxis_title="Amount ($)",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROI Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['roi'],
            mode='lines+markers',
            name='ROI',
            line=dict(color='#9C27B0', width=3),
            fill='tozeroy'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Return on Investment (%)",
            xaxis_title="Time",
            yaxis_title="ROI (%)",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Budget Utilization
    budget_spent = metrics['total_spend']
    budget_total = budget_spent + metrics['budget_remaining']
    utilization = metrics['budget_utilization']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=utilization,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Budget Utilization (%)"},
        delta={'reference': 95},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 90], 'color': "gray"},
                {'range': [90, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Latency Over Time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['latency_ms'],
            mode='lines',
            name='Latency',
            line=dict(color='#FF5722', width=2)
        ))
        fig.add_hline(y=100, line_dash="dash", line_color="red", 
                     annotation_text="100ms SLA")
        fig.update_layout(
            title="Response Latency (ms)",
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Latency Distribution
        latency_data = df['latency_ms'].values
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=latency_data,
            nbinsx=20,
            marker_color='#FF5722',
            opacity=0.7
        ))
        fig.add_vline(x=100, line_dash="dash", line_color="red",
                     annotation_text="100ms SLA")
        fig.update_layout(
            title="Latency Distribution",
            xaxis_title="Latency (ms)",
            yaxis_title="Frequency",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Latency Percentiles
    st.subheader("Latency Percentiles")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("P50", f"{np.percentile(latency_data, 50):.1f}ms")
    with col2:
        st.metric("P95", f"{np.percentile(latency_data, 95):.1f}ms")
    with col3:
        st.metric("P99", f"{np.percentile(latency_data, 99):.1f}ms")
    with col4:
        sla_compliance = (latency_data < 100).sum() / len(latency_data) * 100
        st.metric("SLA Compliance", f"{sla_compliance:.1f}%")

with tab4:
    # Device Type Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        device_data = {
            'Device': ['Mobile', 'Desktop', 'Tablet'],
            'Impressions': [45000, 32000, 8000]
        }
        fig = px.pie(
            device_data,
            values='Impressions',
            names='Device',
            title='Impressions by Device Type',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Geographic Distribution
        geo_data = {
            'Region': ['North America', 'Europe', 'Asia', 'Others'],
            'Clicks': [450, 320, 180, 50]
        }
        fig = px.bar(
            geo_data,
            x='Region',
            y='Clicks',
            title='Clicks by Region',
            color='Clicks',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================================
# DATA TABLES
# ============================================================================

st.markdown("## 📋 Recent Activity")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Recent Auctions")
    recent_auctions = pd.DataFrame({
        'Request ID': [f'req_{i}' for i in range(10)],
        'Bid': np.random.uniform(1.0, 3.0, 10).round(2),
        'Won': np.random.choice(['✅', '❌'], 10, p=[0.3, 0.7]),
        'Price': np.random.uniform(0.8, 2.5, 10).round(2),
        'Time': pd.date_range(end=datetime.now(), periods=10, freq='1min')
    })
    st.dataframe(recent_auctions, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Top Performing Ads")
    top_ads = pd.DataFrame({
        'Ad ID': [f'ad_{i}' for i in range(10)],
        'Impressions': np.random.randint(1000, 5000, 10),
        'Clicks': np.random.randint(20, 100, 10),
        'CTR': np.random.uniform(1.5, 5.0, 10).round(2),
        'Revenue': np.random.uniform(50, 200, 10).round(2)
    })
    top_ads = top_ads.sort_values('Revenue', ascending=False)
    st.dataframe(top_ads, use_container_width=True, hide_index=True)

# ============================================================================
# CONTROL PANEL
# ============================================================================

st.markdown("---")
st.markdown("## 🎛️ Control Panel")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 Refresh Data", type="primary"):
        st.rerun()

with col2:
    if st.button("⏸️ Pause Campaign"):
        st.success("Campaign paused")

with col3:
    if st.button("📊 Export Report"):
        st.success("Report exported")

with col4:
    if st.button("⚙️ Adjust Budget"):
        st.info("Budget adjustment panel opened")

# ============================================================================
# AUTO REFRESH
# ============================================================================

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>RTB System Dashboard v1.0.0 | Last Updated: {}</p>
    <p>🟢 All Systems Operational</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
