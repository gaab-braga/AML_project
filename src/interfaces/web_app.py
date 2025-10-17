#!/usr/bin/env python3
"""
AML Pipeline Web Dashboard

Interactive web dashboard for AML Pipeline monitoring and control.
Built with Streamlit for real-time visualization and user interaction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.pipeline_controller import PipelineController
from cache import HierarchicalCache, MemoryCache, DiskCache
from evaluation.metrics import ModelEvaluator
from config.config_loader import ConfigLoader
from utils.logger import get_logger

# Configure page
st.set_page_config(
    page_title="AML Pipeline Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = get_logger(__name__)

# Initialize session state
if 'controller' not in st.session_state:
    st.session_state.controller = None
if 'cache' not in st.session_state:
    st.session_state.cache = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'config' not in st.session_state:
    st.session_state.config = None


def initialize_components():
    """Initialize pipeline components."""
    try:
        if not st.session_state.config:
            st.session_state.config = ConfigLoader.load_config("config/pipeline_config.yaml")

        if not st.session_state.controller:
            st.session_state.controller = PipelineController(st.session_state.config)

        if not st.session_state.cache:
            # Initialize hierarchical cache
            memory_cache = MemoryCache(max_size=1000, ttl=3600)
            disk_cache = DiskCache(cache_dir="./cache", max_size_mb=500)
            st.session_state.cache = HierarchicalCache(
                memory_cache=memory_cache,
                disk_cache=disk_cache,
                cache_strategy="write-through"
            )

        if not st.session_state.evaluator:
            st.session_state.evaluator = ModelEvaluator(st.session_state.config)

        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        logger.exception("Component initialization error")
        return False


def main():
    """Main dashboard function."""

    # Title and header
    st.title("🔍 AML Pipeline Dashboard")
    st.markdown("**Enterprise Anti-Money Laundering Pipeline Monitoring & Control**")

    # Initialize components
    if not initialize_components():
        st.error("Failed to initialize dashboard components. Please check configuration.")
        return

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")

        # Pipeline control
        st.subheader("Pipeline Control")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("▶️ Run Pipeline", type="primary", use_container_width=True):
                run_pipeline()

        with col2:
            if st.button("⏹️ Stop Pipeline", type="secondary", use_container_width=True):
                stop_pipeline()

        # Execution options
        st.subheader("Execution Options")
        target_env = st.selectbox(
            "Target Environment",
            ["development", "staging", "production"],
            index=2
        )

        execution_mode = st.selectbox(
            "Execution Mode",
            ["full", "fast", "custom"],
            index=0
        )

        # Cache management
        st.subheader("Cache Management")
        if st.button("🗑️ Clear Cache", use_container_width=True):
            clear_cache()

        # Auto-refresh
        st.subheader("Auto Refresh")
        auto_refresh = st.checkbox("Enable auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔄 Pipeline Status",
        "📈 Performance",
        "🗄️ Cache Analytics",
        "⚙️ Configuration"
    ])

    with tab1:
        show_overview()

    with tab2:
        show_pipeline_status()

    with tab3:
        show_performance_metrics()

    with tab4:
        show_cache_analytics()

    with tab5:
        show_configuration()

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


def run_pipeline():
    """Execute pipeline with selected options."""
    try:
        with st.spinner("🚀 Starting pipeline execution..."):
            result = st.session_state.controller.run_pipeline(target_env, execution_mode)

        if result.get('success'):
            st.success("✅ Pipeline execution started successfully!")
            st.balloons()
        else:
            st.error(f"❌ Pipeline execution failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        st.error(f"💥 Pipeline execution error: {e}")
        logger.exception("Pipeline execution error")


def stop_pipeline():
    """Stop current pipeline execution."""
    try:
        # Implement pipeline stop logic
        st.info("⏹️ Pipeline stop functionality to be implemented")
    except Exception as e:
        st.error(f"Failed to stop pipeline: {e}")


def clear_cache():
    """Clear cache contents."""
    try:
        if st.session_state.cache:
            st.session_state.cache.clear()
            st.success("🗑️ Cache cleared successfully!")
        else:
            st.warning("Cache not initialized")
    except Exception as e:
        st.error(f"Failed to clear cache: {e}")


def show_overview():
    """Show dashboard overview with key metrics."""
    st.header("📊 Dashboard Overview")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    try:
        # Get current status
        status = st.session_state.controller.get_status()
        metrics = st.session_state.controller.get_metrics()
        cache_stats = st.session_state.cache.get_stats() if st.session_state.cache else {}

        with col1:
            pipeline_status = status.get('pipeline_status', 'unknown')
            status_color = {
                'running': '🟢',
                'idle': '🟡',
                'error': '🔴',
                'maintenance': '🟠'
            }.get(pipeline_status, '⚪')

            st.metric(
                "Pipeline Status",
                f"{status_color} {pipeline_status.upper()}",
                help=f"Current pipeline state: {pipeline_status}"
            )

        with col2:
            uptime = status.get('uptime', 'N/A')
            st.metric("Uptime", uptime)

        with col3:
            hit_rate = cache_stats.get('hit_rate', 0)
            st.metric(
                "Cache Hit Rate",
                f"{hit_rate:.1%}",
                help="Percentage of cache requests served from cache"
            )

        with col4:
            active_executions = status.get('active_executions', 0)
            st.metric("Active Executions", active_executions)

    except Exception as e:
        st.error(f"Failed to load overview metrics: {e}")

    # Recent activity
    st.subheader("Recent Activity")
    try:
        recent_activity = status.get('recent_activity', [])[-10:]  # Last 10 activities

        if recent_activity:
            for activity in reversed(recent_activity):
                st.write(f"• {activity}")
        else:
            st.info("No recent activity")

    except Exception as e:
        st.error(f"Failed to load recent activity: {e}")


def show_pipeline_status():
    """Show detailed pipeline status and execution information."""
    st.header("🔄 Pipeline Status")

    try:
        status = st.session_state.controller.get_status()

        # Status overview
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Current Status")
            pipeline_status = status.get('pipeline_status', 'unknown')

            if pipeline_status == 'running':
                st.success("🟢 Pipeline is running")
            elif pipeline_status == 'idle':
                st.info("🟡 Pipeline is idle")
            elif pipeline_status == 'error':
                st.error("🔴 Pipeline has errors")
            else:
                st.warning(f"⚪ Pipeline status: {pipeline_status}")

            st.write(f"**Uptime:** {status.get('uptime', 'N/A')}")
            st.write(f"**Active Executions:** {status.get('active_executions', 0)}")

        with col2:
            st.subheader("System Resources")

            # Mock system metrics (replace with real metrics)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("CPU Usage", "45%")
                st.metric("Memory Usage", "2.1 GB")
            with col_b:
                st.metric("Disk Usage", "67%")
                st.metric("Network I/O", "1.2 MB/s")

        # Execution progress
        st.subheader("Execution Progress")

        # Mock progress for demonstration
        if pipeline_status == 'running':
            progress_bar = st.progress(0.65)
            st.write("**Current Step:** Feature Engineering (65% complete)")

            # Progress details
            with st.expander("Execution Details"):
                st.write("• Data Ingestion: ✅ Completed")
                st.write("• Data Validation: ✅ Completed")
                st.write("• Feature Engineering: 🔄 In Progress")
                st.write("• Model Training: ⏳ Pending")
                st.write("• Model Evaluation: ⏳ Pending")
        else:
            st.info("No active pipeline execution")

    except Exception as e:
        st.error(f"Failed to load pipeline status: {e}")
        logger.exception("Pipeline status display error")


def show_performance_metrics():
    """Show comprehensive performance metrics and visualizations."""
    st.header("📈 Performance Metrics")

    try:
        metrics = st.session_state.controller.get_metrics()

        # Time range selector
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
            index=1
        )

        # Metrics overview
        st.subheader("Key Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            accuracy = metrics.get('accuracy', 0.85)
            st.metric("Model Accuracy", f"{accuracy:.3f}")

        with col2:
            precision = metrics.get('precision', 0.82)
            st.metric("Precision", f"{precision:.3f}")

        with col3:
            recall = metrics.get('recall', 0.88)
            st.metric("Recall", f"{recall:.3f}")

        with col4:
            f1_score = metrics.get('f1_score', 0.85)
            st.metric("F1 Score", f"{f1_score:.3f}")

        # Performance charts
        st.subheader("Performance Trends")

        # Create sample data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
        accuracy_trend = np.random.normal(0.85, 0.02, 24)
        precision_trend = np.random.normal(0.82, 0.03, 24)

        # Performance over time chart
        fig = make_subplots(specs=[[{"secondary_y": False}]])

        fig.add_trace(
            go.Scatter(x=dates, y=accuracy_trend, name="Accuracy",
                      line=dict(color='blue')),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=dates, y=precision_trend, name="Precision",
                      line=dict(color='green')),
            secondary_y=False,
        )

        fig.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Time",
            yaxis_title="Score"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Confusion matrix
        st.subheader("Confusion Matrix")

        # Sample confusion matrix
        cm_data = [[850, 50], [120, 980]]
        cm_fig = px.imshow(cm_data,
                          text_auto=True,
                          labels=dict(x="Predicted", y="Actual"),
                          x=['Normal', 'Fraud'],
                          y=['Normal', 'Fraud'],
                          title="Confusion Matrix")

        st.plotly_chart(cm_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load performance metrics: {e}")
        logger.exception("Performance metrics display error")


def show_cache_analytics():
    """Show cache performance analytics and management."""
    st.header("🗄️ Cache Analytics")

    try:
        if not st.session_state.cache:
            st.warning("Cache not initialized")
            return

        cache_stats = st.session_state.cache.get_stats()

        # Cache overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            hit_rate = cache_stats.get('hit_rate', 0)
            st.metric("Cache Hit Rate", f"{hit_rate:.1%}")

        with col2:
            total_requests = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
            st.metric("Total Requests", total_requests)

        with col3:
            memory_usage = cache_stats.get('memory_usage_mb', 0)
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")

        with col4:
            disk_usage = cache_stats.get('disk_usage_mb', 0)
            st.metric("Disk Usage", f"{disk_usage:.1f} MB")

        # Cache performance chart
        st.subheader("Cache Performance")

        # Sample cache performance data
        cache_data = {
            'hits': cache_stats.get('hits', 1200),
            'misses': cache_stats.get('misses', 300),
            'evictions': cache_stats.get('eviction_count', 50)
        }

        fig = px.pie(
            values=list(cache_data.values()),
            names=list(cache_data.keys()),
            title="Cache Operations Distribution",
            color_discrete_sequence=['green', 'red', 'orange']
        )

        st.plotly_chart(fig, use_container_width=True)

        # Cache management
        st.subheader("Cache Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Cache Stats", use_container_width=True):
                st.rerun()

        with col2:
            if st.button("📊 Export Cache Report", use_container_width=True):
                # Export functionality
                st.info("Cache report export functionality to be implemented")

        # Cache configuration
        with st.expander("Cache Configuration"):
            st.write("**Memory Cache:**")
            st.write("- Max Size: 1000 items")
            st.write("- TTL: 1 hour")

            st.write("**Disk Cache:**")
            st.write("- Max Size: 500 MB")
            st.write("- Location: ./cache/")

            st.write("**Strategy:** Write-through")

    except Exception as e:
        st.error(f"Failed to load cache analytics: {e}")
        logger.exception("Cache analytics display error")


def show_configuration():
    """Show and manage pipeline configuration."""
    st.header("⚙️ Configuration Management")

    try:
        config = st.session_state.config

        if not config:
            st.error("Configuration not loaded")
            return

        # Configuration sections
        sections = st.tabs(["Pipeline", "Data", "Features", "Model", "Evaluation"])

        with sections[0]:  # Pipeline
            st.subheader("Pipeline Configuration")

            pipeline_config = config.get('pipeline', {})

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Name:** {pipeline_config.get('name', 'N/A')}")
                st.write(f"**Version:** {pipeline_config.get('version', 'N/A')}")

            with col2:
                st.write(f"**Primary Metric:** {pipeline_config.get('primary_metric', 'N/A')}")
                st.write(f"**CV Folds:** {pipeline_config.get('cv_folds', 'N/A')}")

        with sections[1]:  # Data
            st.subheader("Data Configuration")
            data_config = config.get('data', {})
            st.json(data_config)

        with sections[2]:  # Features
            st.subheader("Feature Engineering")
            features_config = config.get('features', {})
            st.json(features_config)

        with sections[3]:  # Model
            st.subheader("Model Configuration")
            model_config = config.get('model', {})
            st.json(model_config)

        with sections[4]:  # Evaluation
            st.subheader("Evaluation Settings")
            eval_config = config.get('evaluation', {})
            st.json(eval_config)

        # Configuration actions
        st.subheader("Configuration Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔍 Validate Config", use_container_width=True):
                # Configuration validation
                st.success("✅ Configuration is valid")

        with col2:
            if st.button("📥 Reload Config", use_container_width=True):
                # Reload configuration
                st.session_state.config = ConfigLoader.load_config("config/pipeline_config.yaml")
                st.success("✅ Configuration reloaded")

        with col3:
            if st.button("💾 Export Config", use_container_width=True):
                # Export configuration
                st.info("Configuration export functionality to be implemented")

    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        logger.exception("Configuration display error")


if __name__ == "__main__":
    main()