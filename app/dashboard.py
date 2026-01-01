import streamlit as st
import requests
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Stock Analyzer", layout="wide")

st.title("📈 Stock Fundamental Dashboard")
st.markdown("---")

# Sidebar input
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="e.g., AAPL, MSFT, GOOGL")
if st.sidebar.button("🔍 Analyze", type="primary"):

    with st.spinner("Analyzing fundamentals..."):
        try:
            resp = requests.post("http://localhost:8000/analyze", json={"symbol": symbol.upper()}).json()
            
            # Row 1: Score + Rating
            col1, col2 = st.columns(2)
            col1.metric("Score", f"{resp['score']:.1f}/100", delta=None)
            col2.metric("Recommendation", resp['rating'], delta=None)
            
            st.markdown(resp['disclaimer'])
            
            # Row 2: Metrics table
            metrics_df = st.dataframe(
                {k: [v] for k, v in resp['metrics'].items()},
                use_container_width=True
            )
            
            # Row 3: Charts (responsive)
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("📊 Key Metrics")
                fig1 = go.Figure(**json.loads(resp['charts']['metrics_bar']))
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_chart2:
                st.subheader("📈 Price Trend (1Y)")
                fig2 = go.Figure(**json.loads(resp['charts']['price_trend']))
                st.plotly_chart(fig2, use_container_width=True)
            
            # Full financials pie
            st.subheader("💰 Income Breakdown")
            fig3 = go.Figure(**json.loads(resp['charts']['financials_pie']))
            st.plotly_chart(fig3, use_container_width=True)
            
        except Exception as e:
            st.error(f"Analysis failed: {e}")

# Footer
st.markdown("---")
st.caption("Powered by FastAPI + yfinance + Plotly | Educational demo only")
