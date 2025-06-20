import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.visualizations import create_overview_charts
from utils.analytics import get_summary_stats

st.set_page_config(
    page_title="Video Game Sales Analytics Platform",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🎮 Video Game Sales Analytics Platform")
    st.markdown("---")
    
    # Sidebar navigation info
    st.sidebar.title("Navigation")
    st.sidebar.info(
        """
        Use the pages in the sidebar to explore different aspects of video game sales data:
        
        📊 **Data Upload & Cleaning**: Upload and preprocess your dataset
        
        🔍 **Exploratory Analysis**: Statistical analysis and insights
        
        📈 **Interactive Dashboard**: Dynamic visualizations and filters
        
        🌍 **Regional Analysis**: Geographic sales breakdowns
        """
    )
    
    # Main page content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to the Video Game Sales Analytics Platform")
        st.markdown("""
        This comprehensive analytics platform helps you analyze video game sales data with:
        
        ### Key Features:
        - **Data Upload & Cleaning**: Import CSV files and automatically clean your data
        - **Exploratory Data Analysis**: Statistical summaries and trend analysis
        - **Interactive Visualizations**: Dynamic charts with filtering capabilities
        - **Regional Analysis**: Geographic breakdowns of sales performance
        - **Platform & Genre Comparison**: Performance metrics across different categories
        - **Time Series Analysis**: Sales trends over time
        
        ### Getting Started:
        1. Navigate to **Data Upload & Cleaning** to import your dataset
        2. Explore **Exploratory Analysis** for statistical insights
        3. Use **Interactive Dashboard** for dynamic visualizations
        4. Check **Regional Analysis** for geographic insights
        """)
    
    with col2:
        st.header("Quick Stats")
        
        # Check if data exists in session state
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            data = st.session_state.processed_data
            
            # Display quick statistics
            st.metric("Total Records", len(data))
            
            if 'Global_Sales' in data.columns:
                total_sales = data['Global_Sales'].sum()
                st.metric("Total Global Sales (M)", f"{total_sales:.2f}")
            
            if 'Platform' in data.columns:
                unique_platforms = data['Platform'].nunique()
                st.metric("Unique Platforms", unique_platforms)
            
            if 'Genre' in data.columns:
                unique_genres = data['Genre'].nunique()
                st.metric("Unique Genres", unique_genres)
            
            # Quick overview chart
            if len(data) > 0:
                st.subheader("Sales Overview")
                chart = create_overview_charts(data)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Upload data in the 'Data Upload & Cleaning' page to see statistics here.")
            
            st.markdown("""
            ### Expected Data Format:
            Your CSV file should contain columns such as:
            - **Name**: Game title
            - **Platform**: Gaming platform (PS4, Xbox, PC, etc.)
            - **Year**: Release year
            - **Genre**: Game genre
            - **Publisher**: Game publisher
            - **Global_Sales**: Global sales figures
            - **NA_Sales, EU_Sales, JP_Sales**: Regional sales
            """)
    
    st.markdown("---")
    st.markdown("### Sample Analysis Workflow")
    
    workflow_cols = st.columns(4)
    
    with workflow_cols[0]:
        st.markdown("**1. Data Upload**")
        st.markdown("📁 Import CSV files")
        st.markdown("🧹 Clean and validate")
        
    with workflow_cols[1]:
        st.markdown("**2. Exploration**")
        st.markdown("📊 Statistical analysis")
        st.markdown("🔍 Trend identification")
        
    with workflow_cols[2]:
        st.markdown("**3. Visualization**")
        st.markdown("📈 Interactive charts")
        st.markdown("🎛️ Dynamic filtering")
        
    with workflow_cols[3]:
        st.markdown("**4. Insights**")
        st.markdown("🌍 Regional analysis")
        st.markdown("💡 Actionable findings")

if __name__ == "__main__":
    main()
