import streamlit as st
import pandas as pd
import numpy as np
from utils.visualizations import (
    create_sales_trend_chart, create_platform_comparison, create_genre_analysis,
    create_publisher_analysis, create_scatter_analysis, create_time_series_analysis
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Interactive Dashboard", page_icon="📈", layout="wide")

def main():
    st.title("📈 Interactive Dashboard")
    st.markdown("Dynamic visualizations with filtering and interactive capabilities.")
    
    # Check if data exists
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("⚠️ No data found. Please upload and clean your data first in the 'Data Upload & Cleaning' page.")
        st.stop()
    
    data = st.session_state.processed_data
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    filtered_data = apply_filters(data)
    
    # Dashboard layout selection
    st.sidebar.header("📊 Dashboard Layout")
    dashboard_type = st.sidebar.selectbox(
        "Select Dashboard Type",
        ["Overview Dashboard", "Sales Performance", "Market Analysis", "Temporal Trends", "Custom Analysis"]
    )
    
    # Show filtered data info
    if len(filtered_data) != len(data):
        st.info(f"📊 Showing {len(filtered_data):,} records (filtered from {len(data):,} total records)")
    
    # Display selected dashboard
    if dashboard_type == "Overview Dashboard":
        show_overview_dashboard(filtered_data)
    elif dashboard_type == "Sales Performance":
        show_sales_performance_dashboard(filtered_data)
    elif dashboard_type == "Market Analysis":
        show_market_analysis_dashboard(filtered_data)
    elif dashboard_type == "Temporal Trends":
        show_temporal_trends_dashboard(filtered_data)
    elif dashboard_type == "Custom Analysis":
        show_custom_analysis_dashboard(filtered_data)

def apply_filters(data):
    """Apply sidebar filters to the data."""
    filtered_data = data.copy()
    
    # Year filter
    if 'Year' in data.columns:
        min_year = int(data['Year'].min()) if not pd.isna(data['Year'].min()) else 1980
        max_year = int(data['Year'].max()) if not pd.isna(data['Year'].max()) else 2025
        
        year_range = st.sidebar.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1
        )
        filtered_data = filtered_data[
            (filtered_data['Year'] >= year_range[0]) & 
            (filtered_data['Year'] <= year_range[1])
        ]
    
    # Platform filter
    if 'Platform' in data.columns:
        platforms = sorted(data['Platform'].dropna().unique())
        selected_platforms = st.sidebar.multiselect(
            "Platforms",
            platforms,
            default=platforms
        )
        if selected_platforms:
            filtered_data = filtered_data[filtered_data['Platform'].isin(selected_platforms)]
    
    # Genre filter
    if 'Genre' in data.columns:
        genres = sorted(data['Genre'].dropna().unique())
        selected_genres = st.sidebar.multiselect(
            "Genres",
            genres,
            default=genres
        )
        if selected_genres:
            filtered_data = filtered_data[filtered_data['Genre'].isin(selected_genres)]
    
    # Publisher filter (top 20 for performance)
    if 'Publisher' in data.columns:
        top_publishers = data['Publisher'].value_counts().head(20).index.tolist()
        selected_publishers = st.sidebar.multiselect(
            "Publishers (Top 20)",
            top_publishers,
            default=top_publishers
        )
        if selected_publishers:
            filtered_data = filtered_data[filtered_data['Publisher'].isin(selected_publishers)]
    
    # Sales range filter
    if 'Global_Sales' in data.columns:
        min_sales = float(data['Global_Sales'].min())
        max_sales = float(data['Global_Sales'].max())
        
        sales_range = st.sidebar.slider(
            "Global Sales Range (Millions)",
            min_value=min_sales,
            max_value=max_sales,
            value=(min_sales, max_sales),
            step=0.1
        )
        filtered_data = filtered_data[
            (filtered_data['Global_Sales'] >= sales_range[0]) & 
            (filtered_data['Global_Sales'] <= sales_range[1])
        ]
    
    return filtered_data

def show_overview_dashboard(data):
    """Show comprehensive overview dashboard."""
    st.header("🎮 Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", f"{len(data):,}")
    with col2:
        if 'Global_Sales' in data.columns:
            st.metric("Total Sales", f"{data['Global_Sales'].sum():.2f}M")
        else:
            st.metric("Unique Platforms", data['Platform'].nunique() if 'Platform' in data.columns else 0)
    with col3:
        if 'Platform' in data.columns:
            st.metric("Platforms", data['Platform'].nunique())
        else:
            st.metric("Unique Genres", data['Genre'].nunique() if 'Genre' in data.columns else 0)
    with col4:
        if 'Genre' in data.columns:
            st.metric("Genres", data['Genre'].nunique())
        else:
            st.metric("Data Points", len(data))
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Top games
        if 'Global_Sales' in data.columns and 'Name' in data.columns:
            st.subheader("🏆 Top 10 Best-Selling Games")
            top_games = data.nlargest(10, 'Global_Sales')
            fig = px.bar(
                top_games,
                x='Global_Sales',
                y='Name',
                orientation='h',
                title="Top 10 Games by Global Sales",
                color='Global_Sales',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Platform distribution
        if 'Platform' in data.columns:
            st.subheader("🎯 Platform Distribution")
            platform_counts = data['Platform'].value_counts().head(10)
            fig = px.pie(
                values=platform_counts.values,
                names=platform_counts.index,
                title="Games by Platform (Top 10)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Genre analysis
        if 'Genre' in data.columns and 'Global_Sales' in data.columns:
            st.subheader("📊 Genre Performance")
            genre_sales = data.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
            fig = px.bar(
                x=genre_sales.index,
                y=genre_sales.values,
                title="Total Sales by Genre",
                color=genre_sales.values,
                color_continuous_scale='Viridis'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Yearly trends
        if 'Year' in data.columns and 'Global_Sales' in data.columns:
            st.subheader("📈 Sales Trends Over Time")
            yearly_sales = data.groupby('Year')['Global_Sales'].sum()
            fig = px.line(
                x=yearly_sales.index,
                y=yearly_sales.values,
                title="Global Sales by Year",
                markers=True
            )
            fig.update_layout(xaxis_title="Year", yaxis_title="Sales (Millions)")
            st.plotly_chart(fig, use_container_width=True)

def show_sales_performance_dashboard(data):
    """Show sales performance focused dashboard."""
    st.header("💰 Sales Performance Dashboard")
    
    if 'Global_Sales' not in data.columns:
        st.error("Sales data not available in the dataset.")
        return
    
    # Sales metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sales", f"{data['Global_Sales'].sum():.2f}M")
    with col2:
        st.metric("Average Sales", f"{data['Global_Sales'].mean():.4f}M")
    with col3:
        st.metric("Median Sales", f"{data['Global_Sales'].median():.4f}M")
    with col4:
        st.metric("Max Sales", f"{data['Global_Sales'].max():.2f}M")
    
    # Platform comparison
    st.subheader("🎮 Platform Performance Comparison")
    fig = create_platform_comparison(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Publisher analysis
    st.subheader("🏢 Publisher Performance")
    fig = create_publisher_analysis(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Sales distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sales Distribution")
        fig = px.histogram(
            data,
            x='Global_Sales',
            nbins=50,
            title="Distribution of Global Sales",
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Top Performers by Category")
        category = st.selectbox(
            "Select category:",
            ['Platform', 'Genre', 'Publisher'] if all(col in data.columns for col in ['Platform', 'Genre', 'Publisher']) 
            else [col for col in ['Platform', 'Genre', 'Publisher'] if col in data.columns]
        )
        
        if category:
            top_category = data.groupby(category)['Global_Sales'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=top_category.values,
                y=top_category.index,
                orientation='h',
                title=f"Top 10 {category} by Sales",
                color=top_category.values
            )
            st.plotly_chart(fig, use_container_width=True)

def show_market_analysis_dashboard(data):
    """Show market analysis dashboard."""
    st.header("🌍 Market Analysis Dashboard")
    
    # Regional analysis
    regional_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    available_regional = [col for col in regional_cols if col in data.columns]
    
    if available_regional:
        st.subheader("🌎 Regional Market Analysis")
        
        # Regional totals
        regional_totals = {col.replace('_Sales', ''): data[col].sum() for col in available_regional}
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional pie chart
            fig = px.pie(
                values=list(regional_totals.values()),
                names=list(regional_totals.keys()),
                title="Market Share by Region"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Regional comparison by genre
            if 'Genre' in data.columns:
                regional_genre = data.groupby('Genre')[available_regional].sum()
                fig = px.bar(
                    regional_genre,
                    title="Regional Sales by Genre",
                    barmode='stack'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    # Market concentration analysis
    st.subheader("📊 Market Concentration Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Platform concentration
        if 'Platform' in data.columns and 'Global_Sales' in data.columns:
            platform_sales = data.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)
            total_sales = platform_sales.sum()
            
            # Calculate market share
            market_share = (platform_sales / total_sales * 100).round(2)
            
            fig = px.bar(
                x=market_share.head(10).index,
                y=market_share.head(10).values,
                title="Platform Market Share (%)",
                color=market_share.head(10).values
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Genre concentration
        if 'Genre' in data.columns and 'Global_Sales' in data.columns:
            genre_sales = data.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
            total_sales = genre_sales.sum()
            
            # Calculate market share
            market_share = (genre_sales / total_sales * 100).round(2)
            
            fig = px.bar(
                x=market_share.index,
                y=market_share.values,
                title="Genre Market Share (%)",
                color=market_share.values
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Competitive landscape
    if 'Publisher' in data.columns and 'Global_Sales' in data.columns:
        st.subheader("🏆 Competitive Landscape")
        
        publisher_metrics = data.groupby('Publisher').agg({
            'Global_Sales': ['sum', 'count', 'mean']
        }).reset_index()
        
        publisher_metrics.columns = ['Publisher', 'Total_Sales', 'Game_Count', 'Avg_Sales']
        publisher_metrics = publisher_metrics.sort_values('Total_Sales', ascending=False).head(20)
        
        # Bubble chart
        fig = px.scatter(
            publisher_metrics,
            x='Game_Count',
            y='Avg_Sales',
            size='Total_Sales',
            hover_name='Publisher',
            title="Publisher Performance: Game Count vs Average Sales (Bubble size = Total Sales)",
            labels={'Game_Count': 'Number of Games', 'Avg_Sales': 'Average Sales per Game'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_temporal_trends_dashboard(data):
    """Show temporal trends dashboard."""
    st.header("📅 Temporal Trends Dashboard")
    
    if 'Year' not in data.columns:
        st.error("Year data not available for temporal analysis.")
        return
    
    # Time series analysis
    fig = create_time_series_analysis(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Decade analysis
    st.subheader("📊 Analysis by Decade")
    
    # Create decade column
    data_with_decade = data.copy()
    data_with_decade['Decade'] = (data_with_decade['Year'] // 10) * 10
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Games per decade
        games_per_decade = data_with_decade.groupby('Decade').size()
        fig = px.bar(
            x=games_per_decade.index.astype(str) + 's',
            y=games_per_decade.values,
            title="Number of Games Released by Decade",
            color=games_per_decade.values
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales per decade
        if 'Global_Sales' in data.columns:
            sales_per_decade = data_with_decade.groupby('Decade')['Global_Sales'].sum()
            fig = px.bar(
                x=sales_per_decade.index.astype(str) + 's',
                y=sales_per_decade.values,
                title="Total Sales by Decade",
                color=sales_per_decade.values
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Evolution of genres over time
    if 'Genre' in data.columns:
        st.subheader("🎮 Genre Evolution Over Time")
        
        # Top genres by year
        genre_yearly = data_with_decade.groupby(['Year', 'Genre']).size().reset_index(name='Count')
        top_genres = data['Genre'].value_counts().head(5).index.tolist()
        
        genre_yearly_filtered = genre_yearly[genre_yearly['Genre'].isin(top_genres)]
        
        fig = px.line(
            genre_yearly_filtered,
            x='Year',
            y='Count',
            color='Genre',
            title="Evolution of Top 5 Genres Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_custom_analysis_dashboard(data):
    """Show customizable analysis dashboard."""
    st.header("🔧 Custom Analysis Dashboard")
    
    st.subheader("📊 Create Custom Visualizations")
    
    # Chart type selection
    chart_type = st.selectbox(
        "Select Chart Type:",
        ["Scatter Plot", "Bar Chart", "Line Chart", "Box Plot", "Histogram", "Heatmap"]
    )
    
    # Column selection based on chart type
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    all_cols = numeric_cols + categorical_cols
    
    col1, col2 = st.columns(2)
    
    if chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("X-axis:", numeric_cols)
            y_col = st.selectbox("Y-axis:", [col for col in numeric_cols if col != x_col])
        with col2:
            size_col = st.selectbox("Size (optional):", [None] + numeric_cols)
            color_col = st.selectbox("Color (optional):", [None] + all_cols)
        
        if x_col and y_col:
            fig = create_scatter_analysis(data, x_col, y_col, size_col, color_col)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("X-axis:", all_cols)
            y_col = st.selectbox("Y-axis:", numeric_cols)
        with col2:
            color_col = st.selectbox("Color (optional):", [None] + categorical_cols)
            top_n = st.slider("Show top N values:", 5, 50, 10)
        
        if x_col and y_col:
            # Aggregate data
            if x_col in categorical_cols:
                agg_data = data.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(top_n)
                fig = px.bar(
                    x=agg_data.index,
                    y=agg_data.values,
                    title=f"{y_col} by {x_col}",
                    color=agg_data.values if not color_col else None
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Line Chart":
        with col1:
            x_col = st.selectbox("X-axis:", numeric_cols)
            y_col = st.selectbox("Y-axis:", numeric_cols)
        with col2:
            color_col = st.selectbox("Group by (optional):", [None] + categorical_cols)
        
        if x_col and y_col:
            if color_col:
                # Group by categorical variable
                grouped_data = data.groupby([x_col, color_col])[y_col].mean().reset_index()
                fig = px.line(
                    grouped_data,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{y_col} vs {x_col} by {color_col}"
                )
            else:
                # Simple aggregation
                agg_data = data.groupby(x_col)[y_col].mean()
                fig = px.line(
                    x=agg_data.index,
                    y=agg_data.values,
                    title=f"{y_col} vs {x_col}",
                    markers=True
                )
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Box Plot":
        with col1:
            y_col = st.selectbox("Y-axis (numeric):", numeric_cols)
        with col2:
            x_col = st.selectbox("X-axis (categorical, optional):", [None] + categorical_cols)
        
        if y_col:
            if x_col:
                fig = px.box(data, x=x_col, y=y_col, title=f"Distribution of {y_col} by {x_col}")
            else:
                fig = px.box(data, y=y_col, title=f"Distribution of {y_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Histogram":
        with col1:
            x_col = st.selectbox("Column:", numeric_cols)
        with col2:
            bins = st.slider("Number of bins:", 10, 100, 30)
            color_col = st.selectbox("Color by (optional):", [None] + categorical_cols)
        
        if x_col:
            fig = px.histogram(
                data,
                x=x_col,
                nbins=bins,
                color=color_col,
                title=f"Distribution of {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Data export for custom analysis
    st.subheader("📥 Export Filtered Data")
    
    if st.button("💾 Export Current View"):
        csv_data = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv_data,
            file_name="filtered_video_game_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
