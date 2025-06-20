import streamlit as st
import pandas as pd
import numpy as np
from utils.analytics import get_summary_stats, perform_sales_analysis, identify_trends_and_insights, generate_recommendations
from utils.visualizations import create_correlation_heatmap, create_time_series_analysis
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Exploratory Analysis", page_icon="🔍", layout="wide")

def main():
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Comprehensive statistical analysis and insights from your video game sales data.")
    
    # Check if data exists
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("⚠️ No data found. Please upload and clean your data first in the 'Data Upload & Cleaning' page.")
        st.stop()
    
    data = st.session_state.processed_data
    
    # Sidebar for analysis options
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Overview", "Statistical Summary", "Sales Analysis", "Correlation Analysis", "Time Series", "Insights & Recommendations"]
    )
    
    if analysis_type == "Overview":
        show_overview_analysis(data)
    elif analysis_type == "Statistical Summary":
        show_statistical_summary(data)
    elif analysis_type == "Sales Analysis":
        show_sales_analysis(data)
    elif analysis_type == "Correlation Analysis":
        show_correlation_analysis(data)
    elif analysis_type == "Time Series":
        show_time_series_analysis(data)
    elif analysis_type == "Insights & Recommendations":
        show_insights_and_recommendations(data)

def show_overview_analysis(data):
    """Display overview analysis of the dataset."""
    st.header("📊 Dataset Overview")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Total Columns", len(data.columns))
    with col3:
        if 'Global_Sales' in data.columns:
            st.metric("Total Sales (M)", f"{data['Global_Sales'].sum():.2f}")
        else:
            st.metric("Data Types", len(data.dtypes.unique()))
    with col4:
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        st.metric("Data Completeness", f"{100-missing_pct:.1f}%")
    
    # Data distribution
    st.subheader("Data Distribution by Column Type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric columns distribution
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.write("**Numeric Columns:**")
            for col in numeric_cols:
                with st.expander(f"📈 {col}"):
                    col_stats = data[col].describe()
                    st.write(col_stats)
                    
                    # Histogram
                    fig = px.histogram(data, x=col, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Categorical columns distribution
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.write("**Categorical Columns:**")
            for col in categorical_cols[:5]:  # Limit to first 5 for performance
                with st.expander(f"📋 {col}"):
                    value_counts = data[col].value_counts().head(10)
                    st.write(f"**Unique values:** {data[col].nunique()}")
                    st.write("**Top 10 values:**")
                    st.write(value_counts)
                    
                    # Bar chart
                    if len(value_counts) > 0:
                        fig = px.bar(
                            x=value_counts.values,
                            y=value_counts.index,
                            orientation='h',
                            title=f"Top Values in {col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

def show_statistical_summary(data):
    """Display comprehensive statistical summary."""
    st.header("📈 Statistical Summary")
    
    with st.spinner("Generating statistical summary..."):
        summary = get_summary_stats(data)
    
    # Basic information
    st.subheader("Basic Information")
    basic_info = summary['basic_info']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{basic_info['total_records']:,}")
    with col2:
        st.metric("Total Columns", basic_info['total_columns'])
    with col3:
        st.metric("Memory Usage", f"{basic_info['memory_usage_mb']:.2f} MB")
    with col4:
        st.metric("Duplicate Records", basic_info['duplicate_records'])
    
    # Missing data analysis
    if summary['missing_data']:
        st.subheader("Missing Data Analysis")
        missing_df = pd.DataFrame([
            {"Column": col, "Missing Count": info['count'], "Percentage": f"{info['percentage']:.2f}%"}
            for col, info in summary['missing_data'].items()
        ])
        st.dataframe(missing_df, use_container_width=True)
        
        # Missing data visualization
        fig = px.bar(
            missing_df,
            x="Column",
            y="Missing Count",
            title="Missing Values by Column",
            color="Percentage"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing data found!")
    
    # Numeric columns analysis
    if summary['numeric_summary']:
        st.subheader("Numeric Columns Analysis")
        
        numeric_df = pd.DataFrame(summary['numeric_summary']).T
        st.dataframe(numeric_df, use_container_width=True)
        
        # Distribution analysis
        st.subheader("Distribution Analysis")
        selected_numeric = st.selectbox(
            "Select numeric column for detailed analysis:",
            list(summary['numeric_summary'].keys())
        )
        
        if selected_numeric:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    data,
                    x=selected_numeric,
                    title=f"Distribution of {selected_numeric}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    data,
                    y=selected_numeric,
                    title=f"Box Plot of {selected_numeric}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Categorical columns analysis
    if summary['categorical_summary']:
        st.subheader("Categorical Columns Analysis")
        
        for col, info in summary['categorical_summary'].items():
            with st.expander(f"📊 {col} Analysis"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write(f"**Unique Count:** {info['unique_count']}")
                    st.write(f"**Most Frequent:** {info['most_frequent']}")
                    st.write(f"**Frequency:** {info['most_frequent_count']}")
                
                with col2:
                    if info['top_5_values']:
                        top_values_df = pd.DataFrame(
                            list(info['top_5_values'].items()),
                            columns=['Value', 'Count']
                        )
                        fig = px.bar(
                            top_values_df,
                            x='Count',
                            y='Value',
                            orientation='h',
                            title=f"Top 5 Values in {col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

def show_sales_analysis(data):
    """Display sales-specific analysis."""
    st.header("💰 Sales Analysis")
    
    with st.spinner("Analyzing sales data..."):
        sales_analysis = perform_sales_analysis(data)
    
    if not sales_analysis:
        st.warning("Sales analysis requires sales data columns (Global_Sales, etc.)")
        return
    
    # Global sales analysis
    if 'global_sales' in sales_analysis:
        st.subheader("Global Sales Overview")
        global_stats = sales_analysis['global_sales']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sales", f"{global_stats['total_sales']:.2f}M")
        with col2:
            st.metric("Average Sales", f"{global_stats['average_sales']:.4f}M")
        with col3:
            if global_stats['top_selling_game'] != 'N/A':
                st.metric("Top Game", global_stats['top_selling_game'])
    
    # Platform analysis
    if 'platform_analysis' in sales_analysis:
        st.subheader("Platform Performance")
        platform_stats = sales_analysis['platform_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Performer:**")
            st.write(f"Platform: {platform_stats['top_performer']['name']}")
            st.write(f"Total Sales: {platform_stats['top_performer']['total_sales']:.2f}M")
            st.write(f"Game Count: {platform_stats['top_performer']['game_count']}")
            st.write(f"Avg Sales: {platform_stats['top_performer']['avg_sales']:.4f}M")
        
        with col2:
            st.write("**Summary Statistics:**")
            st.write(f"Total Platforms: {platform_stats['summary_stats']['total_categories']}")
            st.write(f"Avg Sales per Platform: {platform_stats['summary_stats']['avg_sales_per_category']:.4f}M")
            st.write(f"Most Productive: {platform_stats['summary_stats']['most_productive']['name']}")
        
        # Top platforms chart
        if platform_stats['top_5']:
            top_platforms_df = pd.DataFrame(platform_stats['top_5'])
            fig = px.bar(
                top_platforms_df,
                x='Platform',
                y='total_sales',
                title='Top 5 Platforms by Total Sales',
                color='avg_sales'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Genre analysis
    if 'genre_analysis' in sales_analysis:
        st.subheader("Genre Performance")
        genre_stats = sales_analysis['genre_analysis']
        
        # Similar structure as platform analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Performing Genre:**")
            st.write(f"Genre: {genre_stats['top_performer']['name']}")
            st.write(f"Total Sales: {genre_stats['top_performer']['total_sales']:.2f}M")
            st.write(f"Game Count: {genre_stats['top_performer']['game_count']}")
        
        with col2:
            if genre_stats['top_5']:
                top_genres_df = pd.DataFrame(genre_stats['top_5'])
                fig = px.pie(
                    top_genres_df,
                    values='total_sales',
                    names='Genre',
                    title='Sales Distribution by Top 5 Genres'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    if 'temporal_analysis' in sales_analysis:
        st.subheader("Temporal Analysis")
        temporal_stats = sales_analysis['temporal_analysis']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Time Span:**")
            st.write(f"From: {temporal_stats['year_range']['start']}")
            st.write(f"To: {temporal_stats['year_range']['end']}")
            st.write(f"Span: {temporal_stats['year_range']['span']} years")
        
        with col2:
            if 'peak_sales_year' in temporal_stats:
                st.write("**Peak Sales Year:**")
                st.write(f"Year: {temporal_stats['peak_sales_year']['year']}")
                st.write(f"Sales: {temporal_stats['peak_sales_year']['total_sales']:.2f}M")
        
        with col3:
            if 'peak_release_year' in temporal_stats:
                st.write("**Peak Release Year:**")
                st.write(f"Year: {temporal_stats['peak_release_year']['year']}")
                st.write(f"Games: {temporal_stats['peak_release_year']['game_count']}")
    
    # Regional analysis
    if 'regional_analysis' in sales_analysis:
        st.subheader("Regional Sales Analysis")
        regional_stats = sales_analysis['regional_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Regional Sales", f"{regional_stats['total_regional_sales']:.2f}M")
            st.write(f"**Dominant Region:** {regional_stats['dominant_region'][0]}")
            st.write(f"**Sales:** {regional_stats['dominant_region'][1]:.2f}M")
        
        with col2:
            # Regional breakdown chart
            regional_data = {
                region.replace('_Sales', ''): info['total_sales']
                for region, info in regional_stats['regional_breakdown'].items()
            }
            
            fig = px.pie(
                values=list(regional_data.values()),
                names=list(regional_data.keys()),
                title='Regional Sales Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)

def show_correlation_analysis(data):
    """Display correlation analysis."""
    st.header("🔗 Correlation Analysis")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Correlation analysis requires at least 2 numeric columns.")
        return
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig = create_correlation_heatmap(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed correlation analysis
    st.subheader("Detailed Correlation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox("Select first variable:", numeric_cols)
    with col2:
        var2 = st.selectbox("Select second variable:", [col for col in numeric_cols if col != var1])
    
    if var1 and var2:
        correlation = data[var1].corr(data[var2])
        st.metric("Correlation Coefficient", f"{correlation:.4f}")
        
        # Interpretation
        if abs(correlation) >= 0.7:
            strength = "Strong"
        elif abs(correlation) >= 0.3:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        direction = "positive" if correlation > 0 else "negative"
        st.write(f"**Interpretation:** {strength} {direction} correlation")
        
        # Scatter plot
        fig = px.scatter(
            data,
            x=var1,
            y=var2,
            title=f"Scatter Plot: {var1} vs {var2}",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_time_series_analysis(data):
    """Display time series analysis."""
    st.header("📅 Time Series Analysis")
    
    if 'Year' not in data.columns:
        st.warning("Time series analysis requires a 'Year' column.")
        return
    
    # Time series chart
    fig = create_time_series_analysis(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional time series insights
    st.subheader("Time Series Insights")
    
    # Filter reasonable years
    data_filtered = data[(data['Year'] >= 1980) & (data['Year'] <= 2025)]
    
    if 'Global_Sales' in data.columns:
        yearly_sales = data_filtered.groupby('Year')['Global_Sales'].sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Peak Sales Year", yearly_sales.idxmax())
            st.metric("Peak Sales", f"{yearly_sales.max():.2f}M")
        
        with col2:
            # Calculate trend
            years = yearly_sales.index.values
            sales = yearly_sales.values
            trend = np.polyfit(years, sales, 1)[0]
            st.metric("Sales Trend", f"{trend:+.2f}M/year")
            
            # Calculate volatility
            volatility = np.std(sales) / np.mean(sales) * 100
            st.metric("Sales Volatility", f"{volatility:.1f}%")
        
        with col3:
            # Recent performance
            recent_years = yearly_sales.tail(5)
            recent_avg = recent_years.mean()
            overall_avg = yearly_sales.mean()
            performance = ((recent_avg - overall_avg) / overall_avg) * 100
            
            st.metric("Recent vs Overall Avg", f"{performance:+.1f}%")

def show_insights_and_recommendations(data):
    """Display insights and recommendations."""
    st.header("💡 Insights & Recommendations")
    
    # Generate insights
    with st.spinner("Generating insights..."):
        insights = identify_trends_and_insights(data)
        recommendations = generate_recommendations(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Key Insights")
        for i, insight in enumerate(insights, 1):
            st.write(f"{i}. {insight}")
    
    with col2:
        st.subheader("📋 Recommendations")
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"{i}. {recommendation}")
    
    # Export insights
    st.subheader("Export Analysis Results")
    
    # Create comprehensive report
    report = {
        'insights': insights,
        'recommendations': recommendations,
        'summary_stats': get_summary_stats(data),
        'sales_analysis': perform_sales_analysis(data)
    }
    
    # Convert to readable format
    report_text = "# Video Game Sales Analysis Report\n\n"
    report_text += "## Key Insights\n"
    for i, insight in enumerate(insights, 1):
        report_text += f"{i}. {insight}\n"
    
    report_text += "\n## Recommendations\n"
    for i, recommendation in enumerate(recommendations, 1):
        report_text += f"{i}. {recommendation}\n"
    
    st.download_button(
        label="📥 Download Analysis Report",
        data=report_text,
        file_name="video_game_sales_analysis_report.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main()
