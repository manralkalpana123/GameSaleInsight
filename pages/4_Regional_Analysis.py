import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Regional Analysis", page_icon="🌍", layout="wide")

def main():
    st.title("🌍 Regional Analysis")
    st.markdown("Comprehensive analysis of video game sales across different geographic regions.")
    
    # Check if data exists
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("⚠️ No data found. Please upload and clean your data first in the 'Data Upload & Cleaning' page.")
        st.stop()
    
    data = st.session_state.processed_data
    
    # Check for regional columns
    regional_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    available_regional = [col for col in regional_cols if col in data.columns]
    
    if not available_regional:
        st.error("❌ No regional sales data found in the dataset.")
        st.info("Expected columns: NA_Sales, EU_Sales, JP_Sales, Other_Sales")
        return
    
    st.success(f"✅ Found regional data for: {', '.join([col.replace('_Sales', '') for col in available_regional])}")
    
    # Sidebar filters
    st.sidebar.header("🎛️ Regional Filters")
    filtered_data = apply_regional_filters(data)
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Regional Overview", "Market Share Analysis", "Regional Preferences", "Comparative Analysis", "Geographic Trends"]
    )
    
    # Display analysis based on selection
    if analysis_type == "Regional Overview":
        show_regional_overview(filtered_data, available_regional)
    elif analysis_type == "Market Share Analysis":
        show_market_share_analysis(filtered_data, available_regional)
    elif analysis_type == "Regional Preferences":
        show_regional_preferences(filtered_data, available_regional)
    elif analysis_type == "Comparative Analysis":
        show_comparative_analysis(filtered_data, available_regional)
    elif analysis_type == "Geographic Trends":
        show_geographic_trends(filtered_data, available_regional)

def apply_regional_filters(data):
    """Apply regional-specific filters."""
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
            default=platforms[:10] if len(platforms) > 10 else platforms
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
    
    return filtered_data

def show_regional_overview(data, regional_cols):
    """Show comprehensive regional overview."""
    st.header("🌎 Regional Sales Overview")
    
    # Calculate regional totals
    regional_totals = {}
    for col in regional_cols:
        regional_totals[col.replace('_Sales', '')] = data[col].sum()
    
    total_regional_sales = sum(regional_totals.values())
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    regions = list(regional_totals.keys())
    values = list(regional_totals.values())
    
    for i, (region, value) in enumerate(regional_totals.items()):
        with [col1, col2, col3, col4][i % 4]:
            percentage = (value / total_regional_sales * 100) if total_regional_sales > 0 else 0
            st.metric(
                f"{region} Sales",
                f"{value:.2f}M",
                f"{percentage:.1f}% of total"
            )
    
    # Regional distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        st.subheader("📊 Regional Market Share")
        fig = px.pie(
            values=values,
            names=regions,
            title="Sales Distribution by Region",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart
        st.subheader("📈 Regional Sales Comparison")
        fig = px.bar(
            x=regions,
            y=values,
            title="Total Sales by Region",
            color=values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_title="Region", yaxis_title="Sales (Millions)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional statistics table
    st.subheader("📋 Regional Statistics Summary")
    
    regional_stats = []
    for col in regional_cols:
        region = col.replace('_Sales', '')
        stats = {
            'Region': region,
            'Total Sales (M)': f"{data[col].sum():.2f}",
            'Average Sales (M)': f"{data[col].mean():.4f}",
            'Median Sales (M)': f"{data[col].median():.4f}",
            'Max Sales (M)': f"{data[col].max():.2f}",
            'Games with Sales': f"{(data[col] > 0).sum():,}",
            'Market Share (%)': f"{(data[col].sum() / total_regional_sales * 100):.2f}" if total_regional_sales > 0 else "0.00"
        }
        regional_stats.append(stats)
    
    regional_stats_df = pd.DataFrame(regional_stats)
    st.dataframe(regional_stats_df, use_container_width=True)

def show_market_share_analysis(data, regional_cols):
    """Show detailed market share analysis."""
    st.header("📊 Market Share Analysis")
    
    # Market share by genre
    if 'Genre' in data.columns:
        st.subheader("🎮 Market Share by Genre")
        
        genre_regional = data.groupby('Genre')[regional_cols].sum()
        
        # Stacked bar chart
        fig = go.Figure()
        
        regions = [col.replace('_Sales', '') for col in regional_cols]
        colors = px.colors.qualitative.Set1[:len(regions)]
        
        for i, col in enumerate(regional_cols):
            region = col.replace('_Sales', '')
            fig.add_trace(go.Bar(
                name=region,
                x=genre_regional.index,
                y=genre_regional[col],
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            barmode='stack',
            title='Regional Sales by Genre (Stacked)',
            xaxis_title='Genre',
            yaxis_title='Sales (Millions)',
            xaxis={'tickangle': 45}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Market share percentage heatmap
        st.subheader("🔥 Genre Market Share Heatmap")
        genre_regional_pct = genre_regional.div(genre_regional.sum(axis=1), axis=0) * 100
        
        fig = px.imshow(
            genre_regional_pct.T,
            title='Market Share Percentage by Genre and Region',
            labels={'x': 'Genre', 'y': 'Region', 'color': 'Market Share (%)'},
            aspect='auto',
            color_continuous_scale='RdYlBu_r'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Market share by platform
    if 'Platform' in data.columns:
        st.subheader("🎯 Market Share by Platform")
        
        platform_regional = data.groupby('Platform')[regional_cols].sum()
        top_platforms = platform_regional.sum(axis=1).nlargest(15).index
        platform_regional_top = platform_regional.loc[top_platforms]
        
        # Create subplots for each region
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[col.replace('_Sales', '') for col in regional_cols],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, col in enumerate(regional_cols):
            if i < len(positions):
                row, col_pos = positions[i]
                region_data = platform_regional_top[col].sort_values(ascending=False)
                
                fig.add_trace(
                    go.Bar(x=region_data.values, y=region_data.index, orientation='h',
                           name=col.replace('_Sales', ''), showlegend=False),
                    row=row, col=col_pos
                )
        
        fig.update_layout(height=800, title="Top 15 Platforms by Regional Sales")
        st.plotly_chart(fig, use_container_width=True)

def show_regional_preferences(data, regional_cols):
    """Show regional preferences analysis."""
    st.header("🎭 Regional Preferences Analysis")
    
    # Regional preference index
    st.subheader("📈 Regional Preference Index")
    st.info("Values > 1 indicate above-average preference for that region, < 1 indicates below-average preference")
    
    # Calculate preference index for genres
    if 'Genre' in data.columns:
        # Calculate global average share for each genre
        global_genre_share = data.groupby('Genre')['Global_Sales'].sum() / data['Global_Sales'].sum() if 'Global_Sales' in data.columns else None
        
        if global_genre_share is not None:
            genre_regional = data.groupby('Genre')[regional_cols].sum()
            regional_totals = {col: data[col].sum() for col in regional_cols}
            
            preference_index = pd.DataFrame()
            
            for col in regional_cols:
                region = col.replace('_Sales', '')
                region_genre_share = genre_regional[col] / regional_totals[col]
                preference_index[region] = region_genre_share / global_genre_share
            
            # Heatmap of preference index
            fig = px.imshow(
                preference_index.T,
                title='Regional Preference Index by Genre',
                labels={'x': 'Genre', 'y': 'Region', 'color': 'Preference Index'},
                aspect='auto',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=1
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top preferences for each region
            st.subheader("🏆 Top Genre Preferences by Region")
            
            cols = st.columns(len(regional_cols))
            
            for i, col in enumerate(regional_cols):
                region = col.replace('_Sales', '')
                with cols[i]:
                    st.write(f"**{region} Top Preferences:**")
                    top_prefs = preference_index[region].nlargest(5)
                    for genre, index in top_prefs.items():
                        st.write(f"• {genre}: {index:.2f}x")
    
    # Regional bestsellers comparison
    st.subheader("🥇 Regional Bestsellers Comparison")
    
    if 'Name' in data.columns:
        cols = st.columns(len(regional_cols))
        
        for i, col in enumerate(regional_cols):
            region = col.replace('_Sales', '')
            with cols[i]:
                st.write(f"**Top 10 in {region}:**")
                top_games = data.nlargest(10, col)[['Name', col]]
                for idx, (_, row) in enumerate(top_games.iterrows(), 1):
                    st.write(f"{idx}. {row['Name'][:30]}{'...' if len(row['Name']) > 30 else ''}")
                    st.write(f"   {row[col]:.2f}M")

def show_comparative_analysis(data, regional_cols):
    """Show comparative analysis between regions."""
    st.header("⚖️ Comparative Regional Analysis")
    
    # Region selector for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        region1 = st.selectbox("Select first region:", [col.replace('_Sales', '') for col in regional_cols])
    with col2:
        region2 = st.selectbox("Select second region:", 
                              [col.replace('_Sales', '') for col in regional_cols if col.replace('_Sales', '') != region1])
    
    if region1 and region2:
        region1_col = f"{region1}_Sales"
        region2_col = f"{region2}_Sales"
        
        # Correlation analysis
        if region1_col in data.columns and region2_col in data.columns:
            correlation = data[region1_col].corr(data[region2_col])
            
            st.subheader(f"📊 {region1} vs {region2} Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Correlation", f"{correlation:.3f}")
            with col2:
                st.metric(f"{region1} Total", f"{data[region1_col].sum():.2f}M")
            with col3:
                st.metric(f"{region2} Total", f"{data[region2_col].sum():.2f}M")
            
            # Scatter plot comparison
            fig = px.scatter(
                data,
                x=region1_col,
                y=region2_col,
                title=f'{region1} vs {region2} Sales Correlation',
                hover_data=['Name'] if 'Name' in data.columns else None,
                trendline="ols"
            )
            fig.update_xaxes(title=f"{region1} Sales (Millions)")
            fig.update_yaxes(title=f"{region2} Sales (Millions)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance comparison by category
            if 'Genre' in data.columns:
                st.subheader(f"🎮 Genre Performance Comparison: {region1} vs {region2}")
                
                genre_comparison = data.groupby('Genre')[[region1_col, region2_col]].sum()
                genre_comparison['Ratio'] = genre_comparison[region1_col] / genre_comparison[region2_col]
                genre_comparison = genre_comparison.sort_values('Ratio', ascending=False)
                
                # Bar chart comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name=region1,
                    x=genre_comparison.index,
                    y=genre_comparison[region1_col],
                    offsetgroup=1
                ))
                fig.add_trace(go.Bar(
                    name=region2,
                    x=genre_comparison.index,
                    y=genre_comparison[region2_col],
                    offsetgroup=2
                ))
                
                fig.update_layout(
                    barmode='group',
                    title=f'Genre Sales Comparison: {region1} vs {region2}',
                    xaxis_title='Genre',
                    yaxis_title='Sales (Millions)',
                    xaxis={'tickangle': 45}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show genres where each region dominates
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Genres where {region1} dominates:**")
                    region1_strong = genre_comparison[genre_comparison['Ratio'] > 1.5].head(5)
                    for genre, row in region1_strong.iterrows():
                        st.write(f"• {genre}: {row['Ratio']:.2f}x stronger")
                
                with col2:
                    st.write(f"**Genres where {region2} dominates:**")
                    region2_strong = genre_comparison[genre_comparison['Ratio'] < 0.67].tail(5)
                    for genre, row in region2_strong.iterrows():
                        st.write(f"• {genre}: {1/row['Ratio']:.2f}x stronger")

def show_geographic_trends(data, regional_cols):
    """Show geographic trends over time."""
    st.header("📈 Geographic Trends Over Time")
    
    if 'Year' not in data.columns:
        st.error("Year data required for geographic trends analysis.")
        return
    
    # Filter reasonable years
    data_filtered = data[(data['Year'] >= 1980) & (data['Year'] <= 2025)]
    
    # Yearly regional trends
    st.subheader("🌍 Regional Sales Trends Over Time")
    
    yearly_regional = data_filtered.groupby('Year')[regional_cols].sum()
    
    fig = go.Figure()
    
    regions = [col.replace('_Sales', '') for col in regional_cols]
    colors = px.colors.qualitative.Set1[:len(regions)]
    
    for i, col in enumerate(regional_cols):
        region = col.replace('_Sales', '')
        fig.add_trace(go.Scatter(
            x=yearly_regional.index,
            y=yearly_regional[col],
            mode='lines+markers',
            name=region,
            line=dict(color=colors[i], width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='Regional Sales Trends Over Time',
        xaxis_title='Year',
        yaxis_title='Sales (Millions)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Market share evolution
    st.subheader("📊 Market Share Evolution")
    
    yearly_regional_pct = yearly_regional.div(yearly_regional.sum(axis=1), axis=0) * 100
    
    fig = go.Figure()
    
    for i, col in enumerate(regional_cols):
        region = col.replace('_Sales', '')
        fig.add_trace(go.Scatter(
            x=yearly_regional_pct.index,
            y=yearly_regional_pct[col],
            mode='lines+markers',
            name=region,
            stackgroup='one',
            line=dict(width=0),
            fillcolor=colors[i]
        ))
    
    fig.update_layout(
        title='Regional Market Share Evolution (%)',
        xaxis_title='Year',
        yaxis_title='Market Share (%)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional growth analysis
    st.subheader("📈 Regional Growth Analysis")
    
    # Calculate year-over-year growth
    regional_growth = yearly_regional.pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average growth rates
        st.write("**Average Annual Growth Rates:**")
        avg_growth = regional_growth.mean()
        for col in regional_cols:
            region = col.replace('_Sales', '')
            growth = avg_growth[col]
            st.metric(f"{region} Growth", f"{growth:+.2f}%/year")
    
    with col2:
        # Volatility analysis
        st.write("**Growth Volatility (Standard Deviation):**")
        growth_volatility = regional_growth.std()
        for col in regional_cols:
            region = col.replace('_Sales', '')
            volatility = growth_volatility[col]
            st.metric(f"{region} Volatility", f"{volatility:.2f}%")
    
    # Peak years for each region
    st.subheader("🏆 Peak Performance Years by Region")
    
    peak_years = {}
    for col in regional_cols:
        region = col.replace('_Sales', '')
        peak_year = yearly_regional[col].idxmax()
        peak_value = yearly_regional[col].max()
        peak_years[region] = {'year': peak_year, 'sales': peak_value}
    
    cols = st.columns(len(regional_cols))
    for i, (region, info) in enumerate(peak_years.items()):
        with cols[i]:
            st.metric(
                f"{region} Peak Year",
                f"{info['year']}",
                f"{info['sales']:.2f}M sales"
            )

if __name__ == "__main__":
    main()
