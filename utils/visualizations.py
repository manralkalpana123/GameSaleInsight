import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

def create_overview_charts(data):
    """Create overview charts for the main page."""
    try:
        if 'Global_Sales' in data.columns and len(data) > 0:
            # Simple bar chart of top 10 games by sales
            top_games = data.nlargest(10, 'Global_Sales')
            fig = px.bar(
                top_games, 
                x='Global_Sales', 
                y='Name' if 'Name' in data.columns else data.index,
                orientation='h',
                title="Top 10 Games by Global Sales",
                labels={'Global_Sales': 'Sales (Millions)', 'Name': 'Game Title'}
            )
            fig.update_layout(height=400)
            return fig
    except Exception as e:
        st.error(f"Error creating overview chart: {str(e)}")
    return None

def create_sales_trend_chart(data, group_by='Year'):
    """Create sales trend over time."""
    try:
        if group_by in data.columns and 'Global_Sales' in data.columns:
            trend_data = data.groupby(group_by)['Global_Sales'].sum().reset_index()
            
            fig = px.line(
                trend_data, 
                x=group_by, 
                y='Global_Sales',
                title=f'Global Sales Trend by {group_by}',
                labels={'Global_Sales': 'Sales (Millions)'}
            )
            fig.update_traces(mode='lines+markers')
            return fig
    except Exception as e:
        st.error(f"Error creating trend chart: {str(e)}")
    return None

def create_platform_comparison(data):
    """Create platform comparison chart."""
    try:
        if 'Platform' in data.columns and 'Global_Sales' in data.columns:
            platform_data = data.groupby('Platform')['Global_Sales'].agg(['sum', 'count']).reset_index()
            platform_data.columns = ['Platform', 'Total_Sales', 'Game_Count']
            platform_data = platform_data.sort_values('Total_Sales', ascending=False).head(15)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Total Sales by Platform', 'Number of Games by Platform'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Sales chart
            fig.add_trace(
                go.Bar(x=platform_data['Platform'], y=platform_data['Total_Sales'], 
                      name='Total Sales', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Game count chart
            fig.add_trace(
                go.Bar(x=platform_data['Platform'], y=platform_data['Game_Count'], 
                      name='Game Count', marker_color='lightcoral'),
                row=1, col=2
            )
            
            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(tickangle=45)
            return fig
    except Exception as e:
        st.error(f"Error creating platform comparison: {str(e)}")
    return None

def create_genre_analysis(data):
    """Create genre analysis charts."""
    try:
        if 'Genre' in data.columns and 'Global_Sales' in data.columns:
            genre_data = data.groupby('Genre').agg({
                'Global_Sales': ['sum', 'mean', 'count']
            }).reset_index()
            
            genre_data.columns = ['Genre', 'Total_Sales', 'Avg_Sales', 'Game_Count']
            genre_data = genre_data.sort_values('Total_Sales', ascending=False)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Total Sales by Genre', 
                    'Average Sales by Genre',
                    'Number of Games by Genre',
                    'Sales Distribution by Genre'
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "box"}]]
            )
            
            # Total sales
            fig.add_trace(
                go.Bar(x=genre_data['Genre'], y=genre_data['Total_Sales'], 
                      name='Total Sales', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Average sales
            fig.add_trace(
                go.Bar(x=genre_data['Genre'], y=genre_data['Avg_Sales'], 
                      name='Avg Sales', marker_color='lightgreen'),
                row=1, col=2
            )
            
            # Game count
            fig.add_trace(
                go.Bar(x=genre_data['Genre'], y=genre_data['Game_Count'], 
                      name='Game Count', marker_color='lightcoral'),
                row=2, col=1
            )
            
            # Box plot for distribution
            fig.add_trace(
                go.Box(x=data['Genre'], y=data['Global_Sales'], 
                      name='Sales Distribution'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False)
            fig.update_xaxes(tickangle=45)
            return fig
    except Exception as e:
        st.error(f"Error creating genre analysis: {str(e)}")
    return None

def create_regional_heatmap(data):
    """Create regional sales heatmap."""
    try:
        regional_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        available_cols = [col for col in regional_cols if col in data.columns]
        
        if len(available_cols) > 0 and 'Genre' in data.columns:
            regional_data = data.groupby('Genre')[available_cols].sum()
            
            fig = px.imshow(
                regional_data.T,
                title='Regional Sales Heatmap by Genre',
                labels={'x': 'Genre', 'y': 'Region', 'color': 'Sales (Millions)'},
                aspect='auto'
            )
            return fig
    except Exception as e:
        st.error(f"Error creating regional heatmap: {str(e)}")
    return None

def create_publisher_analysis(data, top_n=15):
    """Create publisher analysis chart."""
    try:
        if 'Publisher' in data.columns and 'Global_Sales' in data.columns:
            publisher_data = data.groupby('Publisher').agg({
                'Global_Sales': ['sum', 'count', 'mean']
            }).reset_index()
            
            publisher_data.columns = ['Publisher', 'Total_Sales', 'Game_Count', 'Avg_Sales']
            publisher_data = publisher_data.sort_values('Total_Sales', ascending=False).head(top_n)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Top Publishers by Total Sales', 'Average Sales per Game'),
            )
            
            fig.add_trace(
                go.Bar(x=publisher_data['Total_Sales'], y=publisher_data['Publisher'], 
                      orientation='h', name='Total Sales', marker_color='skyblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=publisher_data['Avg_Sales'], y=publisher_data['Publisher'], 
                      orientation='h', name='Avg Sales', marker_color='lightcoral'),
                row=1, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            return fig
    except Exception as e:
        st.error(f"Error creating publisher analysis: {str(e)}")
    return None

def create_scatter_analysis(data, x_col, y_col, size_col=None, color_col=None):
    """Create customizable scatter plot."""
    try:
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col,
            size=size_col if size_col else None,
            color=color_col if color_col else None,
            hover_data=['Name'] if 'Name' in data.columns else None,
            title=f'{y_col} vs {x_col}'
        )
        return fig
    except Exception as e:
        st.error(f"Error creating scatter plot: {str(e)}")
    return None

def create_time_series_analysis(data):
    """Create comprehensive time series analysis."""
    try:
        if 'Year' in data.columns and 'Global_Sales' in data.columns:
            # Filter out invalid years
            data_filtered = data[(data['Year'] >= 1980) & (data['Year'] <= 2025)]
            
            yearly_stats = data_filtered.groupby('Year').agg({
                'Global_Sales': ['sum', 'mean', 'count']
            }).reset_index()
            
            yearly_stats.columns = ['Year', 'Total_Sales', 'Avg_Sales', 'Game_Count']
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    'Total Global Sales by Year',
                    'Average Sales per Game by Year', 
                    'Number of Games Released by Year'
                ),
                vertical_spacing=0.08
            )
            
            # Total sales
            fig.add_trace(
                go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Total_Sales'],
                          mode='lines+markers', name='Total Sales', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Average sales
            fig.add_trace(
                go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Avg_Sales'],
                          mode='lines+markers', name='Avg Sales', line=dict(color='green')),
                row=2, col=1
            )
            
            # Game count
            fig.add_trace(
                go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Game_Count'],
                          mode='lines+markers', name='Game Count', line=dict(color='red')),
                row=3, col=1
            )
            
            fig.update_layout(height=900, showlegend=False)
            fig.update_xaxes(title_text="Year")
            fig.update_yaxes(title_text="Sales (Millions)", row=1, col=1)
            fig.update_yaxes(title_text="Average Sales", row=2, col=1)
            fig.update_yaxes(title_text="Number of Games", row=3, col=1)
            
            return fig
    except Exception as e:
        st.error(f"Error creating time series analysis: {str(e)}")
    return None

def create_correlation_heatmap(data):
    """Create correlation heatmap for numeric columns."""
    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = data[numeric_cols].corr()
            
            fig = px.imshow(
                correlation_matrix,
                title='Correlation Heatmap of Numeric Variables',
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            return fig
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
    return None
