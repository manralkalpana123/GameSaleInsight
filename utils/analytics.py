import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import streamlit as st

def get_summary_stats(data: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive summary statistics for the dataset."""
    summary = {
        'basic_info': {
            'total_records': len(data),
            'total_columns': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024**2),
            'duplicate_records': data.duplicated().sum()
        },
        'missing_data': {},
        'numeric_summary': {},
        'categorical_summary': {}
    }
    
    # Missing data analysis
    missing_counts = data.isnull().sum()
    summary['missing_data'] = {
        col: {
            'count': int(missing_counts[col]),
            'percentage': round((missing_counts[col] / len(data)) * 100, 2)
        }
        for col in data.columns if missing_counts[col] > 0
    }
    
    # Numeric columns analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'count': int(data[col].count()),
            'mean': round(data[col].mean(), 4),
            'median': round(data[col].median(), 4),
            'std': round(data[col].std(), 4),
            'min': round(data[col].min(), 4),
            'max': round(data[col].max(), 4),
            'q25': round(data[col].quantile(0.25), 4),
            'q75': round(data[col].quantile(0.75), 4),
            'skewness': round(data[col].skew(), 4),
            'kurtosis': round(data[col].kurtosis(), 4)
        }
    
    # Categorical columns analysis
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        value_counts = data[col].value_counts()
        summary['categorical_summary'][col] = {
            'unique_count': int(data[col].nunique()),
            'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else 'N/A',
            'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'top_5_values': value_counts.head().to_dict()
        }
    
    return summary

def perform_sales_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Perform specific analysis on video game sales data."""
    analysis = {}
    
    # Global sales analysis
    if 'Global_Sales' in data.columns:
        analysis['global_sales'] = {
            'total_sales': round(data['Global_Sales'].sum(), 2),
            'average_sales': round(data['Global_Sales'].mean(), 4),
            'top_selling_game': get_top_item(data, 'Global_Sales', 'Name'),
            'sales_distribution': get_distribution_stats(data['Global_Sales'])
        }
    
    # Platform analysis
    if 'Platform' in data.columns:
        platform_stats = analyze_by_category(data, 'Platform', 'Global_Sales')
        analysis['platform_analysis'] = platform_stats
    
    # Genre analysis
    if 'Genre' in data.columns:
        genre_stats = analyze_by_category(data, 'Genre', 'Global_Sales')
        analysis['genre_analysis'] = genre_stats
    
    # Publisher analysis
    if 'Publisher' in data.columns:
        publisher_stats = analyze_by_category(data, 'Publisher', 'Global_Sales')
        analysis['publisher_analysis'] = publisher_stats
    
    # Year analysis
    if 'Year' in data.columns:
        year_stats = analyze_temporal_trends(data)
        analysis['temporal_analysis'] = year_stats
    
    # Regional analysis
    regional_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    available_regional = [col for col in regional_cols if col in data.columns]
    if available_regional:
        analysis['regional_analysis'] = analyze_regional_sales(data, available_regional)
    
    return analysis

def analyze_by_category(data: pd.DataFrame, category_col: str, sales_col: str) -> Dict[str, Any]:
    """Analyze sales performance by category (Platform, Genre, Publisher, etc.)."""
    if category_col not in data.columns or sales_col not in data.columns:
        return {}
    
    category_stats = data.groupby(category_col)[sales_col].agg([
        'sum', 'mean', 'median', 'count', 'std'
    ]).reset_index()
    
    category_stats.columns = [category_col, 'total_sales', 'avg_sales', 'median_sales', 'game_count', 'std_sales']
    category_stats = category_stats.sort_values('total_sales', ascending=False)
    
    return {
        'top_performer': {
            'name': str(category_stats.iloc[0][category_col]),
            'total_sales': round(category_stats.iloc[0]['total_sales'], 2),
            'game_count': int(category_stats.iloc[0]['game_count']),
            'avg_sales': round(category_stats.iloc[0]['avg_sales'], 4)
        },
        'summary_stats': {
            'total_categories': len(category_stats),
            'avg_sales_per_category': round(category_stats['avg_sales'].mean(), 4),
            'most_productive': {
                'name': str(category_stats.loc[category_stats['game_count'].idxmax()][category_col]),
                'game_count': int(category_stats['game_count'].max())
            }
        },
        'top_5': category_stats.head().to_dict('records')
    }

def analyze_temporal_trends(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze trends over time."""
    if 'Year' not in data.columns:
        return {}
    
    # Filter reasonable years
    data_filtered = data[(data['Year'] >= 1980) & (data['Year'] <= 2025)]
    
    yearly_stats = data_filtered.groupby('Year').agg({
        'Global_Sales': ['sum', 'mean', 'count'] if 'Global_Sales' in data.columns else ['count']
    }).reset_index()
    
    # Flatten column names
    yearly_stats.columns = ['Year'] + ['_'.join(col).strip('_') for col in yearly_stats.columns[1:]]
    
    # Find peak years
    analysis = {
        'year_range': {
            'start': int(yearly_stats['Year'].min()),
            'end': int(yearly_stats['Year'].max()),
            'span': int(yearly_stats['Year'].max() - yearly_stats['Year'].min())
        }
    }
    
    if 'Global_Sales_sum' in yearly_stats.columns:
        peak_sales_year = yearly_stats.loc[yearly_stats['Global_Sales_sum'].idxmax()]
        analysis['peak_sales_year'] = {
            'year': int(peak_sales_year['Year']),
            'total_sales': round(peak_sales_year['Global_Sales_sum'], 2)
        }
    
    if 'Global_Sales_count' in yearly_stats.columns:
        peak_release_year = yearly_stats.loc[yearly_stats['Global_Sales_count'].idxmax()]
        analysis['peak_release_year'] = {
            'year': int(peak_release_year['Year']),
            'game_count': int(peak_release_year['Global_Sales_count'])
        }
    
    return analysis

def analyze_regional_sales(data: pd.DataFrame, regional_cols: List[str]) -> Dict[str, Any]:
    """Analyze regional sales distribution."""
    regional_totals = {}
    
    for col in regional_cols:
        if col in data.columns:
            regional_totals[col] = data[col].sum()
    
    total_regional_sales = sum(regional_totals.values())
    
    analysis = {
        'total_regional_sales': round(total_regional_sales, 2),
        'regional_breakdown': {
            region: {
                'total_sales': round(sales, 2),
                'percentage': round((sales / total_regional_sales) * 100, 2) if total_regional_sales > 0 else 0
            }
            for region, sales in regional_totals.items()
        },
        'dominant_region': max(regional_totals.items(), key=lambda x: x[1]) if regional_totals else ('N/A', 0)
    }
    
    return analysis

def get_top_item(data: pd.DataFrame, sort_col: str, name_col: str, n: int = 1) -> str:
    """Get the top item based on a sorting column."""
    if sort_col not in data.columns:
        return 'N/A'
    
    if name_col in data.columns:
        top_items = data.nlargest(n, sort_col)[name_col].tolist()
        return top_items[0] if top_items else 'N/A'
    else:
        return f"Index: {data[sort_col].idxmax()}"

def get_distribution_stats(series: pd.Series) -> Dict[str, float]:
    """Get distribution statistics for a numeric series."""
    return {
        'percentile_90': round(series.quantile(0.9), 4),
        'percentile_95': round(series.quantile(0.95), 4),
        'percentile_99': round(series.quantile(0.99), 4),
        'outlier_threshold_upper': round(series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25)), 4),
        'outlier_count': int(sum(series > (series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25)))))
    }

def identify_trends_and_insights(data: pd.DataFrame) -> List[str]:
    """Identify key trends and insights from the data."""
    insights = []
    
    try:
        # Sales insights
        if 'Global_Sales' in data.columns:
            total_sales = data['Global_Sales'].sum()
            avg_sales = data['Global_Sales'].mean()
            insights.append(f"Total global sales across all games: {total_sales:.2f} million units")
            insights.append(f"Average sales per game: {avg_sales:.2f} million units")
            
            # Top performing game
            if 'Name' in data.columns:
                top_game = data.loc[data['Global_Sales'].idxmax()]
                insights.append(f"Best-selling game: '{top_game['Name']}' with {top_game['Global_Sales']:.2f} million units")
        
        # Platform insights
        if 'Platform' in data.columns and 'Global_Sales' in data.columns:
            platform_sales = data.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)
            top_platform = platform_sales.index[0]
            insights.append(f"Most successful platform: {top_platform} with {platform_sales.iloc[0]:.2f} million total sales")
        
        # Genre insights
        if 'Genre' in data.columns and 'Global_Sales' in data.columns:
            genre_sales = data.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
            top_genre = genre_sales.index[0]
            insights.append(f"Most successful genre: {top_genre} with {genre_sales.iloc[0]:.2f} million total sales")
        
        # Temporal insights
        if 'Year' in data.columns and 'Global_Sales' in data.columns:
            yearly_sales = data.groupby('Year')['Global_Sales'].sum()
            peak_year = yearly_sales.idxmax()
            insights.append(f"Peak sales year: {peak_year} with {yearly_sales.max():.2f} million units sold")
        
        # Regional insights
        regional_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        available_regional = [col for col in regional_cols if col in data.columns]
        if available_regional:
            regional_totals = {col: data[col].sum() for col in available_regional}
            dominant_region = max(regional_totals.items(), key=lambda x: x[1])
            insights.append(f"Dominant market: {dominant_region[0].replace('_Sales', '')} with {dominant_region[1]:.2f} million total sales")
        
    except Exception as e:
        insights.append(f"Error generating insights: {str(e)}")
    
    return insights

def generate_recommendations(data: pd.DataFrame) -> List[str]:
    """Generate business recommendations based on data analysis."""
    recommendations = []
    
    try:
        # Platform recommendations
        if 'Platform' in data.columns and 'Global_Sales' in data.columns:
            platform_performance = data.groupby('Platform').agg({
                'Global_Sales': ['mean', 'count']
            }).reset_index()
            platform_performance.columns = ['Platform', 'avg_sales', 'game_count']
            
            # Find underperforming platforms with high game counts
            underperforming = platform_performance[
                (platform_performance['avg_sales'] < platform_performance['avg_sales'].median()) &
                (platform_performance['game_count'] > platform_performance['game_count'].median())
            ]
            
            if not underperforming.empty:
                recommendations.append(
                    f"Consider focusing on quality over quantity for platforms like {', '.join(underperforming['Platform'].head(3).tolist())} "
                    "which have many games but lower average sales."
                )
        
        # Genre recommendations
        if 'Genre' in data.columns and 'Global_Sales' in data.columns:
            genre_stats = data.groupby('Genre')['Global_Sales'].agg(['mean', 'std']).reset_index()
            high_potential_genres = genre_stats[
                (genre_stats['mean'] > genre_stats['mean'].quantile(0.75)) &
                (genre_stats['std'] < genre_stats['std'].median())
            ]
            
            if not high_potential_genres.empty:
                recommendations.append(
                    f"Focus on {', '.join(high_potential_genres['Genre'].tolist())} genres "
                    "which show high average sales with consistent performance."
                )
        
        # Temporal recommendations
        if 'Year' in data.columns and 'Global_Sales' in data.columns:
            recent_years = data[data['Year'] >= (data['Year'].max() - 5)]
            recent_avg = recent_years['Global_Sales'].mean()
            overall_avg = data['Global_Sales'].mean()
            
            if recent_avg < overall_avg:
                recommendations.append(
                    "Recent years show declining average sales. Consider market research to understand "
                    "changing consumer preferences and adapt strategies accordingly."
                )
        
        # Regional recommendations
        regional_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        available_regional = [col for col in regional_cols if col in data.columns]
        
        if len(available_regional) > 1:
            regional_totals = {col: data[col].sum() for col in available_regional}
            max_region = max(regional_totals.items(), key=lambda x: x[1])
            min_region = min(regional_totals.items(), key=lambda x: x[1])
            
            if max_region[1] > min_region[1] * 3:  # If one region dominates significantly
                recommendations.append(
                    f"Consider expanding marketing efforts in {min_region[0].replace('_Sales', '')} market, "
                    f"which shows potential for growth compared to the dominant {max_region[0].replace('_Sales', '')} market."
                )
        
    except Exception as e:
        recommendations.append(f"Error generating recommendations: {str(e)}")
    
    return recommendations if recommendations else ["Upload and analyze data to get personalized recommendations."]
