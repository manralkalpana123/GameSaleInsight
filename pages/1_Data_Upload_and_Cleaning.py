import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Upload & Cleaning", page_icon="📊", layout="wide")

def main():
    st.title("📊 Data Upload & Cleaning")
    st.markdown("Upload your video game sales dataset and perform data cleaning operations.")
    
    # Initialize data processor
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    
    processor = st.session_state.data_processor
    
    # File upload section
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing video game sales data",
        type="csv",
        help="Expected columns: Name, Platform, Year, Genre, Publisher, Global_Sales, NA_Sales, EU_Sales, JP_Sales, Other_Sales"
    )
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading data..."):
            data = processor.load_data(uploaded_file)
        
        if data is not None:
            st.success(f"Data loaded successfully! Shape: {data.shape}")
            
            # Store original data in session state
            st.session_state.original_data = data.copy()
            
            # Data preview
            st.subheader("Data Preview")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.dataframe(data.head(10), use_container_width=True)
            
            with col2:
                st.metric("Total Rows", len(data))
                st.metric("Total Columns", len(data.columns))
                st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            
            # Data quality analysis
            st.header("2. Data Quality Analysis")
            quality_report = processor.analyze_data_quality()
            
            if quality_report:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Missing Values")
                    missing_df = pd.DataFrame([
                        {"Column": col, "Missing Count": count, "Percentage": f"{(count/len(data)*100):.2f}%"}
                        for col, count in quality_report['missing_values'].items() if count > 0
                    ])
                    
                    if not missing_df.empty:
                        st.dataframe(missing_df, use_container_width=True)
                        
                        # Missing values visualization
                        fig = px.bar(
                            missing_df, 
                            x="Column", 
                            y="Missing Count",
                            title="Missing Values by Column"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("No missing values found!")
                
                with col2:
                    st.subheader("Data Types")
                    dtype_df = pd.DataFrame([
                        {"Column": col, "Data Type": str(dtype)}
                        for col, dtype in quality_report['data_types'].items()
                    ])
                    st.dataframe(dtype_df, use_container_width=True)
                    
                    # Data types distribution
                    type_counts = dtype_df['Data Type'].value_counts()
                    fig = px.pie(
                        values=type_counts.values,
                        names=type_counts.index,
                        title="Data Types Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    st.subheader("Summary Statistics")
                    st.metric("Duplicate Rows", quality_report['duplicate_rows'])
                    st.metric("Numeric Columns", len(quality_report['numeric_columns']))
                    st.metric("Categorical Columns", len(quality_report['categorical_columns']))
            
            # Column information
            st.subheader("Detailed Column Information")
            column_info = processor.get_column_info()
            
            if column_info:
                for col_name, info in column_info.items():
                    with st.expander(f"📋 {col_name} ({info['dtype']})"):
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            st.write(f"**Non-null values:** {info['non_null_count']:,}")
                            st.write(f"**Null values:** {info['null_count']:,}")
                            st.write(f"**Unique values:** {info['unique_values']:,}")
                            st.write(f"**Memory usage:** {info['memory_usage']/1024:.2f} KB")
                        
                        with col_detail2:
                            if 'min' in info:  # Numeric column
                                st.write(f"**Min:** {info['min']:.4f}")
                                st.write(f"**Max:** {info['max']:.4f}")
                                st.write(f"**Mean:** {info['mean']:.4f}")
                                st.write(f"**Median:** {info['median']:.4f}")
                                st.write(f"**Std Dev:** {info['std']:.4f}")
                            elif 'top_values' in info:  # Categorical column
                                st.write("**Top values:**")
                                for value, count in list(info['top_values'].items())[:5]:
                                    st.write(f"  • {value}: {count}")
            
            # Data cleaning section
            st.header("3. Data Cleaning Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cleaning Configuration")
                
                cleaning_options = {}
                
                # Remove duplicates
                cleaning_options['remove_duplicates'] = st.checkbox(
                    "Remove duplicate rows",
                    help="Remove identical rows from the dataset"
                )
                
                # Missing value strategy
                missing_strategy = st.selectbox(
                    "Missing values strategy",
                    ["drop", "fill_numeric", "fill_categorical", "keep"],
                    help="Choose how to handle missing values"
                )
                cleaning_options['missing_strategy'] = missing_strategy
                
                # Type conversion
                cleaning_options['convert_types'] = st.checkbox(
                    "Convert data types",
                    help="Automatically convert columns to appropriate data types"
                )
                
                # Remove outliers
                cleaning_options['remove_outliers'] = st.checkbox(
                    "Remove outliers",
                    help="Remove statistical outliers using IQR method"
                )
                
                # Standardize text
                cleaning_options['standardize_text'] = st.checkbox(
                    "Standardize text columns",
                    help="Trim whitespace and standardize capitalization"
                )
            
            with col2:
                st.subheader("Cleaning Preview")
                
                if st.button("🧹 Clean Data", type="primary"):
                    with st.spinner("Cleaning data..."):
                        cleaned_data = processor.clean_data(cleaning_options)
                    
                    if cleaned_data is not None:
                        st.success("Data cleaned successfully!")
                        
                        # Store cleaned data in session state
                        st.session_state.processed_data = cleaned_data
                        
                        # Show cleaning report
                        cleaning_report = processor.get_cleaning_report()
                        if cleaning_report['steps_applied']:
                            st.subheader("Cleaning Steps Applied:")
                            for step in cleaning_report['steps_applied']:
                                st.write(f"✅ {step}")
                        
                        # Before/After comparison
                        st.subheader("Before vs After Cleaning")
                        comparison_df = pd.DataFrame({
                            'Metric': ['Total Rows', 'Total Columns', 'Missing Values', 'Duplicates'],
                            'Before': [
                                len(st.session_state.original_data),
                                len(st.session_state.original_data.columns),
                                st.session_state.original_data.isnull().sum().sum(),
                                st.session_state.original_data.duplicated().sum()
                            ],
                            'After': [
                                len(cleaned_data),
                                len(cleaned_data.columns),
                                cleaned_data.isnull().sum().sum(),
                                cleaned_data.duplicated().sum()
                            ]
                        })
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Cleaned data preview
                        st.subheader("Cleaned Data Preview")
                        st.dataframe(cleaned_data.head(10), use_container_width=True)
            
            # Export cleaned data
            if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
                st.header("4. Export Cleaned Data")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Download Options")
                    
                    # Generate export data
                    export_data = processor.export_data(st.session_state.processed_data)
                    
                    st.download_button(
                        label="📥 Download Cleaned Data (CSV)",
                        data=export_data,
                        file_name="cleaned_video_game_sales.csv",
                        mime="text/csv",
                        help="Download the cleaned dataset as CSV file"
                    )
                
                with col2:
                    st.subheader("Next Steps")
                    st.info("""
                    **Your data is ready for analysis!**
                    
                    Navigate to other pages to:
                    • **Exploratory Analysis**: Statistical insights
                    • **Interactive Dashboard**: Dynamic visualizations  
                    • **Regional Analysis**: Geographic breakdowns
                    """)
    
    else:
        # Show data format help when no file is uploaded
        st.info("👆 Please upload a CSV file to begin data cleaning")
        
        st.subheader("Expected Data Format")
        st.markdown("""
        Your CSV file should contain video game sales data with columns such as:
        
        | Column | Description | Example |
        |--------|-------------|---------|
        | **Name** | Game title | "Super Mario Bros." |
        | **Platform** | Gaming platform | "PS4", "Xbox One", "PC" |
        | **Year** | Release year | 2020 |
        | **Genre** | Game genre | "Action", "Sports", "RPG" |
        | **Publisher** | Game publisher | "Nintendo", "Sony" |
        | **Global_Sales** | Global sales (millions) | 15.75 |
        | **NA_Sales** | North America sales | 7.25 |
        | **EU_Sales** | Europe sales | 5.50 |
        | **JP_Sales** | Japan sales | 2.00 |
        | **Other_Sales** | Other regions sales | 1.00 |
        """)
        
        st.subheader("Sample Data Structure")
        sample_data = pd.DataFrame({
            'Name': ['Super Mario Bros.', 'Tetris', 'Grand Theft Auto V'],
            'Platform': ['NES', 'GB', 'PS4'],
            'Year': [1985, 1989, 2013],
            'Genre': ['Platform', 'Puzzle', 'Action'],
            'Publisher': ['Nintendo', 'Nintendo', 'Take-Two Interactive'],
            'Global_Sales': [40.24, 30.26, 20.32],
            'NA_Sales': [29.08, 23.20, 7.01],
            'EU_Sales': [3.58, 2.26, 9.27],
            'JP_Sales': [6.81, 4.22, 0.97],
            'Other_Sales': [0.77, 0.58, 3.07]
        })
        st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()
