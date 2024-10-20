import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Set page configuration
st.set_page_config(page_title="Car Insurance Claims Analysis", layout="wide")

# Data loading function
@st.cache_data
def load_data():
    try:
        merged_df = pd.read_csv("car_insurance_merged_data.csv")
        insurance_df = pd.read_csv("df_insurance_data_before_merge.csv")
        return merged_df, insurance_df
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please make sure the CSV files are in the same directory as the app.")
        return None, None

# Load the data
merged_df, insurance_df = load_data()

if merged_df is None or insurance_df is None:
    st.stop()
else:
    st.success("Data loaded successfully!")




# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to", ['Data Overview', 'Data Statistics', 'Data Merging and Missingness', 'EDA', 'Correlation Analysis', 'Category Analysis', 'Slope Analysis'])



#Step 2: Data Overview Page
if page == 'Data Overview':
    st.title('Data Overview')
    
    st.header("Merged Dataset")
    st.write(merged_df.head())
    st.write(f"Shape: {merged_df.shape}")
    
    st.header("Insurance Dataset (Before Merge)")
    st.write(insurance_df.head())
    st.write(f"Shape: {insurance_df.shape}")


#Step 3: Data Statistics Page
elif page == 'Data Statistics':
    st.title('Data Statistics')
    
    dataset = st.radio("Choose Dataset", ['Merged', 'Insurance (Before Merge)'])
    
    if dataset == 'Merged':
        df = merged_df
    else:
        df = insurance_df
    
    st.write(df.describe())
    
    st.subheader("Data Types")
    st.write(df.dtypes)



#Step 4: Data Merging and Missingness Page
elif page == 'Data Merging and Missingness':
    st.title('Data Merging and Missingness')
    
    st.write("Explanation of merging process goes here.")
    
    st.subheader("Missingness in Merged Dataset")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(merged_df.isnull(), cbar=False, cmap='viridis', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Missing Values Count")
    st.write(merged_df.isnull().sum().sort_values(ascending=False))




#Step 5: EDA Page
elif page == 'EDA':
    st.title('Exploratory Data Analysis')
    
    analysis_type = st.selectbox("Choose Analysis Type", ['Univariate', 'Bivariate'])
    
    if analysis_type == 'Univariate':
        variable = st.selectbox("Select Variable", merged_df.columns)
        if merged_df[variable].dtype in ['int64', 'float64']:
            fig = px.histogram(merged_df, x=variable, title=f'Distribution of {variable}')
        else:
            fig = px.bar(merged_df[variable].value_counts(), title=f'Distribution of {variable}')
        st.plotly_chart(fig)
    
    elif analysis_type == 'Bivariate':
        x_var = st.selectbox("Select X variable", merged_df.columns)
        y_var = st.selectbox("Select Y variable", merged_df.columns)
        
        if merged_df[x_var].dtype in ['int64', 'float64'] and merged_df[y_var].dtype in ['int64', 'float64']:
            fig = px.scatter(merged_df, x=x_var, y=y_var, title=f'{x_var} vs {y_var}')
        elif merged_df[x_var].dtype in ['int64', 'float64']:
            fig = px.box(merged_df, x=x_var, y=y_var, title=f'{y_var} by {x_var}')
        elif merged_df[y_var].dtype in ['int64', 'float64']:
            fig = px.box(merged_df, x=x_var, y=y_var, title=f'{y_var} by {x_var}')
        else:
            fig = px.bar(merged_df.groupby(x_var)[y_var].count(), title=f'{y_var} count by {x_var}')
        
        st.plotly_chart(fig)




#Step 6: Correlation Analysis
# Assume 'page' is defined somewhere earlier in your code
elif page == 'Correlation Analysis':
    st.title('Correlation Analysis')

    # Dropdown to select the number of variables
    all_numeric_columns = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_columns = st.multiselect("Select Variables for Correlation Analysis", options=all_numeric_columns, default=all_numeric_columns)

    # Correlation method selection
    col1, col2 = st.columns(2)
    with col1:
        corr_method = st.selectbox("Select Correlation Method", ['pearson', 'spearman', 'kendall'])
    with col2:
        # Use valid continuous color scales for imshow
        color_scheme = st.selectbox("Select Color Scheme", ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'BuGn', 'RdBu'])

    # Filter the DataFrame to only include selected columns
    numeric_df = merged_df[selected_columns]

    # Calculate the correlation matrix for selected columns
    corr_matrix = numeric_df.corr(method=corr_method)

    # Heatmap
    fig_heatmap = px.imshow(corr_matrix, 
                            color_continuous_scale=color_scheme, 
                            title=f'{corr_method.capitalize()} Correlation Heatmap',
                            labels=dict(color="Correlation"),
                            zmin=-1, zmax=1)

    fig_heatmap.update_traces(hovertemplate='X: %{x}<br>Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>')
    fig_heatmap.update_layout(width=800, height=800)

    st.plotly_chart(fig_heatmap)
    
    # Top correlations
    top_corr = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    top_corr = top_corr[top_corr < 1]  # Remove self-correlations
    
    # Top 25 correlations table
    st.subheader("Top 25 Correlations")
    top_25_corr_df = pd.DataFrame(top_corr.head(25)).reset_index()
    top_25_corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
    st.dataframe(top_25_corr_df.style.format({'Correlation': '{:.4f}'}), height=400)
    
    # Top 10 correlations horizontal bar plot
    st.subheader("Top 10 Correlations - Horizontal Bar Plot")
    top_10_corr = top_corr.head(10)

    # Check if top_10_corr is not empty
    if not top_10_corr.empty:
        top_10_corr_df = pd.DataFrame(top_10_corr).reset_index()
        
        # Ensure that the DataFrame has the correct number of columns
        if top_10_corr_df.shape[1] == 2:
            top_10_corr_df.columns = ['Variable Pair', 'Correlation']
            top_10_corr_df['Variable Pair'] = top_10_corr_df['Variable Pair'].apply(lambda x: f"{x[0]} - {x[1]}")
        
            fig_bar = px.bar(top_10_corr_df, 
                             x='Correlation', 
                             y='Variable Pair', 
                             orientation='h',
                             title='Top 10 Correlations',
                             color='Correlation',
                             color_continuous_scale=color_scheme)

            fig_bar.update_traces(texttemplate='%{x:.4f}', textposition='outside')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar)
        else:
            st.warning("Not enough data to display top correlations.")
    else:
        st.warning("No correlations to display.")

    # Download button
    csv = corr_matrix.to_csv(index=True)
    st.download_button(
        label="Download Full Correlation Matrix as CSV",
        data=csv,
        file_name=f"correlation_matrix_{corr_method}.csv",
        mime="text/csv",
    )

# Step 7: Category Analysis
elif page == 'Category Analysis':
    st.title('Category Analysis')
    
    category = st.selectbox("Select Category", ['EDUCATION', 'OCCUPATION', 'CAR_TYPE'])
    metric = st.radio("Select Metric", ['CLM_AMT', 'CLM_FREQ'])
    
    fig = px.bar(insurance_df.groupby(category)[metric].mean().reset_index(), 
                 x=category, y=metric, 
                 title=f'Average {metric} by {category}')
    st.plotly_chart(fig)





#Step 8: Slope Analysis
elif page == 'Slope Analysis':
    st.title('Slope Analysis')
    
    target = st.radio("Select Target Variable", ['CLM_FREQ', 'CLM_AMT'])
    
    # You'll need to implement the calculate_slopes function based on your specific requirements
    # slopes_df = calculate_slopes(clm_freq_pairs if target == 'CLM_FREQ' else clm_amt_pairs, target)
    
    # st.subheader(f"Slope Summary Table for {target}")
    # st.write(slopes_df)
    
    st.write("Slope analysis visualization goes here.")
    # You'll need to implement the slope analysis visualization based on your specific requirements
