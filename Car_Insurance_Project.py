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
elif page == 'Correlation Analysis':
    st.title('Correlation Analysis')
    
    corr_method = st.selectbox("Select Correlation Method", ['pearson', 'spearman', 'kendall'])
    color_scheme = st.selectbox("Select Color Scheme", ['coolwarm', 'viridis', 'plasma'])
    
    numeric_df = merged_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method=corr_method)
    
    fig = px.imshow(corr_matrix, color_continuous_scale=color_scheme, title=f'{corr_method.capitalize()} Correlation Heatmap')
    st.plotly_chart(fig)
    
    st.subheader("Top 25 Correlations")
    top_corr = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    top_25_corr = top_corr[top_corr < 1].head(25)
    st.write(top_25_corr)





#Step 7: Category Analysis
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
