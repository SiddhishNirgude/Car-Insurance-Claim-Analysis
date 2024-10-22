import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Set page configuration
st.set_page_config(page_title="Car Insurance Claims Analysis", layout="wide")

# Custom CSS for background color
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #FAF3E0;  /* Light cream/ivory color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Data loading function
@st.cache_data
def load_data():
    try:
        # Load the merged dataset and the step CSV files for missingness visualization
        merged_df = pd.read_csv("car_insurance_merged_data.csv")
        insurance_df = pd.read_csv("df_insurance_data_before_merge.csv")
        real_estate_df = pd.read_csv("reduced_real_estate_data.csv")  # Load real estate data
        step1_df = pd.read_csv("step1_missingness_after_merging.csv")  # Load step 1 CSV
        step2_df = pd.read_csv("step2_missingness_after_cleaning_insurance.csv")  # Load step 2 CSV
        return merged_df, insurance_df, real_estate_df, step1_df, step2_df
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please make sure the CSV files are in the same directory as the app.")
        return None, None, None, None, None

# Load the data
merged_df, insurance_df, real_estate_df, step1_df, step2_df = load_data()

# Check if all data was loaded successfully
if (merged_df is None or insurance_df is None or real_estate_df is None or 
    step1_df is None or step2_df is None):
    st.stop()
else:
    st.success("Data loaded successfully!")

# Display the car insurance illustration
st.image("Imageof-Auto-Insurance.jpg", 
         caption="Car Insurance Illustration", use_column_width=True)

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to", ['Data Overview', 'Data Statistics', 'Data Merging and Missingness', 
                                    'EDA', 'Correlation Analysis', 'Category Analysis', 'Slope Analysis'])


# Step 2: Data Overview Page
if page == 'Data Overview':
    st.title('Data Overview')

    # Display the merged dataset
    st.header("Merged Dataset")
    st.write(merged_df.head())
    st.write(f"Shape: {merged_df.shape}")

    # Dropdown for merged dataset column description
    st.subheader("Column Definitions for Merged Dataset")

    # Create a dictionary of definitions for each column
    column_definitions = {
        "ID": "A unique identifier for each record in the dataset.",
        "KIDSDRIV": "Indicates whether there are kids in the household that drive.",
        "BIRTH": "The birth year of the policyholder.",
        "AGE": "The age of the policyholder.",
        "HOMEKIDS": "Indicates whether there are kids in the household.",
        "YOJ": "Years of Job - the number of years the policyholder has been employed.",
        "INCOME": "The annual income of the policyholder.",
        "PARENT1": "Indicates whether the policyholder is a single parent.",
        "HOME_VAL": "The estimated value of the home owned by the policyholder.",
        "MSTATUS": "Marital status of the policyholder (e.g., single, married).",
        "GENDER": "Gender of the policyholder.",
        "EDUCATION": "The highest level of education attained by the policyholder.",
        "OCCUPATION": "The occupation of the policyholder.",
        "TRAVTIME": "The average travel time to work for the policyholder.",
        "CAR_USE": "Indicates how the policyholder primarily uses their car (e.g., personal, business).",
        "BLUEBOOK": "The car's value according to the Blue Book.",
        "TIF": "Time in force - indicates how long the policy has been active.",
        "CAR_TYPE": "The type of car owned by the policyholder.",
        "RED_CAR": "Indicates whether the car is red.",
        "OLDCLAIM": "Indicates whether the policyholder has made an old claim.",
        "CLM_FREQ": "The frequency of claims made by the policyholder.",
        "REVOKED": "Indicates whether the policy has been revoked.",
        "MVR_PTS": "Points from the Motor Vehicle Record, indicating the policyholder's driving history.",
        "CLM_AMT": "The amount claimed by the policyholder for insurance claims.",
        "CAR_AGE": "The age of the car owned by the policyholder.",
        "CLAIM_FLAG": "Indicates whether a claim has been filed (binary flag).",
        "URBANICITY": "Indicates the urban or rural status of the policyholder's residence.",
        "ZIP_CODE": "The postal code indicating the location of the policyholder.",
        "STATE": "The two-letter code representing the state of the policyholder.",
        "price": "The selling price of a property in the real estate dataset.",
        "house_size": "The size of the house, usually measured in square feet.",
        "acre_lot": "The size of the property lot measured in acres.",
        "CITY": "The city where the policyholder resides."
    }

    # Create a list of columns for the dropdown
    column_options = merged_df.columns.tolist()

    # Dropdown selection
    selected_column = st.selectbox("Select a column to see its definition:", column_options)

    # Display the selected column's definition
    if selected_column in column_definitions:
        st.write(f"**{selected_column}**: {column_definitions[selected_column]}")
    else:
        st.write("Definition not available for the selected column.")
    
    # Display the insurance dataset before merge
    st.header("Insurance Dataset (Before Merge)")
    st.write(insurance_df.head())
    st.write(f"Shape: {insurance_df.shape}")
    
    # Display the real estate dataset
    st.header("Real Estate Dataset")
    st.write(real_estate_df.head())
    st.write(f"Shape: {real_estate_df.shape}")
    
    # Explanation of merging process
    st.subheader("Merging Process Explanation")
    
    st.markdown("""
    ### Goal of Merging the Datasets
    By merging the car insurance claims dataset with real estate data, we aim to explore whether the **socioeconomic status** or **property characteristics** of an area have an influence on car insurance claims. Specifically, we seek to investigate the following questions:
    1. Does living in a **wealthier** area (reflected by higher house prices) affect the **frequency** or **amount** of car insurance claims?
    2. Do properties with **larger house sizes** or **bigger lot sizes** correlate with **higher** or **lower** car insurance claims?
    3. Can we identify any patterns in claims based on the **relationship** between the value of real estate and the **claim amounts**?

    ### Primary Dataset: Car Insurance Claims
    - The car insurance claims dataset serves as the **primary** dataset. It contains information about car insurance claims, such as claim amounts, accident details, and policyholder data.
    
    ### Secondary Dataset: Real Estate Data
    - The real estate dataset contains details about **house prices**, **lot size**, and **house size**. From this dataset, we have extracted the following columns:
      - `price`
      - `house_size`
      - `acre_lot`
    
    ### Merging Criteria
    - The two datasets were merged on the `ZIP_CODE` column using a **left merge**. This ensures that all records from the car insurance dataset are kept, while real estate data is added where available.
    
    ### Challenges Faced During Merging
    1. **Duplicate ZIP Codes**: We checked for duplicate ZIP codes in the merged dataset to avoid duplicating rows. The `ZIP_CODE` field must uniquely identify locations.
    2. **Missing Values**: After merging, there were missing values in some of the real estate fields (`price`, `house_size`, etc.) for ZIP codes that did not have corresponding real estate data.
    3. **Invalid ZIP Codes**: We validated that all ZIP codes in the merged dataset are 5-digit numbers.
    4. **State Mismatch**: We ensured that the `STATE` column in both datasets contained valid two-letter US state codes. In case of conflicts, we kept the state information from the car insurance dataset.
    """)


# Step 3: Data Statistics Page
if page == 'Data Statistics':
    st.title('Data Statistics')

    # Statistics for Merged Dataset
    st.header("Merged Dataset Statistics")
    st.write(merged_df.describe())
    
    # Add any notable statistics or insights about the merged dataset
    st.markdown("""
    ### Key Statistics of the Merged Dataset
    - **Count**: The number of non-missing entries for each column.
    - **Mean**, **Median**, **Standard Deviation**: Basic summary statistics of numerical columns like `CLM_AMT`, `price`, `house_size`, etc.
    - **Missing Values**: After merging, certain columns from the real estate dataset may have missing values for ZIP codes without matching data.
    - **Outliers**: High values for `CLM_AMT` and `price` may indicate outliers that can affect the analysis.
    """)
    
    # Visualize missing data in merged dataset
    st.subheader("Missing Values in Merged Dataset")
    missing_values = merged_df.isnull().sum()
    st.write(missing_values[missing_values > 0])
    
    # Duplicates check
    st.subheader("Duplicate Rows in Merged Dataset")
    duplicates_count = merged_df.duplicated(subset='ZIP_CODE').sum()
    st.write(f"Number of duplicate rows based on ZIP_CODE: {duplicates_count}")
    
    # Insurance Dataset Statistics
    st.header("Car Insurance Dataset Statistics")
    st.write(insurance_df.describe())
    
    st.markdown("""
    ### Key Insights from Car Insurance Dataset
    - **Claim Amount Distribution**: Average and spread of claim amounts provide insights into typical claim sizes.
    - **Policyholder Data**: Examining statistics on age, accident severity, and other policyholder attributes helps identify trends.
    """)
    
    # Real Estate Dataset Statistics
    st.header("Real Estate Dataset Statistics")
    st.write(real_estate_df.describe())
    
    st.markdown("""
    ### Key Insights from Real Estate Dataset
    - **House Price Distribution**: The range and distribution of house prices provide insights into the socioeconomic conditions of different ZIP codes.
    - **House and Lot Size**: Comparing statistics on house size and lot size across ZIP codes helps in understanding the distribution of property characteristics.
    """)

# Step 4: Data Merging and Missingness Page
elif page == 'Data Merging and Missingness':
    st.title('Data Merging and Missingness')

    st.write("### Explanation of Merging Process")
    st.write("Describe the merging process of the datasets, highlighting key considerations and any data integrity checks performed.")

    # Step 1: Initial missing value visualization for the merged dataset
    st.subheader("Missing Values Heatmap Before Handling")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(step1_df.isnull(), cbar=False, cmap='viridis', ax=ax)
    plt.title('Missing Values Heatmap After Merging')
    st.pyplot(fig)

    # Displaying missing values count before handling
    st.subheader("Missing Values Count Before Handling")
    st.write(step1_df.isnull().sum().sort_values(ascending=False))

    # Step 2: Explanation of handling missing values for insurance columns
    st.header("Handling Missing Values for Insurance Columns")
    st.markdown("""
    In the car insurance dataset, we addressed missing values as follows:
    - **AGE**: Missing values were filled with the median age to avoid skewing the data.
    - **Years of Job (YOJ)**: Filled missing values with the median, providing a central tendency measure.
    - **INCOME**: Cleaned the income column by removing dollar signs and commas, then filled missing values with the median.
    - **HOME_VAL**: Similar to INCOME, cleaned and filled missing values with the median to ensure consistency.
    - **OCCUPATION**: Missing values were filled with the mode, representing the most common occupation.
    - **CAR_AGE**: Filled missing values with the median to maintain data integrity.
    """)

    # Visualization after handling missing values for insurance columns
    st.subheader("Missing Values Heatmap After Handling Insurance Columns")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(step2_df.isnull(), cbar=False, cmap='viridis', ax=ax)
    plt.title('Missing Values Heatmap After Cleaning Insurance Data')
    st.pyplot(fig)

    # Display remaining missing values in the cleaned dataset
    st.markdown("### Missing Values in Each Column After Handling Insurance Data:")
    st.write(step2_df.isnull().sum().sort_values(ascending=False))

    # Step 3: Explanation of KNN imputation for real estate data
    st.header("Handling Missing Values for Real Estate Columns")
    st.markdown("""
    For the real estate dataset, we utilized KNN imputation to handle missing values:
    - **Numeric Columns**: `price`, `house_size`, `acre_lot`, and `CAR_AGE` were imputed using KNN, which considers the average of the nearest neighbors to fill in missing values.
    - **Categorical Column**: `CITY` was filled with the mode, ensuring that the most frequent city was used for imputation.
    """)

    # Final Heatmap to Show No Missingness in Merged Data
    st.subheader("Final Missing Values Heatmap After All Cleaning Steps")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(merged_df.isnull(), cbar=False, cmap='viridis', ax=ax)
    plt.title('Final Missing Values Heatmap (No Missing Values)')
    st.pyplot(fig)

    # Display remaining missing values in the final cleaned dataset
    st.markdown("### Missing Values in Each Column After All Handling:")
    st.write(merged_df.isnull().sum().sort_values(ascending=False))

    # Success message
    st.success("Missing values visualizations have been displayed successfully.")



# Step 5: EDA Page
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

# Step 6: Correlation Analysis
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
        color_scheme = st.selectbox("Select Color Scheme", ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'BuGn', 'RdBu'])

    # Filter the DataFrame to only include selected columns
    numeric_df = merged_df[selected_columns]

    # Check if numeric_df is empty or contains only one column
    if numeric_df.empty or numeric_df.shape[1] < 2:
        st.warning("Please select at least two numeric columns for correlation analysis.")
    else:
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
        
        # Top 10 correlations horizontal bar plot using Matplotlib and Seaborn
        st.subheader("Top 10 Correlations - Horizontal Bar Plot")
        top_10_corr = top_corr.head(10)

        # Check if top_10_corr has sufficient entries
        if not top_10_corr.empty and len(top_10_corr) >= 10:
            top_10_corr_df = pd.DataFrame(top_10_corr).reset_index()
            # Ensure that we only have 2 columns for variable pairs and correlations
            top_10_corr_df.columns = ['Variable Pair', 'Correlation']
            top_10_corr_df['Variable Pair'] = top_10_corr_df['Variable Pair'].apply(lambda x: f"{x[0]} - {x[1]}")

            # Create the horizontal bar plot using Matplotlib and Seaborn
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Correlation', y='Variable Pair', data=top_10_corr_df, palette=color_scheme)
            plt.title('Top 10 Correlation Values')
            plt.xlabel('Correlation')
            plt.ylabel('Variable Pair')
            plt.xlim(-1, 1)  # Set x-axis limits
            plt.axvline(0, color='gray', linestyle='--')  # Add a vertical line at 0
            st.pyplot(plt)  # Display the plot in Streamlit

        else:
            st.warning("Not enough correlations to display top correlations.")





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

    # Select Category
    category = st.selectbox("Select Category", ['EDUCATION', 'OCCUPATION', 'CAR_TYPE'])
    
    # For CLM_FREQ, we use a stacked bar chart
    metric = 'CLM_FREQ'
    
    # Slider for filtering claim frequency (integral values)
    clm_freq_range = st.slider("Select Claim Frequency Range", 
                                min_value=int(insurance_df['CLM_FREQ'].min()), 
                                max_value=int(insurance_df['CLM_FREQ'].max()), 
                                value=(int(insurance_df['CLM_FREQ'].min()), int(insurance_df['CLM_FREQ'].max())))
    
    # Filter the data based on selected claim frequency range
    filtered_data = insurance_df[(insurance_df['CLM_FREQ'] >= clm_freq_range[0]) & 
                                 (insurance_df['CLM_FREQ'] <= clm_freq_range[1])]

    # Group the data by category and claim frequency
    grouped_data = filtered_data.groupby([category, 'CLM_FREQ']).size().reset_index(name='Count')

    # Stacked column chart (bar chart) for CLM_FREQ
    fig = px.bar(grouped_data, 
                 x=category, 
                 y='Count', 
                 color='CLM_FREQ',  # Stack by CLM_FREQ
                 title=f'Stacked Bar Chart of Claim Frequency by {category}',
                 labels={'Count': 'Number of Claims', 'CLM_FREQ': 'Claim Frequency'},
                 barmode='stack')

    st.plotly_chart(fig)

    # Generate a dynamic hypothesis based on filtered claim frequency data
    max_clm_freq = filtered_data['CLM_FREQ'].max()
    min_clm_freq = filtered_data['CLM_FREQ'].min()
    
    dynamic_hypothesis = (
        f"In the {category} category, claim frequencies range from {min_clm_freq} to {max_clm_freq}. "
        "This suggests that different subgroups within this category may have varying levels of risk or claim patterns, "
        "leading to different claim frequencies."
    )

    # Display the dynamically generated hypothesis
    st.write("Dynamic Hypothesis:")
    st.write(dynamic_hypothesis)




# Step 8: Slope Analysis
elif page == 'Slope Analysis':
    st.title('Slope Analysis')

    # Define the variable pairs for CLM_FREQ and CLM_AMT
    clm_freq_pairs = [
        ('house_size', 'CLM_FREQ'),
        ('price', 'CLM_FREQ'),
        ('OLDCLAIM', 'CLM_FREQ'),
        ('CAR_AGE', 'CLM_FREQ'),
        ('INCOME', 'CLM_FREQ'),
        ('OCCUPATION_z_Blue Collar', 'CLM_FREQ'),
        ('EDUCATION_Bachelors', 'CLM_FREQ'),
        ('GENDER_Male', 'CLM_FREQ'),
        ('TOWN_SIZE', 'CLM_FREQ')
    ]

    clm_amt_pairs = [
        ('house_size', 'CLM_AMT'),
        ('price', 'CLM_AMT'),
        ('OLDCLAIM', 'CLM_AMT'),
        ('CAR_AGE', 'CLM_AMT'),
        ('INCOME', 'CLM_AMT'),
        ('OCCUPATION_z_Blue Collar', 'CLM_AMT'),
        ('EDUCATION_Bachelors', 'CLM_AMT'),
        ('GENDER_Male', 'CLM_AMT'),
        ('TOWN_SIZE', 'CLM_AMT')
    ]

    # Select variable pairs for analysis
    analysis_choice = st.radio("Choose Analysis Type", ['CLM_FREQ Analysis', 'CLM_AMT Analysis'])

    if analysis_choice == 'CLM_FREQ Analysis':
        variable_pairs = clm_freq_pairs
        y_label = 'Claim Frequency'
    else:
        variable_pairs = clm_amt_pairs
        y_label = 'Claim Amount'
    
    # Plot slope analysis
    for x_var, y_var in variable_pairs:
        slope = np.polyfit(insurance_df[x_var], insurance_df[y_var], 1)
        line = np.poly1d(slope)
        
        fig, ax = plt.subplots()
        ax.scatter(insurance_df[x_var], insurance_df[y_var], alpha=0.5)
        ax.plot(insurance_df[x_var], line(insurance_df[x_var]), color='red')
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_label)
        ax.set_title(f'Slope Analysis: {x_var} vs {y_var}')
        st.pyplot(fig)

# Add an "About" section
if st.sidebar.checkbox("Show About"):
    st.sidebar.title("About")
    st.sidebar.info(
        "This app is designed to analyze car insurance claims data. "
        "It provides various functionalities including data overview, statistics, "
        "exploratory data analysis, correlation analysis, and more."
    )
