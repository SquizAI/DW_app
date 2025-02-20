#!/usr/bin/env python
import streamlit as st
st.set_page_config(
    page_title="Advanced Data Wrangling & Analysis App by the dudes who made it",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Doing cool things with Data
# -------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Optional packages for advanced tasks (resampling, XAI, mapping)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
except ImportError:
    SMOTE, ADASYN, RandomUnderSampler, SMOTETomek = None, None, None, None
try:
    import shap
except ImportError:
    shap = None
try:
    import geopandas as gpd
except ImportError:
    gpd = None

# -------------------------------
# Custom CSS for Enhanced Styling
# -------------------------------
st.markdown("""
    <style>
    /* Page background */
    .reportview-container {
        background: #f5f5f5;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f0f0f0;
        font-size: 16px;
    }
    /* Suggestion card styling */
    .suggestion-card {
        background-color: #fff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        border: 1px solid #ddd;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .suggestion-card h4 {
        margin-top: 0;
        color: #333;
    }
    .suggestion-card ul {
        margin: 0;
        padding-left: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Initialize Session State Variables
# -------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None           # Primary dataset
if 'df2' not in st.session_state:
    st.session_state.df2 = None          # Second dataset for integration
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None    # Merged dataset
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None   # Cleaned dataset
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'lambda_result' not in st.session_state:
    st.session_state.lambda_result = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None

# -------------------------------
# Sidebar: Global Settings & Navigation
# -------------------------------
st.sidebar.header("🔧 Settings & Navigation")
api_key_input = st.sidebar.text_input("Enter your OpenAI API Key", type="password", key="api_key_input")
if api_key_input:
    st.session_state.openai_api_key = api_key_input

# Updated sidebar navigation with additional modules
module = st.sidebar.selectbox(
    "Select Module",
    [
        "🏠 Home",
        "📤 Data Upload",
        "🔗 Data Integration",
        "🧹 Data Wrangling",
        "🛠 Data Binning",
        "⚙ Lambda & Feature Engineering",
        "📊 Data Analysis & EDA",
        "📊 Feature Importance",
        "📈 Visualizations",
        "🤖 AI Insights & Data Chat",
        "📉 Binary Classification & Resampling",
        "📥 Data Export",
        "📝 Final Report"
    ]
)

# -------------------------------
# Utility Functions
# -------------------------------
def review_data(df: pd.DataFrame) -> str:
    """
    Enhanced data review function providing structural insights.
    """
    suggestions = []
    n_rows, n_cols = df.shape
    suggestions.append(f"The dataset has **{n_rows}** rows and **{n_cols}** columns.")
    
    # Data types summary and unique counts
    dtype_counts = df.dtypes.value_counts().to_dict()
    suggestions.append("Data types distribution:")
    for dtype, count in dtype_counts.items():
        suggestions.append(f"- {dtype}: {count} columns")
    
    suggestions.append("Unique values per column:")
    for col in df.columns:
        suggestions.append(f"- {col}: {df[col].nunique()} unique values")
    
    # Memory usage
    mem_usage = df.memory_usage(deep=True).sum() / (1024**2)
    suggestions.append(f"Estimated memory usage: {mem_usage:.2f} MB")
    
    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        perc_missing = (missing / n_rows * 100).round(2)
        high_missing = perc_missing[perc_missing > 30]
        if not high_missing.empty:
            suggestions.append("Some columns have more than 30% missing values. Consider dropping or imputing these columns:")
            for col, perc in high_missing.items():
                suggestions.append(f"- {col}: {perc}% missing")
        else:
            suggestions.append("Missing values exist but are not severe; consider imputation if necessary.")
    else:
        suggestions.append("No missing values were detected.")
    
    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        suggestions.append(f"There are {dup_count} duplicate rows. Consider removing duplicates.")
    else:
        suggestions.append("No duplicate rows detected.")
    
    # Basic statistics for numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        suggestions.append("Basic statistics for numeric columns:")
        desc = df[num_cols].describe().to_dict()
        for col in num_cols:
            suggestions.append(f"- {col}: mean={desc[col]['mean']:.2f}, std={desc[col]['std']:.2f}")
    else:
        suggestions.append("No numeric columns detected.")
    
    # Return suggestions as a styled HTML card.
    html = '<div class="suggestion-card"><h4>Data Review & Suggestions</h4><ul>'
    for item in suggestions:
        html += f"<li>{item}</li>"
    html += "</ul></div>"
    return html

def impute_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Impute missing values using a given strategy: 'mean', 'median', or 'mode'
    """
    df_imputed = df.copy()
    num_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df_imputed[col].isnull().sum() > 0:
            if strategy == 'mean':
                df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)
            elif strategy == 'median':
                df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
            elif strategy == 'mode':
                df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
    return df_imputed

# -------------------------------
# Module Functions
# -------------------------------

def home_page():
    st.title("Welcome to the Advanced Data Wrangling & Analysis App by the dudes that made it")
    st.markdown("""
    **Overview:**  
    This application allows you to:
    - Upload or load a sample dataset.
    - Perform deep structural and quality investigations.
    - Merge multiple datasets.
    - Clean, impute, and transform data.
    - Perform data binning, lambda transformations, and feature engineering.
    - Conduct exploratory data analysis and feature importance analysis.
    - Visualize data using interactive and static charts.
    - Get AI insights and perform binary classification & resampling.
    - Export cleaned data and generate a final report.
    
    Use the sidebar to navigate through the modules.
    """)
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=200)

def data_upload_page():
    st.title("Data Upload")
    st.markdown("Upload a CSV file or load a sample dataset.")
    # Option to load sample dataset
    if st.button("Load Sample Dataset (Titanic)") or st.session_state.df is None:
        try:
            sample_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
            df_sample = pd.read_csv(sample_url)
            st.session_state.df = df_sample.copy()
            st.success(f"Sample dataset loaded: {df_sample.shape[0]} rows, {df_sample.shape[1]} columns.")
            st.write("### Data Preview:")
            st.dataframe(df_sample.head())
            suggestions_html = review_data(df_sample)
            st.markdown(suggestions_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading sample dataset: {e}")
    
    uploaded_file = st.file_uploader("Or upload your CSV File", type=["csv"], key="upload_csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df.copy()
            st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
            st.write("### Data Preview:")
            st.dataframe(df.head())
            suggestions_html = review_data(df)
            st.markdown(suggestions_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")

def data_integration_page():
    st.title("Data Integration")
    st.markdown("Merge a second dataset with the primary dataset.")
    if st.session_state.df is None:
        st.warning("Please upload the primary dataset first (Data Upload module).")
        return
    st.write("### Primary Dataset:")
    st.dataframe(st.session_state.df.head())
    
    uploaded_file2 = st.file_uploader("Upload Second CSV File", type=["csv"], key="upload_csv2")
    if uploaded_file2 is not None:
        try:
            df2 = pd.read_csv(uploaded_file2)
            st.session_state.df2 = df2.copy()
            st.success(f"Second dataset loaded: {df2.shape[0]} rows, {df2.shape[1]} columns.")
            st.write("### Second Dataset Preview:")
            st.dataframe(df2.head())
        except Exception as e:
            st.error(f"Error reading second file: {e}")
    
    if st.session_state.df2 is not None:
        st.markdown("#### Merge Settings")
        merge_key1 = st.selectbox("Select merge key from Primary Dataset", options=st.session_state.df.columns.tolist(), key="merge_key1")
        merge_key2 = st.selectbox("Select merge key from Second Dataset", options=st.session_state.df2.columns.tolist(), key="merge_key2")
        join_type = st.selectbox("Select type of join", options=["inner", "left", "right", "outer"], key="join_type")
        if st.button("Merge Datasets"):
            try:
                merged_df = pd.merge(st.session_state.df, st.session_state.df2,
                                     left_on=merge_key1, right_on=merge_key2, how=join_type)
                st.session_state.merged_df = merged_df.copy()
                st.success(f"Datasets merged successfully: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns.")
                st.dataframe(merged_df.head())
            except Exception as e:
                st.error(f"Merge error: {e}")

def data_wrangling_page():
    st.title("Data Wrangling & Preprocessing")
    if st.session_state.df is None:
        st.warning("Please upload a dataset first (Data Upload module).")
        return
    df = st.session_state.df.copy()
    st.write(f"**Current Dataset:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head())
    
    st.markdown("#### Basic Operations")
    # Drop columns
    drop_cols = st.multiselect("Select columns to drop", options=df.columns.tolist(), key="drop_cols")
    if st.button("Drop Selected Columns", key="drop_cols_btn"):
        if drop_cols:
            df = df.drop(columns=drop_cols)
            st.success(f"Dropped columns: {drop_cols}")
        else:
            st.info("No columns selected to drop.")
    
    # Rename columns
    st.subheader("Rename Columns")
    rename_dict = {}
    for col in df.columns:
        new_name = st.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{col}")
        if new_name != col:
            rename_dict[col] = new_name
    if st.button("Apply Renaming", key="rename_btn"):
        if rename_dict:
            df = df.rename(columns=rename_dict)
            st.success("Columns renamed.")
        else:
            st.info("No renaming changes applied.")
    
    # Convert data types
    st.subheader("Convert Data Types")
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    convert_cols = st.multiselect("Select columns to convert to 'category'", options=obj_cols, key="convert_cols")
    if st.button("Convert to Category", key="convert_btn"):
        for col in convert_cols:
            df[col] = df[col].astype("category")
        st.success("Selected columns converted to category.")
    
    # Filter rows
    st.subheader("Filter Rows")
    condition = st.text_input("Enter filter condition (e.g., `Age > 30`)", key="filter_condition")
    if st.button("Apply Filter", key="filter_btn"):
        try:
            filtered_df = df.query(condition)
            st.success(f"Filter applied. Rows: {df.shape[0]} → {filtered_df.shape[0]}")
            df = filtered_df
        except Exception as e:
            st.error(f"Filter error: {e}")
    
    # Handle duplicates
    st.subheader("Handle Duplicates")
    dup_cols = st.multiselect("Select columns for duplicate check (or leave blank for all)", options=df.columns.tolist(), key="dup_cols")
    if st.button("Show Duplicates", key="show_dups_btn"):
        dup_rows = df[df.duplicated(subset=dup_cols, keep=False)]
        st.write(f"Found {dup_rows.shape[0]} duplicate rows:")
        st.dataframe(dup_rows)
    if st.button("Drop Duplicates (Keep First)", key="drop_dups_btn"):
        df = df.drop_duplicates(subset=dup_cols, keep="first")
        st.success("Duplicates dropped.")
    
    # Handle missing values
    st.subheader("Handle Missing Values")
    missing_strategy = st.selectbox("Choose imputation strategy", options=["drop", "mean", "median", "mode"], key="missing_strategy")
    if st.button("Apply Missing Value Handling", key="missing_btn"):
        if missing_strategy == "drop":
            df = df.dropna()
            st.success("Rows with missing values dropped.")
        else:
            df = impute_missing_values(df, missing_strategy)
            st.success(f"Missing values imputed using {missing_strategy} strategy.")
    
    # Standardize numeric columns
    st.subheader("Standardize Numeric Columns")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols and st.button("Standardize", key="standardize_btn"):
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        st.success("Numeric columns standardized.")
    
    # Encode categorical columns
    st.subheader("Encode Categorical Columns")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols and st.button("Encode", key="encode_btn"):
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        st.success("Categorical columns encoded.")
    
    # Outlier detection and removal
    st.subheader("Outlier Detection & Removal")
    if num_cols:
        if st.button("Detect Outliers", key="outlier_detect_btn"):
            z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
            outliers_count = (z_scores > 3).sum().sum()
            st.write(f"Detected {outliers_count} outlier data points (Z-score > 3).")
        if st.button("Remove Outliers (Z-score > 3)", key="remove_outliers_btn"):
            z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
            df = df[(z_scores < 3).all(axis=1)]
            st.success("Outliers removed.")
    
    st.markdown("---")
    st.header("Cleaned Dataset Preview")
    st.write(f"Cleaned dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head())
    st.session_state.cleaned_df = df
    st.session_state.df = df  # update primary dataset

def data_binning_page():
    st.title("Data Binning")
    if st.session_state.cleaned_df is None:
        st.warning("Please preprocess your dataset first (Data Wrangling module).")
        return
    df = st.session_state.cleaned_df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("No numeric columns available for binning.")
        return
    col = st.selectbox("Select column to bin", options=num_cols, key="binning_col")
    bins = st.slider("Select number of bins", min_value=2, max_value=20, value=5, key="num_bins")
    if st.button("Apply Binning", key="binning_btn"):
        try:
            binned_series, bin_edges = pd.cut(df[col], bins=bins, retbins=True)
            df[f"{col}_binned"] = binned_series
            st.success(f"Binning applied. New column '{col}_binned' created.")
            fig = px.histogram(df, x=f"{col}_binned", title=f"Binned Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.cleaned_df = df
        except Exception as e:
            st.error(f"Binning error: {e}")

def lambda_feature_engineering_page():
    st.title("Lambda Function & Feature Engineering")
    if st.session_state.cleaned_df is None:
        st.warning("Please preprocess your dataset first (Data Wrangling module).")
        return
    df = st.session_state.cleaned_df.copy()
    st.header("Lambda Function Application")
    col = st.selectbox("Select column to transform", options=df.columns.tolist(), key="lambda_col")
    lambda_expr = st.text_input("Enter a lambda expression (e.g., x: x*2)", key="lambda_expr")
    if st.button("Apply Lambda Transformation", key="apply_lambda"):
        try:
            # Construct the lambda function from input string (use with caution)
            func = eval(lambda_expr)
            df[col] = df[col].apply(func)
            st.success(f"Lambda transformation applied on column {col}.")
            st.dataframe(df[[col]].head())
            st.session_state.lambda_result = df[col]
            st.session_state.cleaned_df = df
        except Exception as e:
            st.error(f"Lambda transformation error: {e}")
    
    st.header("Feature Engineering")
    st.markdown("Create a new feature by combining existing columns.")
    available_cols = df.columns.tolist()
    col1 = st.selectbox("Select first column", options=available_cols, key="fe_col1")
    col2 = st.selectbox("Select second column", options=available_cols, key="fe_col2")
    operation = st.selectbox("Select operation", options=["+", "-", "*", "/"], key="fe_op")
    new_feature_name = st.text_input("Enter new feature name", key="fe_name")
    if st.button("Create New Feature", key="fe_create"):
        try:
            if operation == "+":
                df[new_feature_name] = df[col1] + df[col2]
            elif operation == "-":
                df[new_feature_name] = df[col1] - df[col2]
            elif operation == "*":
                df[new_feature_name] = df[col1] * df[col2]
            elif operation == "/":
                df[new_feature_name] = df[col1] / df[col2]
            st.success(f"New feature '{new_feature_name}' created.")
            st.dataframe(df[[new_feature_name]].head())
            st.session_state.cleaned_df = df
        except Exception as e:
            st.error(f"Feature engineering error: {e}")

def data_analysis_page():
    st.title("Data Analysis & Exploratory Data Analysis (EDA)")
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()
    elif st.session_state.df is not None:
        df = st.session_state.df.copy()
    else:
        st.warning("Please upload and preprocess a dataset first.")
        return
    st.write(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head())
    
    st.markdown("### Interactive Analysis")
    col_choice = st.selectbox("Select a column for analysis", options=df.columns.tolist(), key="analysis_col")
    
    if pd.api.types.is_numeric_dtype(df[col_choice]):
        fig_hist = px.histogram(df, x=col_choice, title=f"Histogram of {col_choice}")
        st.plotly_chart(fig_hist, use_container_width=True)
        fig_box = px.box(df, y=col_choice, title=f"Box Plot of {col_choice}")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        vc = df[col_choice].value_counts().reset_index()
        vc.columns = [col_choice, "Count"]
        fig_bar = px.bar(vc, x=col_choice, y="Count", title=f"Value Counts of {col_choice}")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("### Advanced EDA")
    if st.button("Generate Pair Plot", key="pairplot_btn"):
        try:
            if len(df.select_dtypes(include=[np.number]).columns.tolist()) >= 2:
                sns_pair = sns.pairplot(df.select_dtypes(include=[np.number]))
                st.pyplot(sns_pair)
            else:
                st.error("Not enough numeric columns for pair plot.")
        except Exception as e:
            st.error(f"Pair plot error: {e}")
    if st.button("Show Correlation Heatmap", key="heatmap_btn"):
        try:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                corr = df[num_cols].corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No numeric columns for heatmap.")
        except Exception as e:
            st.error(f"Heatmap error: {e}")

def feature_importance_page():
    st.title("Feature Importance Analysis")
    if st.session_state.cleaned_df is None:
        st.warning("Please preprocess your dataset first (Data Wrangling module).")
        return
    df = st.session_state.cleaned_df.copy()
    st.write("Select a target variable for feature importance analysis:")
    target = st.selectbox("Target Variable", options=df.columns.tolist(), key="fi_target")
    if st.button("Compute Feature Importance", key="fi_btn"):
        try:
            # Decide if classification or regression based on target type and unique values
            if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
                # Regression
                X = df.drop(columns=[target])
                y = df[target]
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X.select_dtypes(include=[np.number]).fillna(0), y)
                importances = model.feature_importances_
                feat_names = X.select_dtypes(include=[np.number]).columns.tolist()
            else:
                # Classification
                X = df.drop(columns=[target])
                y = df[target]
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X.select_dtypes(include=[np.number]).fillna(0), y)
                importances = model.feature_importances_
                feat_names = X.select_dtypes(include=[np.number]).columns.tolist()
            fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
            fi_df = fi_df.sort_values(by="Importance", ascending=False)
            st.dataframe(fi_df)
            fig = px.bar(fi_df, x="Feature", y="Importance", title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.feature_importance = fi_df
        except Exception as e:
            st.error(f"Feature importance analysis error: {e}")

def advanced_visualizations_page():
    st.title("Graph Gallery: 20 Visualization Suggestions")
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()
    elif st.session_state.df is not None:
        df = st.session_state.df.copy()
    else:
        st.warning("Please upload or preprocess a dataset first.")
        return
    st.write("Select a visualization from the dropdown:")
    viz_options = [
        "1. Interactive Histogram",
        "2. Interactive Box Plot",
        "3. Interactive Scatter Plot",
        "4. Interactive Line Chart",
        "5. Interactive Area Chart",
        "6. Interactive Pie Chart",
        "7. Interactive Donut Chart",
        "8. Interactive Violin Plot",
        "9. Interactive Heatmap (Correlation)",
        "10. Interactive Bubble Chart",
        "11. Interactive Parallel Coordinates",
        "12. Interactive Sunburst Chart",
        "13. Interactive Treemap",
        "14. Interactive Waterfall Chart",
        "15. Interactive Radar Chart",
        "16. Static Histogram (Seaborn)",
        "17. Static Box Plot (Seaborn)",
        "18. Static Bar Chart (Matplotlib)",
        "19. Static Scatter Plot (Matplotlib)",
        "20. Static Density Plot (Seaborn)"
    ]
    selected_viz = st.selectbox("Visualization Type", viz_options, key="selected_viz")
    
    # Each visualization option is implemented generically:
    if selected_viz.startswith("1"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column", numeric_cols, key="viz1_col")
            fig = px.histogram(df, x=col, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No numeric columns available.")
    elif selected_viz.startswith("2"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column", numeric_cols, key="viz2_col")
            fig = px.box(df, y=col, title=f"Box Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No numeric columns available.")
    elif selected_viz.startswith("3"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="viz3_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="viz3_y")
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough numeric columns.")
    elif selected_viz.startswith("4"):
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        x_col = st.selectbox("Select X-axis", all_cols, key="viz4_x")
        y_col = st.selectbox("Select Y-axis (numeric)", numeric_cols, key="viz4_y")
        fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart: {y_col} over {x_col}")
        st.plotly_chart(fig, use_container_width=True)
    elif selected_viz.startswith("5"):
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        x_col = st.selectbox("Select X-axis", all_cols, key="viz5_x")
        y_col = st.selectbox("Select Y-axis (numeric)", numeric_cols, key="viz5_y")
        fig = px.area(df, x=x_col, y=y_col, title=f"Area Chart: {y_col} over {x_col}")
        st.plotly_chart(fig, use_container_width=True)
    elif selected_viz.startswith("6"):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            col = st.selectbox("Select categorical column", cat_cols, key="viz6_col")
            df_count = df[col].value_counts().reset_index()
            df_count.columns = [col, "Count"]
            fig = px.pie(df_count, values="Count", names=col, title=f"Pie Chart of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No categorical columns available.")
    elif selected_viz.startswith("7"):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            col = st.selectbox("Select column", cat_cols, key="viz7_col")
            df_count = df[col].value_counts().reset_index()
            df_count.columns = [col, "Count"]
            fig = px.pie(df_count, values="Count", names=col, title=f"Donut Chart of {col}", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No categorical columns available.")
    elif selected_viz.startswith("8"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column", numeric_cols, key="viz8_col")
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            split_col = st.selectbox("Optional split column", ["None"] + cat_cols, key="viz8_split")
            if split_col != "None":
                fig = px.violin(df, y=col, color=split_col, box=True, points="all", title=f"Violin Plot of {col} by {split_col}")
            else:
                fig = px.violin(df, y=col, box=True, points="all", title=f"Violin Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No numeric columns available.")
    elif selected_viz.startswith("9"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough numeric columns.")
    elif selected_viz.startswith("10"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols, key="viz10_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="viz10_y")
            size_col = st.selectbox("Bubble size", numeric_cols, key="viz10_size")
            fig = px.scatter(df, x=x_col, y=y_col, size=size_col, title=f"Bubble Chart: {x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough numeric columns.")
    elif selected_viz.startswith("11"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            fig = px.parallel_coordinates(df, dimensions=numeric_cols, title="Parallel Coordinates Plot")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough numeric columns.")
    elif selected_viz.startswith("12"):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cat_cols) >= 1:
            level1 = st.selectbox("Level 1", cat_cols, key="viz12_l1")
            remaining = [c for c in cat_cols if c != level1]
            level2 = st.selectbox("Level 2 (optional)", ["None"] + remaining, key="viz12_l2")
            level3 = st.selectbox("Level 3 (optional)", ["None"] + remaining, key="viz12_l3")
            path = [level1]
            if level2 != "None":
                path.append(level2)
            if level3 != "None":
                path.append(level3)
            fig = px.sunburst(df, path=path, title="Sunburst Chart")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough categorical columns.")
    elif selected_viz.startswith("13"):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cat_cols) >= 1:
            level1 = st.selectbox("Level 1", cat_cols, key="viz13_l1")
            remaining = [c for c in cat_cols if c != level1]
            level2 = st.selectbox("Level 2 (optional)", ["None"] + remaining, key="viz13_l2")
            level3 = st.selectbox("Level 3 (optional)", ["None"] + remaining, key="viz13_l3")
            path = [level1]
            if level2 != "None":
                path.append(level2)
            if level3 != "None":
                path.append(level3)
            fig = px.treemap(df, path=path, title="Treemap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough categorical columns.")
    elif selected_viz.startswith("14"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column", numeric_cols, key="viz14_col")
            df_sum = df.groupby(col).size().reset_index(name="Count")
            df_sum = df_sum.sort_values(by="Count", ascending=False)
            fig = go.Figure(go.Waterfall(
                measure=["relative"] * len(df_sum),
                x=df_sum[col].astype(str).tolist(),
                y=df_sum["Count"].tolist()
            ))
            fig.update_layout(title_text=f"Waterfall Chart based on {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No numeric column available.")
    elif selected_viz.startswith("15"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 3:
            selected = st.multiselect("Select exactly 3 columns", options=numeric_cols, default=numeric_cols[:3], key="viz15_cols")
            if len(selected) == 3:
                df_radar = df[selected].mean().reset_index()
                df_radar.columns = ["Variable", "Value"]
                fig = px.line_polar(df_radar, r="Value", theta="Variable", line_close=True, title="Radar Chart")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Please select exactly 3 numeric columns.")
        else:
            st.error("Not enough numeric columns.")
    elif selected_viz.startswith("16"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column", numeric_cols, key="viz16_col")
            fig, ax = plt.subplots()
            sns.histplot(df[col], ax=ax)
            ax.set_title(f"Static Histogram of {col}")
            st.pyplot(fig)
        else:
            st.error("No numeric columns available.")
    elif selected_viz.startswith("17"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column", numeric_cols, key="viz17_col")
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f"Static Box Plot of {col}")
            st.pyplot(fig)
        else:
            st.error("No numeric columns available.")
    elif selected_viz.startswith("18"):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            col = st.selectbox("Select column", cat_cols, key="viz18_col")
            df_counts = df[col].value_counts().reset_index()
            df_counts.columns = [col, "Count"]
            fig, ax = plt.subplots()
            ax.bar(df_counts[col], df_counts["Count"])
            ax.set_title(f"Static Bar Chart of {col}")
            ax.set_xticklabels(df_counts[col], rotation=45)
            st.pyplot(fig)
        else:
            st.error("No categorical columns available.")
    elif selected_viz.startswith("19"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="viz19_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="viz19_y")
            fig, ax = plt.subplots()
            ax.scatter(df[x_col], df[y_col])
            ax.set_title(f"Static Scatter Plot: {x_col} vs {y_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
        else:
            st.error("Not enough numeric columns.")
    elif selected_viz.startswith("20"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column", numeric_cols, key="viz20_col")
            fig, ax = plt.subplots()
            sns.kdeplot(df[col], ax=ax)
            ax.set_title(f"Static Density Plot of {col}")
            st.pyplot(fig)
        else:
            st.error("No numeric columns available.")

def ai_insights_page():
    st.title("AI Insights & Data Chat")
    st.markdown("""
    **Talk to Your Data:**  
    Generate insights for a specific column or ask a general question about your dataset.  
    Enter your query below.
    """)
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.session_state.openai_api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()
    elif st.session_state.df is not None:
        df = st.session_state.df.copy()
    else:
        st.warning("Please upload and preprocess a dataset first.")
        return
    query_type = st.radio("Choose Query Type", ["Column Insights", "General Data Chat"], key="query_type")
    if query_type == "Column Insights":
        col_ai = st.selectbox("Select a column for insights", options=df.columns.tolist(), key="ai_col")
        if pd.api.types.is_numeric_dtype(df[col_ai]):
            stats = df[col_ai].describe().to_dict()
        else:
            stats = {}
        missing = int(df[col_ai].isnull().sum())
        prompt = (f"Analyze the following column data:\n\n"
                  f"Column: {col_ai}\n"
                  f"Missing Values: {missing}\n"
                  f"Statistics: {stats}\n\n"
                  "Provide detailed insights, trends, and recommendations on further processing or modeling this data.")
    else:
        summary = df.head(3).to_dict()
        prompt = (f"Here is a brief summary of the dataset:\n{summary}\n\n"
                  "Now, answer the following question about the dataset. "
                  "Be specific and detailed. Question: ")
        user_question = st.text_area("Enter your question about the dataset", key="user_question")
        prompt += user_question
    if st.button("Ask AI", key="ai_ask_btn"):
        with st.spinner("Generating AI response..."):
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": "You are a data analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = completion.choices[0].message.content
                st.subheader("AI Response")
                st.write(response_text)
            except Exception as e:
                st.error(f"Error generating AI response: {e}")
    st.markdown("### Supporting Charts")
    if pd.api.types.is_numeric_dtype(df.iloc[:,0]):
        fig_hist_ai = px.histogram(df, x=df.columns[0], title=f"Histogram of {df.columns[0]}")
        st.plotly_chart(fig_hist_ai, use_container_width=True)
    else:
        vc_ai = df.iloc[:,0].value_counts().reset_index()
        vc_ai.columns = [df.columns[0], "Count"]
        fig_bar_ai = px.bar(vc_ai, x=df.columns[0], y="Count", title=f"Bar Chart of {df.columns[0]}")
        st.plotly_chart(fig_bar_ai, use_container_width=True)

def binary_classification_resampling_page():
    st.title("Binary Classification & Resampling")
    if st.session_state.cleaned_df is None:
        st.warning("Please preprocess your dataset first (Data Wrangling module).")
        return
    df = st.session_state.cleaned_df.copy()
    st.markdown("Select a binary target variable and evaluate classification performance before and after resampling.")
    target = st.selectbox("Select Binary Target Variable", options=[col for col in df.columns if df[col].nunique()==2], key="binary_target")
    if target:
        X = df.drop(columns=[target])
        # Use only numeric columns for modeling
        X = X.select_dtypes(include=[np.number]).fillna(0)
        y = df[target]
        test_size = st.slider("Select test size fraction", min_value=0.1, max_value=0.5, value=0.3, step=0.1, key="test_size")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        st.subheader("Without Resampling")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        st.write(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1 Score: {f1:.2f}")
        
        resample_option = st.selectbox("Select Resampling Technique", options=["None", "SMOTE", "ADASYN"], key="resample_option")
        if resample_option != "None":
            if resample_option == "SMOTE" and SMOTE is not None:
                resampler = SMOTE(random_state=42)
            elif resample_option == "ADASYN" and ADASYN is not None:
                resampler = ADASYN(random_state=42)
            else:
                st.error("Selected resampling technique not available.")
                return
            X_res, y_res = resampler.fit_resample(X_train, y_train)
            st.subheader(f"With {resample_option}")
            model_res = LogisticRegression(max_iter=1000)
            model_res.fit(X_res, y_res)
            preds_res = model_res.predict(X_test)
            acc_res = accuracy_score(y_test, preds_res)
            prec_res = precision_score(y_test, preds_res)
            rec_res = recall_score(y_test, preds_res)
            f1_res = f1_score(y_test, preds_res)
            st.write(f"Accuracy: {acc_res:.2f}, Precision: {prec_res:.2f}, Recall: {rec_res:.2f}, F1 Score: {f1_res:.2f}")
            st.session_state.classification_results = {
                "Without Resampling": {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1},
                f"With {resample_option}": {"Accuracy": acc_res, "Precision": prec_res, "Recall": rec_res, "F1": f1_res}
            }
    else:
        st.error("No binary target variable available.")

def data_export_page():
    st.title("Data Export")
    if st.session_state.cleaned_df is None:
        st.warning("No cleaned dataset available. Please run the Data Wrangling module first.")
        return
    df = st.session_state.cleaned_df.copy()
    st.markdown("### Download your cleaned data as CSV")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    st.write("### Data Preview:")
    st.dataframe(df.head())

def final_report_page():
    st.title("Final Report")
    st.markdown("### Summary of Analysis and Processing Steps")
    report = ""
    if st.session_state.df is not None:
        report += f"**Primary Dataset:** {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns\n\n"
    if st.session_state.merged_df is not None:
        report += f"**Merged Dataset:** {st.session_state.merged_df.shape[0]} rows, {st.session_state.merged_df.shape[1]} columns\n\n"
    if st.session_state.cleaned_df is not None:
        report += f"**Cleaned Dataset:** {st.session_state.cleaned_df.shape[0]} rows, {st.session_state.cleaned_df.shape[1]} columns\n\n"
    if st.session_state.lambda_result is not None:
        report += "**Lambda Transformation:** Applied successfully.\n\n"
    if st.session_state.feature_importance is not None:
        report += "**Feature Importance Analysis:** Completed.\n\n"
    if st.session_state.classification_results is not None:
        report += "**Binary Classification & Resampling:** Results available.\n\n"
    st.markdown(report)
    st.markdown("### Detailed Report")
    st.text_area("Final Report", value=report, height=300)
    if st.button("Download Report"):
        report_filename = f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, "w") as f:
            f.write(report)
        st.success(f"Report downloaded as {report_filename} (Check your local directory).")

# -------------------------------
# Main: Render the Selected Module
# -------------------------------
if module == "🏠 Home":
    home_page()
elif module == "📤 Data Upload":
    data_upload_page()
elif module == "🔗 Data Integration":
    data_integration_page()
elif module == "🧹 Data Wrangling":
    data_wrangling_page()
elif module == "🛠 Data Binning":
    data_binning_page()
elif module == "⚙ Lambda & Feature Engineering":
    lambda_feature_engineering_page()
elif module == "📊 Data Analysis & EDA":
    data_analysis_page()
elif module == "📊 Feature Importance":
    feature_importance_page()
elif module == "📈 Visualizations":
    advanced_visualizations_page()
elif module == "🤖 AI Insights & Data Chat":
    ai_insights_page()
elif module == "📉 Binary Classification & Resampling":
    binary_classification_resampling_page()
elif module == "📥 Data Export":
    data_export_page()
elif module == "📝 Final Report":
    final_report_page()
