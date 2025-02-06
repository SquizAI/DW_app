#!/usr/bin/env python
# Enhancement #1: Place st.set_page_config as the very first Streamlit command.
import streamlit as st
st.set_page_config(
    page_title="Dataset-Agnostic Data Wrangling & Analysis App by Matty",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standard library and third-party imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from io import StringIO
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Enhancement #2: Optional packages for advanced tasks (resampling, XAI, mapping)
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

# Enhancement #3: Custom CSS for enhanced styling of sidebar and suggestion cards
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

# Enhancement #4: Initialize session state variables early
if 'df' not in st.session_state:
    st.session_state.df = None           # The original dataset
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None   # After wrangling/cleaning
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

# -------------------------------
# Sidebar: Global Settings & Enhanced Navigation
# -------------------------------
st.sidebar.header("🔧 Settings & Navigation")
# Enhancement #5: API key input at the very top of the sidebar.
api_key_input = st.sidebar.text_input("Enter your OpenAI API Key", type="password", key="api_key_input")
if api_key_input:
    st.session_state.openai_api_key = api_key_input

# Enhancement #6: Enhanced sidebar navigation with emojis and clear labels.
module = st.sidebar.selectbox(
    "Select Module", 
    [
        "🏠 Home", 
        "📤 Data Upload", 
        "🧹 Data Wrangling", 
        "📊 Data Analysis", 
        "📈 Visualizations", 
        "🤖 AI Insights & Data Chat", 
        "📥 Data Export"
    ]
)

# -------------------------------
# Utility Function: Data Review & Suggestions
# -------------------------------
def review_data(df: pd.DataFrame) -> str:
    """
    Analyzes the dataset and returns HTML-formatted suggestions.
    Enhancement #7: Provide dynamic suggestions based on the dataset.
    """
    suggestions = []
    n_rows, n_cols = df.shape
    suggestions.append(f"The dataset has **{n_rows}** rows and **{n_cols}** columns.")
    
    # Data types summary
    dtype_counts = df.dtypes.value_counts().to_dict()
    suggestions.append("Data types distribution:")
    for dtype, count in dtype_counts.items():
        suggestions.append(f"- {dtype}: {count} columns")
    
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
    
    # Numeric columns and outlier suggestion
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        suggestions.append("For numeric columns, consider standardization and outlier detection.")
        z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
        outliers = (z_scores > 3).sum().sum()
        suggestions.append(f"Outlier check: Approximately {outliers} data points have a Z-score > 3.")
    else:
        suggestions.append("No numeric columns detected; consider encoding categorical data.")
    
    # Enhancement #8: Return the suggestions as a styled HTML card.
    html = '<div class="suggestion-card"><h4>Data Review & Suggestions</h4><ul>'
    for item in suggestions:
        html += f"<li>{item}</li>"
    html += "</ul></div>"
    return html

# -------------------------------
# Module 1: Home Page
# -------------------------------
def home_page():
    st.title("Welcome to the Data Wrangling & Analysis App")
    st.markdown("""
    **Overview:**  
    This dataset-agnostic application allows you to:
    
    - **Upload** any CSV file.
    - **Clean and preprocess** your data (e.g., drop/rename columns, standardize, encode, filter, handle duplicates/missing values, detect outliers).
    - **Explore and visualize** your data interactively.
    - **Generate AI insights & chat** with your data in natural language.
    - **Export** the cleaned dataset.
    
    Use the sidebar for navigation.
    """)
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=200)

# -------------------------------
# Module 2: Data Upload
# -------------------------------
def data_upload_page():
    st.title("Data Upload")
    st.markdown("Upload any CSV file to get started.")
    
    # Enhancement #9: When a file is uploaded, automatically review the data.
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="upload_csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df.copy()
            st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
            st.write("### Data Preview:")
            st.dataframe(df.head())
            # Display data review suggestions
            suggestions_html = review_data(df)
            st.markdown(suggestions_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("No file uploaded yet.")

# -------------------------------
# Module 3: Data Wrangling & Preprocessing
# -------------------------------
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

    # Convert data types (e.g., to category)
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

    # Drop missing values
    st.subheader("Handle Missing Values")
    if st.button("Drop Rows with Missing Values", key="drop_na_btn"):
        df = df.dropna()
        st.success("Missing values removed.")

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

    # Outlier detection
    st.subheader("Outlier Detection")
    if num_cols and st.button("Detect Outliers", key="outlier_btn"):
        z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
        outliers_z = (z_scores > 3).sum()
        Q1 = df[num_cols].quantile(0.25)
        Q3 = df[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = (((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).sum())
        st.write("Outliers (Z-score method):")
        st.write(outliers_z)
        st.write("Outliers (IQR method):")
        st.write(outliers_iqr)

    st.markdown("---")
    st.header("Cleaned Dataset Preview")
    st.write(f"Cleaned dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head())
    st.session_state.cleaned_df = df

# -------------------------------
# Module 4: Data Analysis & Exploratory Visualization
# -------------------------------
def data_analysis_page():
    st.title("Data Analysis & Exploratory Visualization")
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

    st.markdown("### Static Chart")
    if pd.api.types.is_numeric_dtype(df[col_choice]):
        fig_static, ax_static = plt.subplots()
        sns.histplot(df[col_choice], ax=ax_static)
        ax_static.set_title(f"Static Histogram of {col_choice}")
        st.pyplot(fig_static)
    else:
        st.write("No static chart available for non-numeric columns.")

# -------------------------------
# Module 5: Advanced Visualizations (Graph Gallery)
# -------------------------------
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
            from plotly.graph_objects import Figure, Waterfall
            fig = Figure(Waterfall(
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

# -------------------------------
# Module 6: AI Insights & Data Chat
# -------------------------------
def ai_insights_page():
    st.title("AI Insights & Data Chat")
    st.markdown("""
    **Talk to Your Data:**  
    Generate insights for a specific column or ask a general question about your dataset.  
    Enter your query below.
    """)
    
    # Enhancement #10: Ensure the OpenAI API key is present.
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
    
    # Enhancement #11: Build dynamic prompt based on query type.
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

    # Enhancement #12: Button to send the prompt to OpenAI.
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
    # Enhancement #13: Display supporting charts for context.
    if pd.api.types.is_numeric_dtype(df.iloc[:,0]):
        fig_hist_ai = px.histogram(df, x=df.columns[0], title=f"Histogram of {df.columns[0]}")
        st.plotly_chart(fig_hist_ai, use_container_width=True)
    else:
        vc_ai = df.iloc[:,0].value_counts().reset_index()
        vc_ai.columns = [df.columns[0], "Count"]
        fig_bar_ai = px.bar(vc_ai, x=df.columns[0], y="Count", title=f"Bar Chart of {df.columns[0]}")
        st.plotly_chart(fig_bar_ai, use_container_width=True)

# -------------------------------
# Module 7: Data Export
# -------------------------------
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

# -------------------------------
# Main: Render the Selected Module
# -------------------------------
if module == "🏠 Home":
    home_page()
elif module == "📤 Data Upload":
    data_upload_page()
elif module == "🧹 Data Wrangling":
    data_wrangling_page()
elif module == "📊 Data Analysis":
    data_analysis_page()
elif module == "📈 Visualizations":
    advanced_visualizations_page()
elif module == "🤖 AI Insights & Data Chat":
    ai_insights_page()
elif module == "📥 Data Export":
    data_export_page()
