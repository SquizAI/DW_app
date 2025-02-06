#!/usr/bin/env python
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from io import StringIO

# Optional: Import OpenAI for AI insights
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -------------------------------
# Page Configuration & Session State
# -------------------------------
st.set_page_config(
    page_title="Comprehensive Data Wrangling & Analysis App by Matty",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None           # Original data
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None   # Data after wrangling
if 'additional_df' not in st.session_state:
    st.session_state.additional_df = None
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

# -------------------------------
# Sidebar: Global Settings and Navigation
# -------------------------------
st.sidebar.header("🔧 Settings & Navigation")
# API Key input (for AI Insights) at the very top
api_key_input = st.sidebar.text_input(
    "Enter your OpenAI API Key", 
    type="password",
    key="api_key_input"
)
if api_key_input:
    st.session_state.openai_api_key = api_key_input

# Navigation options
navigation_options = ["Home", "Data Upload", "Data Wrangling", "Data Analysis", "Visualizations", "AI Insights", "Data Export"]
page = st.sidebar.radio("Navigation", navigation_options, key="nav_page")

# -------------------------------
# PAGE: Home
# -------------------------------
def home_page():
    st.title("Welcome to the Data Wrangling & Analysis App")
    st.markdown("""
    This application allows you to:

    - **Upload** your datasets.
    - **Clean and transform** your data with a rich suite of wrangling tools.
    - **Visualize** your data interactively with over 20 graph options.
    - Generate **AI insights** about your data.
    - **Export** your cleaned dataset for further analysis.
    
    Use the navigation on the left to get started.
    """)
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=200)

# -------------------------------
# PAGE: Data Upload
# -------------------------------
def data_upload_page():
    st.title("Data Upload")
    st.write("Upload your primary dataset (CSV file). You can also upload an additional dataset to merge.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="upload_csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df.copy()
            st.success(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
            st.dataframe(df.head())
            st.session_state.df_filename = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Please upload a CSV file to begin.")
    
    st.markdown("---")
    additional_file = st.file_uploader("Upload Additional Dataset (Optional)", type=["csv"], key="upload_csv_additional")
    if additional_file is not None:
        try:
            additional_df = pd.read_csv(additional_file)
            st.session_state.additional_df = additional_df.copy()
            st.write("Additional dataset preview:")
            st.dataframe(additional_df.head())
            st.success("Additional dataset loaded.")
        except Exception as e:
            st.error(f"Error reading additional file: {e}")

# -------------------------------
# PAGE: Data Wrangling
# -------------------------------
def data_wrangling_page():
    st.title("Data Wrangling")
    if st.session_state.df is None:
        st.warning("No dataset loaded. Please use the Data Upload page.")
        return
    
    # Work on a copy of the original data
    df = st.session_state.df.copy()
    st.subheader("Current Data Overview")
    st.write(f"Dataset has **{df.shape[0]}** rows and **{df.shape[1]}** columns.")
    st.dataframe(df.head())

    st.markdown("## Basic Operations")
    # --- Drop Columns ---
    st.subheader("Drop Columns")
    drop_cols = st.multiselect("Select columns to drop", options=df.columns.tolist(), key="drop_cols")
    if st.button("Drop Selected Columns", key="drop_columns_button"):
        if drop_cols:
            df = df.drop(columns=drop_cols)
            st.success(f"Dropped columns: {drop_cols}")
        else:
            st.info("No columns selected.")

    # --- Rename Columns ---
    st.subheader("Rename Columns")
    rename_dict = {}
    for col in df.columns:
        new_name = st.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{col}_{datetime.now().timestamp()}")
        if new_name != col:
            rename_dict[col] = new_name
    if st.button("Apply Renaming", key="apply_rename"):
        if rename_dict:
            df = df.rename(columns=rename_dict)
            st.success("Columns renamed.")
        else:
            st.info("No renaming changes applied.")
    
    # --- Data Type Conversion ---
    st.subheader("Convert Data Types")
    convert_cols = st.multiselect("Select columns to convert to 'category'", 
                                  options=df.select_dtypes(include="object").columns.tolist(), 
                                  key="convert_cols")
    if st.button("Convert to Category", key="convert_category"):
        for col in convert_cols:
            df[col] = df[col].astype("category")
        st.success("Selected columns converted to category.")
    
    # --- Filter Rows ---
    st.subheader("Filter Rows")
    st.write("Enter a condition to filter rows (e.g., `Rating > 3`):")
    filter_condition = st.text_input("Filter condition", key="filter_condition")
    if st.button("Apply Filter", key="apply_filter"):
        try:
            filtered_df = df.query(filter_condition)
            st.success(f"Filter applied. Rows reduced from {df.shape[0]} to {filtered_df.shape[0]}.")
            df = filtered_df
        except Exception as e:
            st.error(f"Error in filter condition: {e}")
    
    # --- Handle Duplicates ---
    st.subheader("Handle Duplicates")
    dup_cols = st.multiselect("Select columns for duplicate check (or leave blank for all)", options=df.columns.tolist(), key="dup_cols")
    if st.button("Show Duplicate Rows", key="show_dup"):
        if dup_cols:
            dup_rows = df[df.duplicated(subset=dup_cols, keep=False)]
        else:
            dup_rows = df[df.duplicated(keep=False)]
        st.write(f"Found {dup_rows.shape[0]} duplicate rows:")
        st.dataframe(dup_rows)
    if st.button("Drop Duplicates (Keep First)", key="drop_dup"):
        if dup_cols:
            df = df.drop_duplicates(subset=dup_cols, keep="first")
        else:
            df = df.drop_duplicates(keep="first")
        st.success("Duplicates dropped.")

    # --- Drop Missing Values ---
    st.subheader("Handle Missing Values")
    if st.button("Drop Rows with Missing Values", key="drop_na"):
        df = df.dropna()
        st.success("Rows with missing values dropped.")
    
    st.markdown("## Specialized Cleaning: Ramen Data")
    if ("ramen" in st.session_state.get("df_filename", "").lower() or 
        st.checkbox("Apply Ramen Data Cleaning Tasks", key="ramen_cleaning_checkbox")):
        st.write("Performing Ramen-specific cleaning tasks:")
        # Rename "Stars" to "Rating"
        if "Stars" in df.columns:
            df = df.rename(columns={"Stars": "Rating"})
            st.write("- Renamed **Stars** to **Rating**.")
        # Convert "Style" to category
        if "Style" in df.columns:
            df["Style"] = df["Style"].astype("category")
            st.write("- Converted **Style** to category type.")
        # Drop "Country" column
        if "Country" in df.columns:
            df = df.drop(columns=["Country"])
            st.write("- Dropped **Country** column.")
        # Rename "Brand" to "Company" and "Variety" to "Product"
        rename_map_ramen = {}
        if "Brand" in df.columns:
            rename_map_ramen["Brand"] = "Company"
        if "Variety" in df.columns:
            rename_map_ramen["Variety"] = "Product"
        if rename_map_ramen:
            df = df.rename(columns=rename_map_ramen)
            st.write("- Renamed columns for ramen data.")
        # Remove duplicates based on Company/Product
        if "Company" in df.columns and "Product" in df.columns:
            dup = df[df.duplicated(subset=["Company", "Product"], keep=False)]
            st.write(f"- Found {dup.shape[0]} duplicate Company/Product rows:")
            st.dataframe(dup)
            df = df.drop_duplicates(subset=["Company", "Product"], keep="first")
            st.write("- Dropped duplicate Company/Product rows (keeping first).")
        # Drop any remaining missing values
        df = df.dropna()
        st.write("- Dropped any remaining rows with missing values.")
    
    st.markdown("---")
    st.header("Cleaned Data Overview")
    st.write(f"Cleaned dataset now has **{df.shape[0]}** rows and **{df.shape[1]}** columns.")
    st.dataframe(df.head())
    st.session_state.cleaned_df = df

# -------------------------------
# PAGE: Data Analysis
# -------------------------------
def data_analysis_page():
    st.title("Data Analysis")
    # Prefer cleaned data if available; otherwise, use the uploaded data
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()
    elif st.session_state.df is not None:
        df = st.session_state.df.copy()
    else:
        st.warning("No dataset loaded. Please upload and/or clean a dataset first.")
        return

    st.subheader("Dataset Overview")
    st.write(f"Dataset has **{df.shape[0]}** rows and **{df.shape[1]}** columns.")
    st.dataframe(df.head())

    st.markdown("## Interactive Visualizations")
    col_choice = st.selectbox("Select a column for analysis", options=df.columns.tolist(), key="analysis_col")
    
    if pd.api.types.is_numeric_dtype(df[col_choice]):
        # Interactive 2D Charts
        fig_hist = px.histogram(df, x=col_choice, title=f"Interactive Histogram of {col_choice}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        fig_box = px.box(df, y=col_choice, title=f"Interactive Box Plot of {col_choice}")
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Interactive 3D Scatter Plot (if at least 3 numeric columns)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 3:
            selected_cols = st.multiselect("Select 3 numeric columns for 3D Scatter Plot", 
                                           options=numeric_cols, default=numeric_cols[:3],
                                           key="scatter3d_cols")
            if len(selected_cols) == 3:
                fig_scatter3d = px.scatter_3d(df, x=selected_cols[0], y=selected_cols[1], z=selected_cols[2],
                                              title="Interactive 3D Scatter Plot")
                st.plotly_chart(fig_scatter3d, use_container_width=True)
    else:
        st.info("Selected column is non-numeric; showing value counts.")
        vc = df[col_choice].value_counts().reset_index()
        vc.columns = [col_choice, "Count"]
        fig_bar = px.bar(vc, x=col_choice, y="Count", title=f"Value Counts of {col_choice}")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("## Static Charts")
    if pd.api.types.is_numeric_dtype(df[col_choice]):
        fig_static, ax_static = plt.subplots()
        sns.histplot(df[col_choice], ax=ax_static)
        ax_static.set_title(f"Static Histogram of {col_choice}")
        st.pyplot(fig_static)
    else:
        st.write("No static chart available for non-numeric columns.")

# -------------------------------
# PAGE: Advanced Visualizations
# -------------------------------
def advanced_visualizations_page():
    st.title("Graph Gallery: 20 Visualization Suggestions")
    # Use cleaned data if available; otherwise, use the uploaded data.
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()
    elif st.session_state.df is not None:
        df = st.session_state.df.copy()
    else:
        st.warning("No dataset available. Please upload or clean data first.")
        return

    st.write("Select a graph type below to explore different visualizations:")

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
    selected_viz = st.selectbox("Select a Visualization", viz_options, key="selected_viz")

    # Visualization implementations:
    if selected_viz.startswith("1"):
        # 1. Interactive Histogram
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column for histogram", numeric_cols, key="hist_col")
            fig = px.histogram(df, x=col, title=f"Interactive Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No numeric columns available for histogram.")
    elif selected_viz.startswith("2"):
        # 2. Interactive Box Plot
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column for box plot", numeric_cols, key="box_col")
            fig = px.box(df, y=col, title=f"Interactive Box Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No numeric columns available for box plot.")
    elif selected_viz.startswith("3"):
        # 3. Interactive Scatter Plot
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y")
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Need at least two numeric columns for scatter plot.")
    elif selected_viz.startswith("4"):
        # 4. Interactive Line Chart
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        x_col = st.selectbox("Select X-axis (time or index)", all_cols, key="line_x")
        y_col = st.selectbox("Select Y-axis (numeric)", numeric_cols, key="line_y")
        fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart: {y_col} over {x_col}")
        st.plotly_chart(fig, use_container_width=True)
    elif selected_viz.startswith("5"):
        # 5. Interactive Area Chart
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        x_col = st.selectbox("Select X-axis", all_cols, key="area_x")
        y_col = st.selectbox("Select Y-axis (numeric)", numeric_cols, key="area_y")
        fig = px.area(df, x=x_col, y=y_col, title=f"Area Chart: {y_col} over {x_col}")
        st.plotly_chart(fig, use_container_width=True)
    elif selected_viz.startswith("6"):
        # 6. Interactive Pie Chart
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            col = st.selectbox("Select categorical column", cat_cols, key="pie_col")
            df_count = df[col].value_counts().reset_index()
            df_count.columns = [col, "Count"]
            fig = px.pie(df_count, values="Count", names=col, title=f"Pie Chart of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No categorical columns available for pie chart.")
    elif selected_viz.startswith("7"):
        # 7. Interactive Donut Chart
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            col = st.selectbox("Select categorical column for donut chart", cat_cols, key="donut_col")
            df_count = df[col].value_counts().reset_index()
            df_count.columns = [col, "Count"]
            fig = px.pie(df_count, values="Count", names=col, title=f"Donut Chart of {col}", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No categorical columns available for donut chart.")
    elif selected_viz.startswith("8"):
        # 8. Interactive Violin Plot
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols, key="violin_col")
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            split_col = st.selectbox("Select categorical column for split (optional)", ["None"] + cat_cols, key="violin_split")
            if split_col != "None":
                fig = px.violin(df, y=col, color=split_col, box=True, points="all", title=f"Violin Plot of {col} by {split_col}")
            else:
                fig = px.violin(df, y=col, box=True, points="all", title=f"Violin Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No numeric columns available for violin plot.")
    elif selected_viz.startswith("9"):
        # 9. Interactive Heatmap (Correlation)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough numeric columns for correlation heatmap.")
    elif selected_viz.startswith("10"):
        # 10. Interactive Bubble Chart
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="bubble_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="bubble_y")
            size_col = st.selectbox("Select column for bubble size", numeric_cols, key="bubble_size")
            fig = px.scatter(df, x=x_col, y=y_col, size=size_col, title=f"Bubble Chart: {x_col} vs {y_col} (size: {size_col})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough numeric columns for bubble chart.")
    elif selected_viz.startswith("11"):
        # 11. Interactive Parallel Coordinates Plot
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            fig = px.parallel_coordinates(df, dimensions=numeric_cols, title="Parallel Coordinates Plot")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough numeric columns for parallel coordinates.")
    elif selected_viz.startswith("12"):
        # 12. Interactive Sunburst Chart
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cat_cols) >= 1:
            level1 = st.selectbox("Level 1", cat_cols, key="sunburst_level1")
            remaining = [c for c in cat_cols if c != level1]
            level2 = st.selectbox("Level 2 (optional)", ["None"] + remaining, key="sunburst_level2")
            level3 = st.selectbox("Level 3 (optional)", ["None"] + remaining, key="sunburst_level3")
            path = [level1]
            if level2 != "None":
                path.append(level2)
            if level3 != "None":
                path.append(level3)
            fig = px.sunburst(df, path=path, title="Sunburst Chart")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough categorical columns for sunburst chart.")
    elif selected_viz.startswith("13"):
        # 13. Interactive Treemap
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cat_cols) >= 1:
            level1 = st.selectbox("Level 1", cat_cols, key="treemap_level1")
            remaining = [c for c in cat_cols if c != level1]
            level2 = st.selectbox("Level 2 (optional)", ["None"] + remaining, key="treemap_level2")
            level3 = st.selectbox("Level 3 (optional)", ["None"] + remaining, key="treemap_level3")
            path = [level1]
            if level2 != "None":
                path.append(level2)
            if level3 != "None":
                path.append(level3)
            fig = px.treemap(df, path=path, title="Treemap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough categorical columns for treemap.")
    elif selected_viz.startswith("14"):
        # 14. Interactive Waterfall Chart (using Plotly Graph Objects)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select numeric column for waterfall", numeric_cols, key="waterfall_col")
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
            st.error("No numeric column available for waterfall chart.")
    elif selected_viz.startswith("15"):
        # 15. Interactive Radar Chart (Polar Chart)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 3:
            selected = st.multiselect("Select 3 numeric columns", options=numeric_cols, default=numeric_cols[:3], key="radar_cols")
            if len(selected) == 3:
                df_radar = df[selected].mean().reset_index()
                df_radar.columns = ["Variable", "Value"]
                fig = px.line_polar(df_radar, r="Value", theta="Variable", line_close=True, title="Radar Chart")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Please select exactly 3 numeric columns for radar chart.")
        else:
            st.error("Not enough numeric columns for radar chart.")
    elif selected_viz.startswith("16"):
        # 16. Static Histogram using Seaborn
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols, key="static_hist_col")
            fig, ax = plt.subplots()
            sns.histplot(df[col], ax=ax)
            ax.set_title(f"Static Histogram of {col}")
            st.pyplot(fig)
        else:
            st.error("No numeric columns available for static histogram.")
    elif selected_viz.startswith("17"):
        # 17. Static Box Plot using Seaborn
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols, key="static_box_col")
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f"Static Box Plot of {col}")
            st.pyplot(fig)
        else:
            st.error("No numeric columns available for static box plot.")
    elif selected_viz.startswith("18"):
        # 18. Static Bar Chart using Matplotlib
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            col = st.selectbox("Select categorical column", cat_cols, key="static_bar_col")
            df_counts = df[col].value_counts().reset_index()
            df_counts.columns = [col, "Count"]
            fig, ax = plt.subplots()
            ax.bar(df_counts[col], df_counts["Count"])
            ax.set_title(f"Static Bar Chart of {col}")
            ax.set_xticklabels(df_counts[col], rotation=45)
            st.pyplot(fig)
        else:
            st.error("No categorical columns available for static bar chart.")
    elif selected_viz.startswith("19"):
        # 19. Static Scatter Plot using Matplotlib
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="static_scatter_x")
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="static_scatter_y")
            fig, ax = plt.subplots()
            ax.scatter(df[x_col], df[y_col])
            ax.set_title(f"Static Scatter Plot: {x_col} vs {y_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
        else:
            st.error("Not enough numeric columns for static scatter plot.")
    elif selected_viz.startswith("20"):
        # 20. Static Density Plot using Seaborn
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols, key="static_density_col")
            fig, ax = plt.subplots()
            sns.kdeplot(df[col], ax=ax)
            ax.set_title(f"Static Density Plot of {col}")
            st.pyplot(fig)
        else:
            st.error("No numeric columns available for static density plot.")

# -------------------------------
# PAGE: AI Insights
# -------------------------------
def ai_insights_page():
    st.title("AI Insights")
    st.markdown("""
    In this section, the app uses OpenAI to generate insights about your data.
    An API key is required (provided in the sidebar above). The insights are accompanied by charts for further context.
    """)
    
    if OpenAI is None:
        st.error("The OpenAI package is not installed. Install it to enable AI Insights.")
        return
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=st.session_state.openai_api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return
    
    # Use cleaned data if available
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy()
    elif st.session_state.df is not None:
        df = st.session_state.df.copy()
    else:
        st.warning("No dataset loaded.")
        return

    col_ai = st.selectbox("Select a column for AI analysis", options=df.columns.tolist(), key="ai_col")
    if st.button("Generate AI Insights", key=f"ai_insights_button_{col_ai}"):
        with st.spinner("Generating insights..."):
            try:
                stats = df[col_ai].describe().to_dict() if pd.api.types.is_numeric_dtype(df[col_ai]) else {}
                missing = int(df[col_ai].isnull().sum())
                prompt = (f"Analyze the following column data:\n\n"
                          f"Column: {col_ai}\n"
                          f"Missing Values: {missing}\n"
                          f"Statistics: {stats}\n\n"
                          "Provide detailed insights and recommendations.")
                completion = client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": "You are a data analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = completion.choices[0].message.content
                st.subheader("AI Generated Insights")
                st.write(response_text)
            except Exception as e:
                st.error(f"Error generating AI insights: {e}")
    
    st.markdown("## Supporting Charts")
    if pd.api.types.is_numeric_dtype(df[col_ai]):
        fig_hist_ai = px.histogram(df, x=col_ai, title=f"Histogram of {col_ai}")
        st.plotly_chart(fig_hist_ai, use_container_width=True)
        fig_box_ai = px.box(df, y=col_ai, title=f"Box Plot of {col_ai}")
        st.plotly_chart(fig_box_ai, use_container_width=True)
    else:
        vc_ai = df[col_ai].value_counts().reset_index()
        vc_ai.columns = [col_ai, "Count"]
        fig_bar_ai = px.bar(vc_ai, x=col_ai, y="Count", title=f"Bar Chart of {col_ai}")
        st.plotly_chart(fig_bar_ai, use_container_width=True)

# -------------------------------
# PAGE: Data Export
# -------------------------------
def data_export_page():
    st.title("Data Export")
    if st.session_state.cleaned_df is None:
        st.warning("No cleaned dataset available. Please perform data wrangling first.")
        return
    df = st.session_state.cleaned_df.copy()
    st.subheader("Download your cleaned data as CSV")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    st.write("Preview of cleaned data:")
    st.dataframe(df.head())

# -------------------------------
# Main: Render Selected Page
# -------------------------------
if page == "Home":
    home_page()
elif page == "Data Upload":
    data_upload_page()
elif page == "Data Wrangling":
    data_wrangling_page()
elif page == "Data Analysis":
    data_analysis_page()
elif page == "Visualizations":
    advanced_visualizations_page()
elif page == "AI Insights":
    ai_insights_page()
elif page == "Data Export":
    data_export_page()
