import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
from io import StringIO
import warnings

if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = Warning

# Streamlit settings
st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("📊 Interactive Data Explorer")
sns.set(font_scale=0.6)

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preview data
    st.subheader("Dataset Preview")
    st.write(df.head().reset_index(drop=True))

    st.subheader("Basic Information")
    st.write(f"**Shape:** {df.shape}")
    st.write("**Missing Values per Column:**")
    st.write(df.isnull().sum())

    st.subheader("Statistics")
    st.write(df.describe(include='all'))

    st.subheader("Data Types and Non-Null Counts")
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    st.header("Data Cleaning Options")

    # Handle missing values
    missing_option = st.selectbox(
        "How to handle missing values?",
        ("Do nothing", "Fill with Mode", "Fill with Median", "Drop Rows", "Fill with ffill", "Fill with bfill")
    )
    if missing_option != "Do nothing":
        before = df.shape[0]
        if missing_option == 'Fill with Mode':
            for col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        elif missing_option == 'Fill with Median':
            for col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        elif missing_option == 'Drop Rows':
            df.dropna(inplace=True)
        elif missing_option == 'Fill with ffill':
            df.fillna(method='ffill', inplace=True)
        elif missing_option == 'Fill with bfill':
            df.fillna(method='bfill', inplace=True)
        after = df.shape[0]
        st.success(f"Missing values handled. Rows before: {before}, after: {after}.")

    # Drop duplicates
    if st.checkbox("Drop Duplicates"):
        before = df.shape[0]
        df.drop_duplicates(inplace=True)
        after = df.shape[0]
        st.success(f"Duplicates removed: {before - after}")

    # Boxplot for outlier detection
    st.subheader("Boxplot for Outlier Detection")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_columns.empty:
        col_for_boxplot = st.selectbox("Select a numeric column", numeric_columns)
        fig, ax = plt.subplots(figsize=(4, 2.5), dpi=150)
        sns.boxplot(x=df[col_for_boxplot], ax=ax)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No numeric columns available.")

    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(4, 2.5), dpi=150)
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No numeric columns for heatmap.")

    # Distribution plot
    st.subheader("Distribution of a Column")
    column = st.selectbox("Select a column", df.columns)
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=150)
    sns.histplot(df[column], kde=True, ax=ax)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Detect outliers but do not modify dataset
if st.checkbox("Show Detected Outliers"):
    outlier_info = {}
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        outlier_info[col] = len(outliers)
    st.write("**Outlier Counts per Column:**", outlier_info)


# Drop columns with too many missing values
if st.checkbox("Drop columns with too many missing values"):
    df.dropna(axis=1, thresh=len(df) * 0.3, inplace=True)
    st.success("Columns with more than 70% missing values dropped.")


    # Display cleaned data
    st.subheader("Cleaned Data Preview")
    st.write(df)    


    # Summary after cleaning
    st.subheader("Summary After Cleaning")
    st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
    st.write(f"**Missing Values:** {df.isnull().sum().sum()}")

    # Sweetviz report
    st.subheader("Automated EDA Report (Sweetviz)")
    if st.button("Create EDA Report"):
        report = sv.analyze(df)
        report_file = "sweetviz_report.html"
        report.show_html(report_file)
        with open(report_file, "rb") as f:
            st.download_button(
                label="Download EDA Report",
                data=f,
                file_name="sweetviz_report.html",
                mime="text/html"
            )

else:
    st.info("Please upload a CSV file to start EDA.")
