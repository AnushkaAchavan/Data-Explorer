# Interactive EDA Web Application

## 📖 Overview
Exploratory Data Analysis (EDA) is one of the most crucial steps in any data science project. 
This project was created to provide a **simple, interactive, and beginner-friendly platform** for quickly understanding datasets.  
The goal is to allow users—students, analysts, or professionals—to upload their CSV data, clean it, visualize patterns, detect outliers, and generate automated reports, all without writing code.

---

## 🎯 Why This Project?
While EDA is essential, many newcomers find coding every step in Jupyter notebooks tedious.  
This app:
- **Simplifies EDA** into a few clicks.
- **Saves time** by automating cleaning and visualization.
- **Teaches data cleaning principles** interactively.
- **Generates a ready-to-use report** for quick insights.

---

## 🛠 How to Run the Project

1. **Clone or download** this repository.
   ```bash
   git clone https://github.com/yourusername/eda-streamlit-app.git
   cd eda-streamlit-app

---

## 🧩 Steps in Project Creation

1. **Data Upload & Preview**  
   - Upload a `.csv` file and view a sample of the dataset.

2. **Basic Data Insights**  
   - View dataset shape, missing values, and summary statistics.

3. **Data Cleaning Options**  
   - Handle missing values (Mode, Median, Forward Fill, Backward Fill, or Drop Rows).  
   - Drop duplicate rows.  
   - Drop columns with more than **70% missing values**.

4. **Outlier Detection**  
   - Visualize numeric columns with boxplots to identify outliers.

5. **Visual Analysis**  
   - Generate correlation heatmaps.  
   - Create other visualizations for better understanding of the data.

6. **Automated EDA Report**  
   - Generate a **Sweetviz** report for comprehensive dataset exploration.

---

## 📊 Output
1. Interactive Streamlit dashboard to explore and clean data.

2. Cleaned dataset with handled missing values and duplicates removed.

3. Visual reports: boxplots, correlation heatmaps, and distribution graphs.

4. Automated Sweetviz report for further analysis.

---

## 📝 Conclusion
This project demonstrates how EDA can be made accessible, interactive, and fast using Streamlit.
It’s designed for data analysts, students, and professionals who want to explore their data without writing extensive code.

 **Future improvements could include:**

-  Support for Excel (.xlsx) files.

-  More advanced visualizations.

-  Automated feature engineering suggestions.

