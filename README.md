# Malnutrition Detection & Analytics Platform

An interactive **data analytics + machine learning dashboard** built using **Streamlit** to analyze child malnutrition across Maharashtra.

This project combines:
- 📊 Data Visualization
- 🧠 Machine Learning
- 🗂️ Data Warehousing (Star Schema)
- 🔄 OLAP Operations
- 📄 Automated Reporting

---

## 🚀 Features

### 📊 Dashboard Insights
- Total children, SAM, MAM, Normal counts
- SAM rate and average MUAC
- City-wise, gender-wise, and age-wise analysis
- Anthropometric analysis (Height, Weight, MUAC)

---

### 🤖 Machine Learning
- **Decision Tree Classifier**
  - Predicts nutrition status (SAM / MAM / Normal)
  - Displays model accuracy
- **Clustering (Custom Logic)**
  - Based on MUAC:
    - SAM Cluster
    - MAM Cluster
    - Normal Cluster

---

### ⚠️ Outlier Detection
- Uses **IQR method**
- Detects abnormal MUAC values
- Optional outlier removal
- Displays:
  - Total outliers
  - Dataset size before & after cleaning

---

### 🌟 Data Warehouse (Star Schema)
Includes:
- **Fact Table** → Nutrition data
- **Dimension Tables**:
  - Child
  - Location
  - Status
  - Date

---

### 🔄 OLAP Operations
Interactive analysis using:
- ⬆️ Roll-Up
- ⬇️ Drill-Down
- 🔪 Slice
- 🎲 Dice
- 🔁 Pivot Tables

---

### 📍 Advanced Visualizations
- Heatmaps (City × Status)
- Scatter plots (Height vs Weight)
- Correlation matrix
- Maharashtra SAM hotspot map

---

### 🔍 Additional Features
- Child ID lookup system
- CSV export (filtered data)
- Star schema table downloads
- 📄 PDF report generation

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Plotly**
- **Scikit-learn**
- **ReportLab**

---

## 📂 Dataset

- Contains child data (0–5 years)
- Features include:
  - Height
  - Weight
  - MUAC
  - Age
  - Gender
  - City
- Nutrition classification based on:
  - MUAC < 11.5 → SAM
  - 11.5–12.5 → MAM
  - > 12.5 → Normal

---

## ⚙️ Installation & Setup

### Clone Repository
```bash
git clone https://github.com/your-username/malnutrition-dashboard.git
cd malnutrition-dashboard

## 🚀 Live App  
👉 https://malnutrition-detection-analytics-platform-93jgreashjndybnvwmh8.streamlit.app/