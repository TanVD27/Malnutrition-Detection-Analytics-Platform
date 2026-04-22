# Malnutrition Detection & Analytics Platform

A comprehensive Streamlit-based dashboard for analyzing child malnutrition data in Maharashtra with Star Schema architecture, OLAP operations, and machine learning insights.

## Features

- **Data Analysis**: Comprehensive cleaning and preprocessing of malnutrition dataset
- **Star Schema Design**: Fact and dimension tables for efficient OLAP queries
- **Visualizations**: Interactive charts and heatmaps using Plotly
- **OLAP Operations**: Roll-Up, Drill-Down, Slice, Dice, and Pivot operations
- **Machine Learning**: Decision tree classification and clustering
- **PDF Reports**: Generate and download detailed analysis reports
- **Data Export**: Download filtered data and schema tables as CSV

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## Dataset

- **File**: `malnutrition_dataset_600.csv`
- **Records**: 600 children
- **Key Metrics**: Height, Weight, MUAC, Age, Gender, City, Area Type

## Architecture

### Star Schema Tables

- **Fact Table**: `fact_nutrition` - Central fact table with measurements
- **Dimensions**:
  - `dim_child` - Child demographics
  - `dim_location` - Geographic information
  - `dim_status` - Nutrition status categories
  - `dim_date` - Temporal dimensions

### Nutrition Status Classification

- **SAM**: Severe Acute Malnutrition (MUAC < 11.5 cm)
- **MAM**: Moderate Acute Malnutrition (MUAC 11.5–12.5 cm)
- **Normal**: MUAC ≥ 12.5 cm

## License

All rights reserved.
