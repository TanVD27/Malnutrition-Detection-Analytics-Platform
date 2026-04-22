import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors as rl_colors

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Malnutrition AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# STYLE
# -------------------------------
st.markdown("""
<style>

/* 🌌 MAIN BACKGROUND */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Poppins', sans-serif;
}

/* ✨ GLASS CARD EFFECT */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(15px);
    border-radius: 15px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
}

/* 📊 KPI CARDS (PREMIUM) */
.metric-card {
    background: linear-gradient(135deg, #141E30, #243B55);
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    transition: all 0.3s ease;
    border: 1px solid rgba(255,255,255,0.1);
}

.metric-card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 10px 25px rgba(0, 242, 255, 0.3);
}

.metric-card h3 {
    font-size: 13px;
    color: #bbb;
    margin: 0;
}

.metric-card h2 {
    font-size: 30px;
    margin: 5px 0 0;
    color: #00f2ff;
}

/* 🧠 SECTION TITLES */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 10px;
    padding-left: 12px;
    border-left: 5px solid #00f2ff;
    background: linear-gradient(to right, #00f2ff22, transparent);
}

/* 🔥 ADVANCED SIDEBAR */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.08);
    color: #00f2ff;
    border: 1px solid rgba(0,242,255,0.3);
}

/* Title */
[data-testid="stSidebar"]::before {
    content: "⚡ Smart Filters";
    display: block;
    font-size: 18px;
    font-weight: bold;
    color: #00f2ff;
    text-align: center;
    padding: 15px 0;
    border-bottom: 1px solid rgba(0, 242, 255, 0.2);
    margin-bottom: 10px;
}

/* Labels */
[data-testid="stSidebar"] label {
    font-size: 12px !important;
    color: #9aa4b2 !important;
}

/* Inputs */
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,242,255,0.2) !important;
    backdrop-filter: blur(10px);
}

/* Hover glow */
[data-testid="stSidebar"] [data-baseweb="select"]:hover {
    border: 1px solid #00f2ff !important;
    box-shadow: 0 0 10px #00f2ff55;
}

/* Selected tags */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: linear-gradient(135deg, #00f2ff, #ff00c8) !important;
    color: black !important;
    border-radius: 6px !important;
}

/* 🔘 BUTTONS */
.stButton>button {
    background: linear-gradient(135deg, #00f2ff, #ff00c8);
    border: none;
    color: black;
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: bold;
}

.stButton>button:hover {
    transform: scale(1.05);
}

/* 📦 EXPANDER */
.streamlit-expanderHeader {
    background-color: rgba(255,255,255,0.05);
    border-radius: 10px;
}

/* 📊 TABLE */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* 🧊 SCROLLBAR */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #00f2ff55;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("Malnutrition Detection & Analytics Platform")
st.markdown("""
<div class='glass-card'>
<h4 style='color:#00f2ff;'>📊 Smart Nutrition Intelligence Dashboard</h4>
<p style='color:#bbb;'>AI-powered insights · OLAP Analysis · Real-time Visualization</p>
</div>
""", unsafe_allow_html=True)

st.markdown("**Maharashtra Analysis** · Star Schema · OLAP · Visualization · Reporting")

# -------------------------------
# LOAD & CLEAN DATA
# -------------------------------
df_raw = pd.read_csv("malnutrition_dataset_600.csv")
df = df_raw.copy()
df = df.drop_duplicates()
df['Gender'] = df['Gender'].str.strip().str.lower()
df['Height_cm'] = pd.to_numeric(df['Height_cm'], errors='coerce')
df['Weight_kg'] = pd.to_numeric(df['Weight_kg'], errors='coerce')
df['MUAC_cm']   = pd.to_numeric(df['MUAC_cm'],   errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
today = pd.to_datetime("today")
df['Age_months'] = ((today - df['DOB']).dt.days // 30).clip(lower=0)
df['Age_Group']  = pd.cut(df['Age_months'],
                           bins=[0,6,12,24,36,60],
                           labels=["0-6m","6-12m","1-2yr","2-3yr","3-5yr"])
df['Birth_Year']  = df['DOB'].dt.year
df['Birth_Month'] = df['DOB'].dt.month

def classify(muac):
    if muac < 11.5:   return "SAM"
    elif muac < 12.5: return "MAM"
    else:             return "Normal"

df['Nutrition_Status'] = df['MUAC_cm'].apply(classify)
df['Status_Code'] = df['Nutrition_Status'].map({"SAM":1,"MAM":2,"Normal":3})
df['BMI'] = (df['Weight_kg'] / ((df['Height_cm']/100)**2)).round(2)


# ================================
# 🔮 MACHINE LEARNING SECTION
# ================================
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# ---------- Prediction Model ----------
features = df[['Height_cm','Weight_kg','MUAC_cm','Age_months']]
target = df['Status_Code']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# ---------- Clustering ----------
# ---------- Meaningful Clustering (Based on MUAC) ----------
def assign_cluster(muac):
    if muac < 11.5:
        return "SAM Cluster"
    elif muac < 12.5:
        return "MAM Cluster"
    else:
        return "Normal Cluster"

df['Cluster'] = df['MUAC_cm'].apply(assign_cluster)

# ---------- Outlier Detection ----------
Q1 = df['MUAC_cm'].quantile(0.25)
Q3 = df['MUAC_cm'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['MUAC_cm'] < Q1 - 1.5*IQR) | (df['MUAC_cm'] > Q3 + 1.5*IQR)]

# ---------- Outlier Removal (Optional) ----------
df_no_outliers = df[~((df['MUAC_cm'] < Q1 - 1.5*IQR) | (df['MUAC_cm'] > Q3 + 1.5*IQR))] 


COLORS = ["#00f2ff","#ff00c8","#39ff14"]
STATUS_COLOR = {"SAM":"#ff00c8","MAM":"#00f2ff","Normal":"#39ff14"}

# ================================
# BUILD STAR SCHEMA TABLES
# ================================
fact_nutrition = df[['Child_ID','City','Nutrition_Status','Status_Code',
                      'Height_cm','Weight_kg','MUAC_cm','BMI','Age_months']].copy()
fact_nutrition['Fact_ID'] = range(1, len(fact_nutrition)+1)

dim_child = df[['Child_ID','DOB','Gender','Age_months','Age_Group',
                'Birth_Year','Birth_Month']].drop_duplicates().reset_index(drop=True)

dim_location = df[['City','Area_Type']].drop_duplicates().reset_index(drop=True)
dim_location['Location_ID'] = range(1, len(dim_location)+1)

dim_status = pd.DataFrame({
    'Status_Code':      [1, 2, 3],
    'Nutrition_Status': ['SAM','MAM','Normal'],
    'Description':      ['Severe Acute Malnutrition (MUAC < 11.5)',
                         'Moderate Acute Malnutrition (MUAC 11.5–12.5)',
                         'Normal (MUAC >= 12.5)'],
    'Risk_Level':       ['Critical','Moderate','Low']
})

dim_date = df[['Birth_Year','Birth_Month']].drop_duplicates().reset_index(drop=True)
dim_date['Quarter'] = ((dim_date['Birth_Month']-1)//3+1).apply(lambda x: f"Q{x}")
dim_date['Date_ID'] = range(1, len(dim_date)+1)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.markdown("### 🔍 Filters")
st.sidebar.markdown("---")

city_filter   = st.sidebar.multiselect("🏙️ City",      df['City'].unique(),      default=df['City'].unique())
gender_filter = st.sidebar.multiselect("👤 Gender",     df['Gender'].unique(),    default=df['Gender'].unique())
area_filter   = st.sidebar.multiselect("🏘️ Area",      df['Area_Type'].unique(), default=df['Area_Type'].unique())
status_filter = st.sidebar.multiselect("📋 Status",     ["SAM","MAM","Normal"],   default=["SAM","MAM","Normal"])

df_filtered = df[
    (df['City'].isin(city_filter)) &
    (df['Gender'].isin(gender_filter)) &
    (df['Area_Type'].isin(area_filter)) &
    (df['Nutrition_Status'].isin(status_filter))
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(df_filtered)}** children selected")

# -------------------------------
# DATA VIEWS
# -------------------------------
with st.expander("📄 View Raw Data"):
    st.dataframe(df_raw.head(20), use_container_width=True)
with st.expander("✅ View Cleaned Data"):
    st.dataframe(df.head(20), use_container_width=True)

# -------------------------------
# KPI CARDS
# -------------------------------
st.markdown("<div class='section-title'>📊 Key Metrics</div>", unsafe_allow_html=True)
sam_pct  = round(len(df_filtered[df_filtered['Nutrition_Status']=='SAM'])/max(len(df_filtered),1)*100, 1)
avg_muac = round(df_filtered['MUAC_cm'].mean(), 2) if len(df_filtered) else 0

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.markdown(f"<div class='metric-card'><h3>Total Children</h3><h2>{len(df_filtered)}</h2></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-card'><h3>SAM 🔴</h3><h2>{len(df_filtered[df_filtered['Nutrition_Status']=='SAM'])}</h2></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-card'><h3>MAM 🟡</h3><h2>{len(df_filtered[df_filtered['Nutrition_Status']=='MAM'])}</h2></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-card'><h3>Normal 🟢</h3><h2>{len(df_filtered[df_filtered['Nutrition_Status']=='Normal'])}</h2></div>", unsafe_allow_html=True)
c5.markdown(f"<div class='metric-card'><h3>SAM Rate</h3><h2>{sam_pct}%</h2></div>", unsafe_allow_html=True)
c6.markdown(f"<div class='metric-card'><h3>Avg MUAC</h3><h2>{avg_muac}</h2></div>", unsafe_allow_html=True)

# ================================================
# STAR SCHEMA (Improved Styling)
# ================================================
st.markdown("<div class='section-title'>🌟 Star Schema Design</div>", unsafe_allow_html=True)
st.markdown("> A **Star Schema** organises data into one central **Fact Table** surrounded by **Dimension Tables**. This powers fast OLAP queries and clean reporting.")

fig_schema = go.Figure()

# Fact Table (center)
fact_text = "<b>FACT: fact_nutrition</b><br>Fact_ID (PK)<br>Child_ID (FK)<br>City<br>Nutrition_Status<br>Status_Code<br>Height_cm<br>Weight_kg<br>MUAC_cm<br>BMI<br>Age_months"
fig_schema.add_trace(go.Scatter(
    x=[0], y=[0], mode='markers+text',
    marker=dict(size=160, color='lightseagreen', line=dict(color='white', width=3), symbol='square'),
    text=[fact_text], textposition='middle center',
    textfont=dict(color='black', size=11, family="Arial"),
    name='Fact Table'
))

# Dimension Tables
dims = [
    (-4, 2, "<b>DIM: Child</b><br>Child_ID (PK)<br>DOB<br>Gender<br>Age_months<br>Age_Group<br>Birth_Year<br>Birth_Month", "lightskyblue"),
    (4, 2, "<b>DIM: Location</b><br>Location_ID (PK)<br>City<br>Area_Type", "lightgreen"),
    (-4, -2, "<b>DIM: Date</b><br>Date_ID (PK)<br>Birth_Year<br>Birth_Month<br>Quarter", "khaki"),
    (4, -2, "<b>DIM: Status</b><br>Status_Code (PK)<br>Nutrition_Status<br>Description<br>Risk_Level", "plum")
]

for x, y, label, color in dims:
    fig_schema.add_trace(go.Scatter(
        x=[x], y=[y], mode='markers+text',
        marker=dict(size=140, color=color, line=dict(color='white', width=3), symbol='square'),
        text=[label], textposition='middle center',
        textfont=dict(color='black', size=10, family="Arial"),
        name=label.split("<br>")[0]
    ))
    # Arrows to Fact Table
    # Offset arrows so they connect edges, not overlap boxes
    offset_x = x * 0.7   # pull arrow start closer to fact table
    offset_y = y * 0.7
    fig_schema.add_annotation(
        x=0, y=0, ax=offset_x, ay=offset_y,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowwidth=2, arrowcolor='white'
    )



fig_schema.update_layout(
    showlegend=False,
    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    font_color="white",
    xaxis=dict(visible=False, range=[-6, 6]),
    yaxis=dict(visible=False, range=[-4, 4]),
    height=650,
    title="⭐ Star Schema – Malnutrition Data Warehouse (with Attributes)"
)

st.plotly_chart(fig_schema, use_container_width=True)



# ================================================
# OLAP OPERATIONS
# ================================================
st.markdown("<div class='section-title'>🔄 OLAP Operations</div>", unsafe_allow_html=True)
st.markdown("> **OLAP (Online Analytical Processing)** allows multi-dimensional analysis. Below are the 4 core operations applied interactively on this dataset.")

olap1,olap2,olap3,olap4,olap5 = st.tabs(["⬆️ Roll-Up","⬇️ Drill-Down","🔪 Slice","🎲 Dice","🔁 Pivot"])

# ROLL-UP
with olap1:
    st.markdown("### ⬆️ Roll-Up")
    st.markdown("**Roll-Up** = Aggregate from lower → higher level. Example: City → Area Type → All Maharashtra")
    level = st.radio("Level:", ["City Level","Area Type Level","All Maharashtra"], horizontal=True)
    if level == "City Level":
        r = df.groupby(['City','Nutrition_Status']).size().reset_index(name='Count')
        fig_r = px.bar(r, x='City', y='Count', color='Nutrition_Status',
                       barmode='stack', color_discrete_sequence=COLORS, title="Roll-Up: City Level")
    elif level == "Area Type Level":
        r = df.groupby(['Area_Type','Nutrition_Status']).size().reset_index(name='Count')
        fig_r = px.bar(r, x='Area_Type', y='Count', color='Nutrition_Status',
                       barmode='stack', color_discrete_sequence=COLORS, title="Roll-Up: Area Type Level")
    else:
        r = df.groupby('Nutrition_Status').size().reset_index(name='Count')
        fig_r = px.pie(r, names='Nutrition_Status', values='Count',
                       color_discrete_sequence=COLORS, title="Roll-Up: All Maharashtra")
    fig_r.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig_r, use_container_width=True)
    st.dataframe(r, use_container_width=True)

# DRILL-DOWN
with olap2:
    st.markdown("### ⬇️ Drill-Down")
    st.markdown("**Drill-Down** = Go from summary → detail. Select a city and drill into its breakdown.")
    sel_city = st.selectbox("Select City:", df['City'].unique())
    drill    = st.radio("Drill Level:", ["Area Type","Gender","Age Group"], horizontal=True)
    city_df  = df[df['City'] == sel_city]
    if drill == "Area Type":
        d = city_df.groupby(['Area_Type','Nutrition_Status']).size().reset_index(name='Count')
        fig_d = px.bar(d, x='Area_Type', y='Count', color='Nutrition_Status',
                       barmode='group', color_discrete_sequence=COLORS,
                       title=f"Drill-Down: {sel_city} → Area Type")
    elif drill == "Gender":
        d = city_df.groupby(['Gender','Nutrition_Status']).size().reset_index(name='Count')
        fig_d = px.bar(d, x='Gender', y='Count', color='Nutrition_Status',
                       barmode='group', color_discrete_sequence=COLORS,
                       title=f"Drill-Down: {sel_city} → Gender")
    else:
        d = city_df.groupby(['Age_Group','Nutrition_Status']).size().reset_index(name='Count')
        fig_d = px.bar(d, x='Age_Group', y='Count', color='Nutrition_Status',
                       barmode='group', color_discrete_sequence=COLORS,
                       title=f"Drill-Down: {sel_city} → Age Group")
    fig_d.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig_d, use_container_width=True)
    st.dataframe(d, use_container_width=True)

# SLICE
with olap3:
    st.markdown("### 🔪 Slice")
    st.markdown("**Slice** = Fix ONE dimension, view all others. Example: Fix Status=SAM → see city & gender breakdown.")
    slice_dim = st.selectbox("Slice Dimension:", ["Nutrition_Status","Gender","Area_Type","City"])
    slice_val = st.selectbox("Value:", df[slice_dim].unique())
    sliced    = df[df[slice_dim] == slice_val]
    c1,c2 = st.columns(2)
    with c1:
        s1 = sliced.groupby('City').size().reset_index(name='Count')
        fig_s1 = px.bar(s1, x='City', y='Count', color='City',
                        title=f"Slice [{slice_dim}={slice_val}] → City")
        fig_s1.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                              font_color="white", showlegend=False)
        st.plotly_chart(fig_s1, use_container_width=True)
    with c2:
        s2 = sliced.groupby('Gender').size().reset_index(name='Count')
        fig_s2 = px.pie(s2, names='Gender', values='Count',
                        color_discrete_sequence=COLORS,
                        title=f"Slice [{slice_dim}={slice_val}] → Gender")
        fig_s2.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
        st.plotly_chart(fig_s2, use_container_width=True)
    st.info(f"**{len(sliced)} children** with `{slice_dim} = {slice_val}`")
    st.dataframe(sliced.head(10), use_container_width=True)

# DICE
with olap4:
    st.markdown("### 🎲 Dice")
    st.markdown("**Dice** = Fix MULTIPLE dimensions (sub-cube). Example: Mumbai + Female + SAM.")
    d_city   = st.multiselect("Cities:",    df['City'].unique(),    default=[df['City'].unique()[0]])
    d_gender = st.multiselect("Genders:",   df['Gender'].unique(),  default=df['Gender'].unique())
    d_status = st.multiselect("Status:",    ["SAM","MAM","Normal"], default=["SAM","MAM","Normal"])
    d_area   = st.multiselect("Area Type:", df['Area_Type'].unique(),default=df['Area_Type'].unique())
    diced = df[
        (df['City'].isin(d_city)) &
        (df['Gender'].isin(d_gender)) &
        (df['Nutrition_Status'].isin(d_status)) &
        (df['Area_Type'].isin(d_area))
    ]
    st.success(f"🎲 Dice Result: **{len(diced)} children** match all filters")
    if len(diced) > 0:
        fig_dice = px.scatter(diced, x='Height_cm', y='Weight_kg',
                              color='Nutrition_Status', size='MUAC_cm',
                              hover_data=['City','Gender','Age_months'],
                              color_discrete_map=STATUS_COLOR,
                              title="Dice Result — Height vs Weight")
        fig_dice.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
        st.plotly_chart(fig_dice, use_container_width=True)
        st.dataframe(diced, use_container_width=True)
    else:
        st.warning("No data matches the selected combination.")

# PIVOT
with olap5:
    st.markdown("### 🔁 Pivot Table")
    st.markdown("**Pivot** = Rotate data to view from a different angle (rows ↔ columns).")
    p_row = st.selectbox("Rows:",    ["City","Gender","Area_Type","Age_Group"], index=0)
    p_col = st.selectbox("Columns:", ["Nutrition_Status","Gender","Area_Type"],  index=0)
    p_val = st.selectbox("Values:",  ["Count","Avg MUAC","Avg Height","Avg Weight"])
    if p_val == "Count":
        pt = pd.crosstab(df[p_row], df[p_col])
    elif p_val == "Avg MUAC":
        pt = df.pivot_table(values='MUAC_cm',   index=p_row, columns=p_col, aggfunc='mean').round(2)
    elif p_val == "Avg Height":
        pt = df.pivot_table(values='Height_cm', index=p_row, columns=p_col, aggfunc='mean').round(2)
    else:
        pt = df.pivot_table(values='Weight_kg', index=p_row, columns=p_col, aggfunc='mean').round(2)
    st.dataframe(pt, use_container_width=True)
    if not pt.empty:
        fig_p = px.imshow(pt, text_auto=True, color_continuous_scale='Turbo',
                          title=f"Pivot: {p_row} × {p_col} ({p_val})")
        fig_p.update_layout(paper_bgcolor="#0E1117", font_color="white")
        st.plotly_chart(fig_p, use_container_width=True)

# ================================
# DISTRIBUTION CHARTS
# ================================
st.markdown("<div class='section-title'>📊 Nutrition Distribution</div>", unsafe_allow_html=True)
c1,c2 = st.columns(2)
with c1:
    fig1 = px.bar(df_filtered, x="Nutrition_Status", color="Nutrition_Status",
                  color_discrete_sequence=COLORS, title="Status Count")
    fig1.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
with c2:
    fig2 = px.pie(df_filtered, names="Nutrition_Status", color_discrete_sequence=COLORS,
                  title="Status Share", hole=0.4)
    fig2.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig2, use_container_width=True)

# ================================
# GENDER & AREA
# ================================
st.markdown("<div class='section-title'>👤 Gender & Area Analysis</div>", unsafe_allow_html=True)
c1,c2 = st.columns(2)
with c1:
    gs = df_filtered.groupby(['Gender','Nutrition_Status']).size().reset_index(name='Count')
    fig_g = px.bar(gs, x='Gender', y='Count', color='Nutrition_Status',
                   barmode='group', color_discrete_sequence=COLORS, title="Gender-wise Status")
    fig_g.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig_g, use_container_width=True)
with c2:
    as_ = df_filtered.groupby(['Area_Type','Nutrition_Status']).size().reset_index(name='Count')
    fig_a = px.bar(as_, x='Area_Type', y='Count', color='Nutrition_Status',
                   barmode='stack', color_discrete_sequence=COLORS, title="Urban vs Rural")
    fig_a.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig_a, use_container_width=True)

# ================================
# AGE GROUP
# ================================
st.markdown("<div class='section-title'>🎂 Age Group Analysis</div>", unsafe_allow_html=True)
c1,c2 = st.columns(2)
with c1:
    age_s = df_filtered.groupby(['Age_Group','Nutrition_Status']).size().reset_index(name='Count')
    fig_age = px.bar(age_s, x='Age_Group', y='Count', color='Nutrition_Status',
                     barmode='stack', color_discrete_sequence=COLORS, title="Status by Age Group")
    fig_age.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig_age, use_container_width=True)
with c2:
    fig_ah = px.histogram(df_filtered, x='Age_months', color='Nutrition_Status',
                          nbins=24, color_discrete_sequence=COLORS, title="Age Distribution")
    fig_ah.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig_ah, use_container_width=True)

# ================================
# ANTHROPOMETRIC
# ================================
st.markdown("<div class='section-title'>📐 Anthropometric Analysis</div>", unsafe_allow_html=True)
c1,c2 = st.columns(2)
with c1:
    fig_box = px.box(df_filtered, x='Nutrition_Status', y='MUAC_cm',
                     color='Nutrition_Status', color_discrete_sequence=COLORS, title="MUAC Distribution")
    fig_box.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)
with c2:
    fig_vio = px.violin(df_filtered, x='Nutrition_Status', y='Weight_kg',
                        color='Nutrition_Status', color_discrete_sequence=COLORS,
                        box=True, title="Weight Distribution")
    fig_vio.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white", showlegend=False)
    st.plotly_chart(fig_vio, use_container_width=True)

fig_sc = px.scatter(df_filtered, x='Height_cm', y='Weight_kg',
                    color='Nutrition_Status', size='MUAC_cm',
                    hover_data=['City','Gender','Age_months'],
                    color_discrete_map=STATUS_COLOR,
                    title="Height vs Weight · bubble = MUAC")
fig_sc.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
st.plotly_chart(fig_sc, use_container_width=True)

# ================================
# CORRELATION
# ================================
st.markdown("<div class='section-title'>🔗 Correlation Matrix</div>", unsafe_allow_html=True)
corr = df_filtered[['Height_cm','Weight_kg','MUAC_cm','Age_months','BMI']].corr().round(2)
fig_corr = go.Figure(data=go.Heatmap(
    z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
    colorscale='Turbo', text=corr.values, texttemplate="%{text}", showscale=True
))
fig_corr.update_layout(title="Feature Correlation", paper_bgcolor="#0E1117",
                        font_color="white", height=400)
st.plotly_chart(fig_corr, use_container_width=True)

# ================================
# 🤖 DATA MINING INSIGHTS
# ================================
st.markdown("<div class='section-title'>🤖 Data Mining Insights</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

# Prediction Accuracy
c1.metric("Prediction Accuracy", f"{round(model_accuracy*100,2)}%")

# Cluster Count
c2.metric("Clusters Formed", df['Cluster'].nunique())

# Outliers
c3.metric("Outliers Detected", len(outliers))

st.write("Original Data Size:", df.shape)
st.write("After Removing Outliers:", df_no_outliers.shape)

# -------------------------------
# OUTLIER DISPLAY
# -------------------------------
with st.expander("⚠️ View Outliers"):
    st.write(outliers)

# ================================
# CITY HEATMAP & MAP
# ================================
st.markdown("<div class='section-title'>🔥 City × Status Heatmap</div>", unsafe_allow_html=True)
heatmap_data = pd.crosstab(df_filtered['City'], df_filtered['Nutrition_Status'])
if not heatmap_data.empty:
    fig_hm = ff.create_annotated_heatmap(
        z=heatmap_data.values, x=list(heatmap_data.columns),
        y=list(heatmap_data.index), colorscale="Turbo"
    )
    fig_hm.update_layout(paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("<div class='section-title'>🗺️ Maharashtra SAM Hotspots</div>", unsafe_allow_html=True)
city_coords = {
    "Mumbai":(19.0760,72.8777),"Pune":(18.5204,73.8567),
    "Nagpur":(21.1458,79.0882),"Nashik":(19.9975,73.7898),
    "Aurangabad":(19.8762,75.3433),"Thane":(19.2183,72.9781)
}
sam_counts = df[df['Nutrition_Status']=="SAM"].groupby("City").size().reset_index(name="SAM_Count")
sam_counts = sam_counts[sam_counts["City"].isin(city_coords)]
sam_counts["lat"] = sam_counts["City"].map(lambda x: city_coords[x][0])
sam_counts["lon"] = sam_counts["City"].map(lambda x: city_coords[x][1])
if not sam_counts.empty:
    fig_map = px.scatter_mapbox(sam_counts, lat="lat", lon="lon",
                                size="SAM_Count", color="SAM_Count", hover_name="City",
                                color_continuous_scale="Turbo", size_max=40, zoom=5)
    fig_map.update_layout(mapbox_style="carto-darkmatter", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig_map, use_container_width=True)

# ================================
# CITY-WISE BREAKDOWN
# ================================
st.markdown("<div class='section-title'>🏙️ City-wise Breakdown</div>", unsafe_allow_html=True)
c1,c2 = st.columns(2)
with c1:
    fig_city = px.histogram(df_filtered, x="City", color="Nutrition_Status",
                            barmode="stack", color_discrete_sequence=COLORS,
                            title="City-wise Status Distribution")
    fig_city.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig_city, use_container_width=True)
with c2:
    city_avg = df_filtered.groupby("City")[['MUAC_cm','Height_cm','Weight_kg']].mean().reset_index()
    city_avg_m = city_avg.melt(id_vars='City', var_name='Metric', value_name='Value')
    fig_line = px.line(city_avg_m, x='City', y='Value', color='Metric',
                       color_discrete_sequence=COLORS,
                       title="City-wise Avg Anthropometrics", markers=True)
    fig_line.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig_line, use_container_width=True)

# ================================
# CHILD LOOKUP
# ================================
st.markdown("<div class='section-title'>🔍 Individual Child Lookup</div>", unsafe_allow_html=True)
child_id = st.text_input("Enter Child ID:", "")
if child_id:
    result = df[df['Child_ID'].astype(str) == child_id.strip()]
    if not result.empty:
        r = result.iloc[0]
        sc = {"SAM":"🔴","MAM":"🟡","Normal":"🟢"}
        cc1,cc2,cc3,cc4 = st.columns(4)
        cc1.metric("Height (cm)", r['Height_cm'])
        cc2.metric("Weight (kg)", r['Weight_kg'])
        cc3.metric("MUAC (cm)",   r['MUAC_cm'])
        cc4.metric("Status", f"{sc.get(r['Nutrition_Status'],'')} {r['Nutrition_Status']}")
        st.dataframe(result, use_container_width=True)
    else:
        st.error("Child ID not found.")

# ================================
# EXPORT
# ================================
st.markdown("<div class='section-title'>📥 Export Data</div>", unsafe_allow_html=True)
csv_data = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Filtered CSV", csv_data,
                   file_name="filtered_data.csv", mime="text/csv")

with st.expander("⬇️ Download Star Schema Tables"):
    c1,c2,c3,c4 = st.columns(4)
    c1.download_button("Fact Table",   fact_nutrition.to_csv(index=False).encode(), "fact_nutrition.csv")
    c2.download_button("Dim Child",    dim_child.to_csv(index=False).encode(),      "dim_child.csv")
    c3.download_button("Dim Location", dim_location.to_csv(index=False).encode(),   "dim_location.csv")
    c4.download_button("Dim Status",   dim_status.to_csv(index=False).encode(),     "dim_status.csv")

# ================================
# PDF REPORT
# ================================
st.markdown("---")

def generate_pdf():
    file_path = "malnutrition_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Malnutrition Dashboard Report", styles['Title']))
    elements.append(Spacer(1,12))
    elements.append(Paragraph("Maharashtra Child Nutrition Analysis (0–5 Years)", styles['Normal']))
    elements.append(Spacer(1,20))
    sam_n  = len(df_filtered[df_filtered['Nutrition_Status']=='SAM'])
    mam_n  = len(df_filtered[df_filtered['Nutrition_Status']=='MAM'])
    norm_n = len(df_filtered[df_filtered['Nutrition_Status']=='Normal'])
    table_data = [
        ["Metric","Value"],
        ["Total Children",  str(len(df_filtered))],
        ["SAM (Severe)",    str(sam_n)],
        ["MAM (Moderate)",  str(mam_n)],
        ["Normal",          str(norm_n)],
        ["SAM Rate",        f"{sam_pct}%"],
        ["Avg MUAC (cm)",   str(avg_muac)],
        ["Avg Height (cm)", str(round(df_filtered['Height_cm'].mean(),2))],
        ["Avg Weight (kg)", str(round(df_filtered['Weight_kg'].mean(),2))],
    ]
    t = Table(table_data, colWidths=[200,200])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),rl_colors.HexColor("#1f1c2c")),
        ('TEXTCOLOR',(0,0),(-1,0),rl_colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.whitesmoke,rl_colors.lightgrey]),
        ('GRID',(0,0),(-1,-1),0.5,rl_colors.grey),
        ('FONTSIZE',(0,0),(-1,-1),11),
        ('PADDING',(0,0),(-1,-1),8),
    ]))
    elements.append(t)
    elements.append(Spacer(1,20))
    elements.append(Paragraph("Star Schema Tables:", styles['Heading2']))
    elements.append(Paragraph("fact_nutrition · dim_child · dim_location · dim_status · dim_date", styles['Normal']))
    elements.append(Spacer(1,10))
    elements.append(Paragraph("OLAP Operations Applied:", styles['Heading2']))
    elements.append(Paragraph("Roll-Up · Drill-Down · Slice · Dice · Pivot", styles['Normal']))
    doc.build(elements)
    return file_path

if st.button("📄 Generate PDF Report"):
    pdf_file = generate_pdf()
    with open(pdf_file,"rb") as f:
        st.download_button("⬇️ Download PDF", f, file_name="malnutrition_report.pdf")