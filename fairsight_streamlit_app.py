"""
FairSight — Streamlit Interactive Dashboard
Single-file Streamlit app to visualize the prototype outputs from the FairSight DEI platform.
Place the CSVs into a `data/` folder at the repo root or update paths in the sidebar.

Files expected (default paths):
- data/jds_simulated.csv
- data/jds_rewrites.csv
- data/performance_reviews_simulated.csv
- data/compensation_simulated.csv
- data/pay_parity_summary.csv
- data/dashboard_metrics.csv

To run locally:
1. Create a virtualenv and install requirements:
   pip install -r requirements.txt
   (requirements.txt should include: streamlit,pandas,plotly)
2. Run:
   streamlit run fairsight_streamlit_app.py

This app offers interactive filters, plots, tables, and CSV download options to include in your assignment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

st.set_page_config(page_title="FairSight DEI Dashboard", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Utility functions
# -----------------------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load: {path} — {e}")
        return pd.DataFrame()

def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def dataframe_download_link(df, filename="data.csv", link_text="Download CSV"):
    b = to_csv_bytes(df)
    b64 = base64.b64encode(b).decode()
    href = f"data:file/csv;base64,{b64}"
    return f"<a href=\"{href}\" download=\"{filename}\">{link_text}</a>"

# -----------------------------
# Sidebar: data path config
# -----------------------------
st.sidebar.header("Data configuration")
st.sidebar.write("Place CSVs in a `data/` folder or change paths below.")
default_paths = {
    "jds": "data/jds_simulated.csv",
    "jds_rewrites": "data/jds_rewrites.csv",
    "reviews": "data/performance_reviews_simulated.csv",
    "comp": "data/compensation_simulated.csv",
    "payparity": "data/pay_parity_summary.csv",
    "dashboard": "data/dashboard_metrics.csv"
}

jds_path = st.sidebar.text_input("Job descriptions CSV", value=default_paths['jds'])
jds_rewrites_path = st.sidebar.text_input("JD rewrites CSV", value=default_paths['jds_rewrites'])
reviews_path = st.sidebar.text_input("Performance reviews CSV", value=default_paths['reviews'])
comp_path = st.sidebar.text_input("Compensation CSV", value=default_paths['comp'])
payparity_path = st.sidebar.text_input("Pay parity CSV", value=default_paths['payparity'])
metrics_path = st.sidebar.text_input("Dashboard metrics CSV", value=default_paths['dashboard'])

# -----------------------------
# Load data
# -----------------------------
st.sidebar.markdown("---")
if st.sidebar.button("Reload data"):
    load_csv.cache_clear()

jds_df = load_csv(jds_path)
jds_rewrites_df = load_csv(jds_rewrites_path)
reviews_df = load_csv(reviews_path)
comp_df = load_csv(comp_path)
parity_df = load_csv(payparity_path)
metrics_df = load_csv(metrics_path)

# -----------------------------
# Header
# -----------------------------
st.title("FairSight — DEI Intelligence Platform (Prototype Dashboard)")
st.markdown(
    """
    Interactive dashboard for the FairSight prototype. Use the left panel to configure file paths and filters.
    The dashboard visualizes JD gendered language, review vagueness, pay parity and provides sample rewrites and downloads for annexure.
    """
)

# -----------------------------
# High-level KPIs
# -----------------------------
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

def metric_value_from_metrics_df(key, fallback="—"):
    if not metrics_df.empty and 'metric' in metrics_df.columns and 'value' in metrics_df.columns:
        row = metrics_df[metrics_df['metric'] == key]
        if not row.empty:
            return row['value'].iloc[0]
    return fallback

with col1:
    k1 = metric_value_from_metrics_df('Hiring Fairness (simulated score, 0-1)')
    st.metric("Hiring Fairness", k1)
with col2:
    k2 = metric_value_from_metrics_df('Promotion Equity (simulated score, 0-1)')
    st.metric("Promotion Equity", k2)
with col3:
    k3 = metric_value_from_metrics_df('Pay Parity Average (median across levels)')
    st.metric("Pay Parity (avg)", k3)
with col4:
    k4 = metric_value_from_metrics_df('JD Gendered Language Avg Score (positive -> masculine-leaning)')
    st.metric("JD Gendered Language", k4)

st.markdown("---")

# -----------------------------
# Job Descriptions Analysis
# -----------------------------
st.subheader("Job Description (JD) Analysis")
if jds_df.empty:
    st.info("No JD data loaded. Upload `jds_simulated.csv` to see JD analysis.")
else:
    st.markdown("**JD table (sample)**")
    st.dataframe(jds_df[['id','title','text','gender_score']].rename(columns={'id':'ID','title':'Title','text':'Text','gender_score':'GenderScore'}))

    st.markdown("**Filter & Visualize**")
    min_score = int(jds_df['gender_score'].min()) if 'gender_score' in jds_df.columns else -2
    max_score = int(jds_df['gender_score'].max()) if 'gender_score' in jds_df.columns else 2
    score_range = st.slider("Gender score range (positive = masculine-leaning)", min_value=min_score, max_value=max_score, value=(min_score,max_score))

    filtered_jds = jds_df[(jds_df['gender_score']>=score_range[0]) & (jds_df['gender_score']<=score_range[1])] if 'gender_score' in jds_df.columns else jds_df

    fig_jd = px.bar(filtered_jds, x='id', y='gender_score', hover_data=['title','text'], labels={'id':'JD ID','gender_score':'Gender Score'})
    fig_jd.update_layout(title='JD Gender Score (positive = masculine-leaning)', xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig_jd, use_container_width=True)

    st.markdown("**Top flagged JDs (most masculine-leaning)**")
    top = jds_df.sort_values('gender_score', ascending=False).head(5) if 'gender_score' in jds_df.columns else jds_df.head(5)
    st.table(top[['id','title','gender_score']].rename(columns={'id':'ID','title':'Title','gender_score':'GenderScore'}))

    if not jds_rewrites_df.empty:
        st.markdown("**Example rewrites**")
        merged = pd.merge(jds_df, jds_rewrites_df, left_on='id', right_on='id', how='left', suffixes=('','_rew'))
        sample_rewrites = merged[['id','title','text','rewritten_text']].rename(columns={'id':'ID','title':'Title','text':'Original','rewritten_text':'Rewritten'})
        st.dataframe(sample_rewrites)
        st.markdown(dataframe_download_link(sample_rewrites, filename='jds_rewrites_for_annexure.csv', link_text='Download JD rewrites CSV'), unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Performance Reviews Analysis
# -----------------------------
st.subheader("Performance Reviews")
if reviews_df.empty:
    st.info("No performance reviews loaded. Upload `performance_reviews_simulated.csv` to see analysis.")
else:
    st.markdown("**Review vagueness distribution**")
    if 'vagueness_score' in reviews_df.columns:
        fig = px.histogram(reviews_df, x='vagueness_score', nbins=8, title='Distribution of Review Vagueness Scores')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Sample reviews (gender + vagueness)**")
    if {'emp_id','gender','review','vagueness_score'}.issubset(reviews_df.columns):
        st.dataframe(reviews_df[['emp_id','gender','review','vagueness_score']].rename(columns={'emp_id':'EmpID','gender':'Gender','review':'Review','vagueness_score':'Vagueness'}))
        st.markdown(dataframe_download_link(reviews_df[['emp_id','gender','review','vagueness_score']], filename='reviews_for_annexure.csv', link_text='Download Reviews CSV'), unsafe_allow_html=True)
    else:
        st.dataframe(reviews_df.head(10))

st.markdown("---")

# -----------------------------
# Compensation & Pay Parity
# -----------------------------
st.subheader("Compensation & Pay Parity")
if comp_df.empty or parity_df.empty:
    st.info("No compensation or parity data available. Upload the compensation CSVs to visualize pay parity.")
else:
    st.markdown("**Median salary by level and gender**")
    if {'level','gender','salary'}.issubset(comp_df.columns):
        med = comp_df.groupby(['level','gender'])['salary'].median().reset_index()
        fig = px.bar(med, x='level', y='salary', color='gender', barmode='group', title='Median Salary by Level and Gender')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(dataframe_download_link(med, filename='median_salary_by_level_gender.csv', link_text='Download median salary CSV'), unsafe_allow_html=True)

    st.markdown("**Pay parity summary**")
    if {'level','median_male','median_female','parity_index'}.issubset(parity_df.columns):
        st.dataframe(parity_df.rename(columns={'median_male':'MedianMale','median_female':'MedianFemale','parity_index':'ParityIndex'}))
        st.markdown(dataframe_download_link(parity_df, filename='pay_parity_summary.csv', link_text='Download pay parity CSV'), unsafe_allow_html=True)
    else:
        st.dataframe(parity_df.head(10))

st.markdown("---")

# -----------------------------
# Interactive Scenario: Remediation Impact Simulator
# -----------------------------
st.subheader("Remediation Impact Simulator")
st.write("Simulate the impact of a targeted remediation (e.g., adjust female salaries at Senior level by X%) and observe parity change.")

if not comp_df.empty and 'level' in comp_df.columns and 'salary' in comp_df.columns and 'gender' in comp_df.columns:
    levels = comp_df['level'].unique().tolist()
    sel_level = st.selectbox('Select level to simulate remediation', options=levels)
    adjust_pct = st.slider('Adjust female salaries by (%)', min_value=0, max_value=30, value=5)

    sim_df = comp_df.copy()
    mask = (sim_df['level']==sel_level) & (sim_df['gender']=='F')
    sim_df.loc[mask,'salary'] = (sim_df.loc[mask,'salary'] * (1 + adjust_pct/100)).round().astype(int)

    sim_parity = sim_df.groupby('level').apply(lambda g: g[g['gender']=='F']['salary'].median() / g[g['gender']=='M']['salary'].median() if (g[g['gender']=='M']['salary'].median()>0 and not g[g['gender']=='F'].empty) else np.nan).reset_index()
    sim_parity.columns = ['level','sim_parity_index']

    merged = parity_df.merge(sim_parity, on='level', how='left') if 'level' in parity_df.columns else sim_parity
    st.dataframe(merged)
    st.markdown(dataframe_download_link(merged, filename='simulated_parity_impact.csv', link_text='Download simulated parity impact CSV'), unsafe_allow_html=True)
else:
    st.info("Compensation dataset is required for simulation.")

st.markdown("---")

# -----------------------------
# Annexure & Export
# -----------------------------
st.header("Annexure & Export")
st.write("Download processed CSVs and example visualizations to include in your assignment annexure.")
cols = st.columns(3)

with cols[0]:
    if not jds_df.empty:
        st.markdown(dataframe_download_link(jds_df, filename='jds_simulated.csv', link_text='Download jds_simulated.csv'), unsafe_allow_html=True)
    if not jds_rewrites_df.empty:
        st.markdown(dataframe_download_link(jds_rewrites_df, filename='jds_rewrites.csv', link_text='Download jds_rewrites.csv'), unsafe_allow_html=True)
with cols[1]:
    if not reviews_df.empty:
        st.markdown(dataframe_download_link(reviews_df, filename='performance_reviews_simulated.csv', link_text='Download performance_reviews_simulated.csv'), unsafe_allow_html=True)
    if not comp_df.empty:
        st.markdown(dataframe_download_link(comp_df, filename='compensation_simulated.csv', link_text='Download compensation_simulated.csv'), unsafe_allow_html=True)
with cols[2]:
    if not parity_df.empty:
        st.markdown(dataframe_download_link(parity_df, filename='pay_parity_summary.csv', link_text='Download pay_parity_summary.csv'), unsafe_allow_html=True)
    if not metrics_df.empty:
        st.markdown(dataframe_download_link(metrics_df, filename='dashboard_metrics.csv', link_text='Download dashboard_metrics.csv'), unsafe_allow_html=True)

st.markdown("---")
st.caption("Note: This dashboard uses prototype/simulated data. For a pilot with real companies, replace data files with anonymized exports and ensure appropriate data governance.")

# -----------------------------
# Footer: Run instructions
# -----------------------------
st.write("### How to run this on GitHub / locally")
st.markdown(
"""
1. Create a GitHub repo and add this file `fairsight_streamlit_app.py` at the root.
2. Add a `data/` folder with the CSVs (or edit the paths in the sidebar).
3. Add a `requirements.txt` with: `streamlit`, `pandas`, `plotly`.
4. Run locally: `streamlit run fairsight_streamlit_app.py`.
5. To share online: deploy on Streamlit Community Cloud — link your GitHub repo and specify the file to run.
"""
)

st.write("If you want, I can also generate a `requirements.txt`, a simple README.md, and a GitHub Actions workflow for deployment.")
