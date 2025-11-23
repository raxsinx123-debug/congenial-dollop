"""
FairSight — Clean Streamlit Dashboard (minimal dependencies)

This version avoids plotly and uses only:
- streamlit
- pandas
- numpy
- openpyxl (for reading Excel)

It will attempt to read /mnt/data/fairsight_dataset.xlsx first; if not found,
it will read CSVs from a local `data/` folder placed alongside this app.
This file is intended to be shareable publicly with minimal runtime errors.

To run locally:
1. python -m venv venv
2. source venv/bin/activate  # or venv\Scripts\activate on Windows
3. pip install -r requirements.txt
4. streamlit run fairsight_streamlit_app_clean.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
import base64

st.set_page_config(page_title="FairSight — Clean DEI Dashboard", layout="wide")

st.title("FairSight — Clean DEI Dashboard (Prototype)")
st.markdown("This dashboard uses the prototype dataset. It prefers an Excel workbook at `/mnt/data/fairsight_dataset.xlsx` or CSVs in `data/`.")

# Paths
excel_path = Path("/mnt/data/fairsight_dataset.xlsx")
data_folder = Path("data")

# Utility: load sheet or csv
def load_datasets():
    datasets = {}
    if excel_path.exists() and excel_path.stat().st_size > 0:
        try:
            xls = pd.read_excel(excel_path, sheet_name=None)
            # Normalize sheet names to expected keys if present
            # Accept both the original sheet names used earlier.
            for k, v in xls.items():
                datasets[k.lower()] = v
            st.sidebar.success("Loaded data from /mnt/data/fairsight_dataset.xlsx")
            return datasets
        except Exception as e:
            st.sidebar.error(f"Error reading Excel file: {e} — falling back to CSVs in ./data/")
    # Fallback to CSVs
    csv_map = {
        "jds_simulated": "jds_simulated.csv",
        "jds_rewrites": "jds_rewrites.csv",
        "performance_reviews": "performance_reviews_simulated.csv",
        "compensation": "compensation_simulated.csv",
        "pay_parity_summary": "pay_parity_summary.csv",
        "dashboard_metrics": "dashboard_metrics.csv"
    }
    for key, fname in csv_map.items():
        p = data_folder / fname
        if p.exists() and p.stat().st_size > 0:
            try:
                datasets[key] = pd.read_csv(p)
            except Exception as e:
                datasets[key] = pd.DataFrame()
        else:
            datasets[key] = pd.DataFrame()
    st.sidebar.info("Loaded CSVs from ./data/ (if present)")
    return datasets

data = load_datasets()

# Helper: create download link for dataframe
def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Display Key Metrics
st.header("Key Metrics (Prototype)")
metrics = data.get("dashboard_metrics", pd.DataFrame())
if not metrics.empty and {'metric','value'}.issubset(metrics.columns):
    cols = st.columns(4)
    keys = metrics['metric'].tolist()
    vals = metrics['value'].tolist()
    for c, k, v in zip(cols, keys, vals):
        c.metric(k, v)
else:
    st.info("No dashboard_metrics found. You can upload an Excel file at /mnt/data/fairsight_dataset.xlsx or place CSVs in the `data/` folder.")

st.markdown("---")

# JD Analysis
st.subheader("Job Descriptions (JDs)")
jds = data.get("jds_simulated", pd.DataFrame())
if jds.empty:
    st.write("No JD data available.")
else:
    st.write("Sample JDs:")
    st.dataframe(jds.head(20))
    if 'gender_score' in jds.columns:
        st.write("Gender score distribution (positive = masculine-leaning):")
        st.bar_chart(jds['gender_score'])

    # Show rewrites if available
    rewrites = data.get("jds_rewrites", pd.DataFrame())
    if not rewrites.empty:
        st.write("Original & Rewritten JDs (sample):")
        merged = pd.merge(jds, rewrites, on='id', how='left') if 'id' in jds.columns and 'id' in rewrites.columns else None
        if merged is not None:
            st.dataframe(merged[['id','title','text','rewritten_text']].head(20))
            st.markdown(get_table_download_link(merged[['id','title','text','rewritten_text']], "jds_rewrites_for_annexure.csv"), unsafe_allow_html=True)

st.markdown("---")

# Performance reviews
st.subheader("Performance Reviews")
reviews = data.get("performance_reviews", pd.DataFrame())
if reviews.empty:
    st.write("No performance reviews data found.")
else:
    st.write("Sample reviews:")
    st.dataframe(reviews.head(20))
    if 'vagueness_score' in reviews.columns:
        st.write("Vagueness score histogram:")
        st.bar_chart(reviews['vagueness_score'].value_counts().sort_index())

st.markdown("---")

# Compensation & Pay Parity
st.subheader("Compensation & Pay Parity")
comp = data.get("compensation", pd.DataFrame())
parity = data.get("pay_parity_summary", pd.DataFrame())
if comp.empty:
    st.write("No compensation dataset found.")
else:
    st.write("Compensation sample:")
    st.dataframe(comp.head(20))
    if {'level','gender','salary'}.issubset(comp.columns):
        pivot = comp.pivot_table(index='level', columns='gender', values='salary', aggfunc='median')
        st.write("Median salary by level & gender:")
        st.dataframe(pivot)
        # Simple chart
        st.write("Median salary chart:")
        st.bar_chart(pivot.fillna(0))

if not parity.empty:
    st.write("Pay parity summary:")
    st.dataframe(parity)
    st.markdown(get_table_download_link(parity, "pay_parity_summary.csv"), unsafe_allow_html=True)

st.markdown("---")

# Simple remediation simulator
st.subheader("Remediation Simulator")
if not comp.empty and {'level','gender','salary'}.issubset(comp.columns):
    levels = comp['level'].unique().tolist()
    sel = st.selectbox("Select level to adjust female salaries", options=levels)
    pct = st.slider("Increase female salaries by (%)", min_value=0, max_value=50, value=5)
    sim = comp.copy()
    mask = (sim['level'] == sel) & (sim['gender'] == 'F')
    sim.loc[mask, 'salary'] = (sim.loc[mask, 'salary'] * (1 + pct/100)).astype(int)
    sim_parity = sim.groupby('level').apply(lambda g: g[g['gender']=='F']['salary'].median() / g[g['gender']=='M']['salary'].median() if (not g[g['gender']=='M'].empty and not g[g['gender']=='F'].empty) else np.nan).reset_index()
    sim_parity.columns = ['level','sim_parity_index']
    st.write("Simulated parity after adjustment:")
    st.dataframe(sim_parity)
    st.markdown(get_table_download_link(sim_parity, "simulated_parity.csv"), unsafe_allow_html=True)
else:
    st.write("Compensation dataset required for simulator.")

st.markdown("---")
st.write("Notes:")
st.write("- This is a simplified prototype for demonstration and assignment purposes.")
st.write("- For production, use secure data ingestion, federated analytics, thorough model validation and human governance.")

st.write("Assignment brief included in repository: `/IIMR_NSGC_AI-Assignment_Guidelines.pdf`")
