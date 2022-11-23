import codecs
import json
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sweetviz as sv

import definitions

st.set_page_config(
    page_title="ArDi Reports",
    page_icon="✅",
    layout="wide",
)

data_file = st.file_uploader("Load sample data from the main table.")
input_df = pd.read_parquet(data_file).sample(n=15)
#definitions.input_df = input_df



my_report = sv.analyze(input_df, pairwise_analysis='off')
my_report.show_html(filepath= "./pages/EDA.html", open_browser=False, layout="vertical", scale=1.0)

report_file = codecs.open("./pages/EDA.html", 'r')
page = report_file.read()
components.html(page, width=17400, height=1000, scrolling=True)