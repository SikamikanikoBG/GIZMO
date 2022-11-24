import codecs
import json
import os

import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
import streamlit.components.v1 as components
import sweetviz as sv
import pandas as pd
import pandas_profiling
import streamlit as st

from streamlit_pandas_profiling import st_profile_report


import definitions

st.set_page_config(
    page_title="GIZMO - Exploratory Data Analysis after data processing",
    page_icon="✅",
    layout="wide",
)

data_file = st.file_uploader("Load sample data from the main table.")
final_features = st.file_uploader("Upload final features.")
final_features = pd.read_pickle(final_features)
final_features = final_features[0:10]
# final_features.append(definitions.params["criterion_column"])
# st.write(final_features)

if st.button("Load graphs"):
    input_df = pd.read_parquet(data_file)
    input_df = input_df[final_features].copy()
    pr = input_df.profile_report()

    st_profile_report(pr)

    #components.html(profile, width=17400, height=1000, scrolling=True)

    #my_report = sv.analyze(input_df)
    #my_report.show_html(filepath="./pages/EDA.html", open_browser=False, layout="vertical", scale=1.0)

    #report_file = codecs.open("./pages/EDA.html", 'r')
    #page = report_file.read()
    #components.html(page, width=17400, height=1000, scrolling=True)
