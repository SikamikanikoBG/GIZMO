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
    page_icon="random",
    layout="wide",
)

# load projects
params_path = f"{definitions.EXTERNAL_DIR}/params/"
file_list = os.listdir(params_path)
files_flags_dict = {}

project_list = []
for proj in file_list:
    if '.json' in proj:
        _, b = proj.split("params_")
        a, _ = b.split(".json")
        project_list.append(a)

selected_project = st.sidebar.selectbox("Select existing project:", sorted(project_list))
output_data_path = f"{definitions.ROOT_DIR}/output_data/{selected_project}"
for file in file_list:
    if selected_project in file:
        with open(os.path.join(params_path + file), 'r', encoding='utf-8') as param_file:
            json_object = json.load(param_file)
            criterion_column = json_object["criterion_column"]

with st.spinner("Loading output data"):
    try:
        input_df = pd.read_parquet(f"{output_data_path}/output_data_file_full.parquet")
    except:
        input_df = pd.read_parquet(f"{output_data_path}/output_data_file.parquet")
st.success("Data loaded")

final_features = pd.read_pickle(f"{output_data_path}/final_features.pkl")
final_features = final_features[0:10]
final_features.append(criterion_column)
# final_features.append(definitions.params["criterion_column"])
# st.write(final_features)

if st.sidebar.button("Generate EDA"):
    input_df = input_df[final_features].copy()

    with st.spinner("Generating graphs"):
        # pr = input_df.profile_report()
        # st_profile_report(pr)
        # components.html(profile, width=1400, height=1000, scrolling=True)

        my_report = sv.analyze(input_df, target_feat=criterion_column)
        my_report.show_html(filepath=f"{output_data_path}/EDA.html", open_browser=False, layout="vertical", scale=1.0)
    st.success("Graphs generated!")

    report_file = codecs.open(f"{output_data_path}/EDA.html", 'r')
    page = report_file.read()
    components.html(page, width=1400, height=1000, scrolling=True)

if st.sidebar.button("Load existing EDA"):
    report_file = codecs.open(f"{output_data_path}/EDA.html", 'r')
    page = report_file.read()
    components.html(page, width=1400, height=1000, scrolling=True)