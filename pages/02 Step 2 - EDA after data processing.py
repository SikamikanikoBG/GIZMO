import codecs
import json
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sweetviz as sv
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


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
            st.caption(f"Target column: {criterion_column}")

with st.spinner("Loading output data"):
    try:
        input_df = pd.read_parquet(f"{output_data_path}/output_data_file_full.parquet")
    except:
        try:
            input_df = pd.read_parquet(f"{output_data_path}/output_data_file.parquet")
        except Exception as e:
            st.warning(f"Bate, no output file for this project. Have you ran - Step 1A, session setup, "
                       f"and 1B - Data preparation? Gledai gi malko tiq neshta, de!\n"
                       f"{e}")
            st.stop()

final_features = pd.read_pickle(f"{output_data_path}/final_features.pkl")
#final_features = final_features
final_features_criterion = final_features.copy()
final_features_criterion.append(criterion_column)

input_df = input_df[final_features_criterion].copy()

tab_eda, tab_segmentation = st.tabs(['Exploratory Data Analysis', 'Quick segmentation'])
with tab_eda:
    if st.button("Generate EDA"):
        with st.spinner("Generating graphs"):
            my_report = sv.analyze(input_df, target_feat=criterion_column)
            my_report.show_html(filepath=f"{output_data_path}/EDA.html", open_browser=False, layout="vertical", scale=1.0)
        st.success("Graphs generated!")

        report_file = codecs.open(f"{output_data_path}/EDA.html", 'r')
        page = report_file.read()
        components.html(page, width=1400, height=1000, scrolling=True)

    if st.button("Load existing EDA"):
        with st.spinner("Generating EDA graphs"):
            report_file = codecs.open(f"{output_data_path}/EDA.html", 'r')
            page = report_file.read()
            components.html(page, width=1400, height=1000, scrolling=True)

with tab_segmentation:
    with st.form("Segmentation settings"):
        # settings
        st.sidebar.info(f"Nb final features {len(final_features)}\n"
                        f"Dataset: {len(input_df)} nb")
        final_features_nodivs = []
        exclude_divs = st.sidebar.selectbox("Exclude ratios the segmentation?", ["No", "Yes"])
        if exclude_divs == "Yes":
            for feat in final_features:
                if '_div_' not in feat:
                    final_features_nodivs.append(feat)
        else:
            final_features_nodivs = final_features.copy()
        seled_feats = st.sidebar.multiselect("Manually select features?", sorted(final_features_nodivs))
        if seled_feats:
            final_features_nodivs = seled_feats.copy()
        max_dept_selected = st.sidebar.slider("Maximum depth", min_value=2, max_value=6, value=3)
        text_size_selected = st.sidebar.slider("Text size of the graph", min_value=7, max_value=14, value=10)

        if st.form_submit_button("Make segmentation!"):
            X = input_df[final_features_nodivs]
            y = input_df[criterion_column]
            # Fit the classifier with default hyper-parameters
            clf = DecisionTreeClassifier(max_depth=max_dept_selected, random_state=1234)
            model = clf.fit(X, y)

            fig = plt.figure(figsize=(50, 40))
            _ = tree.plot_tree(clf,
                               feature_names=final_features_nodivs,
                               fontsize=text_size_selected,
                               proportion=True,
                               #class_names=criterion_column,
                               filled=True
                               )

            st.write(fig)



