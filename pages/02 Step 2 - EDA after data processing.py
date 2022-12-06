import codecs
import json
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sweetviz as sv

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
        input_df = pd.read_parquet(f"{output_data_path}/output_data_file.parquet")

final_features = pd.read_pickle(f"{output_data_path}/final_features.pkl")
final_features = final_features
final_features_criterion = final_features
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
    from matplotlib import pyplot as plt
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree

    # Prepare the data data
    #iris = datasets.load_iris()
    X = input_df[final_features]
    y = input_df[criterion_column]
    # Fit the classifier with default hyper-parameters
    clf = DecisionTreeClassifier(max_depth=4, max_features=10,random_state=1234)
    model = clf.fit(X, y)

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(clf,
                       feature_names=final_features,
                       #class_names=criterion_column,
                       filled=True
                       )

    st.pyplot(fig)

    # ---------------------------------------------------

"""    from dtreeviz.trees import dtreeviz  # remember to load the package

    viz = dtreeviz(clf, X, y,
                   target_name="target",
                   feature_names=final_features,
                   class_names=list(criterion_column))
    st.pyplot(viz)"""


