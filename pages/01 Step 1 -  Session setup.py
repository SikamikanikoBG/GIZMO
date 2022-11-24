import codecs
import json
import os
import subprocess

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sweetviz as sv

import definitions

st.set_page_config(
    page_title="GIZMO - Setup",
    page_icon="random",
    layout="wide",
)

tab_sample_data, tab_settings, tab_log = st.tabs(["Sample data", "Settings", "Log file"])

with tab_sample_data:
    data_file = st.file_uploader("Load 2000 rows sample data from the main table. Upload here the project dataset."
                                 "Supported filetypes: csv, txt, parquet, feather, pickle.")

    if data_file:
        if "parquet" in str(data_file):
            input_df = pd.read_parquet(data_file)

        elif "csv" in str(data_file):
            input_df = pd.read_csv(data_file)

        elif "txt" in str(data_file):
            input_df = pd.read_csv(data_file)

        elif "feather" in str(data_file):
            input_df = pd.read_feather(data_file)

        else:
            input_df = pd.read_pickle(data_file)

        st.info(f"Data loaded with {len(input_df)} records.")
        if len(input_df) > 20000:
            input_df = input_df.sample(n=20000)

        target_feature = st.selectbox("Select which column is used for predictions: ", input_df.columns)
        if st.button("Use this column"):
            my_report = sv.analyze(input_df, target_feat=target_feature)
        else:
            my_report = sv.analyze(input_df)

        my_report.show_html(filepath="./pages/EDA.html", open_browser=False, layout="vertical", scale=1.0)

        report_file = codecs.open("./pages/EDA.html", 'r')
        page = report_file.read()
        components.html(page, width=1400, height=1000, scrolling=True)

with tab_settings:
    # load projects
    params_path = f"{definitions.EXTERNAL_DIR}/params/"

    file_list = os.listdir(params_path)

    files_flags_dict = {}
    col1, col2, col3 = st.columns(3)

    project_list = []
    for proj in file_list:
        if '.json' in proj:
            _, b = proj.split("params_")
            a, _ = b.split(".json")
            project_list.append(a)

    selected_project = st.sidebar.selectbox("Select existing project:", sorted(project_list))
    input_data_path = f"{definitions.ROOT_DIR}/input_data/{selected_project}"
    file_list_input_folder = os.listdir(input_data_path)

    st.header(selected_project)
    selected_param_file = str()
    for file in file_list:
        if selected_project in file:
            selected_param_file = file
    st.caption(selected_param_file)

    definitions.selected_project = selected_project
    definitions.selected_param_file = selected_param_file

    # Display and edit json param file
    try:
        with open(os.path.join(params_path + definitions.selected_param_file), 'r', encoding='utf-8') as param_file:
            json_object = json.load(param_file)
            col1, col2, col_spacer, col3, col4 = st.columns(5)

            with st.form("Settings"):

                with col1:
                    st.markdown("""#### Data processing settings""")
                    new_value_criterion_column = st.text_input(label=f"{'criterion_column'}.",
                                                               value=json_object['criterion_column'])
                    new_value_custom_calculations = st.text_input(label=f"{'custom_calculations'}",
                                                                  value=json_object['custom_calculations'])

                    new_value_main_table = st.text_input(
                        label=f"{'main_table'}. Well, the name of the main data file for the project.",
                        value=json_object['main_table'])
                    new_value_additional_tables = st.text_input(
                        label=f"{'additional_tables'}. Jizzmo will left join them to the main table based on keys that you are specifying here.",
                        value=json_object['additional_tables'])
                    new_value_observation_date_column = st.text_input(label=f"{'observation_date_column'}",
                                                                      value=json_object['observation_date_column'])
                    new_value_periods_to_exclude = st.text_input(
                        label=f"{'periods_to_exclude'}. Example - no full performance period, or bad nb of cases etc.",
                        value=json_object['periods_to_exclude'])
                    new_value_under_sampling = st.text_input(label=f"{'under_sampling'}",
                                                             value=json_object['under_sampling'])
                    new_value_optimal_binning_columns = st.text_input(label=f"{'optimal_binning_columns'}",
                                                                      value=json_object['optimal_binning_columns'])
                    new_value_missing_treatment = st.text_input(label=f"{'missing_treatment'}",
                                                                value=json_object['missing_treatment'])
                    new_value_columns_to_include = st.text_input(label=f"{'columns_to_include'}",
                                                                 value=json_object['columns_to_include'])
                with col2:
                    st.markdown("""#### Based on sample data:""")
                    new_value_criterion_column = st.selectbox(f"{'criterion_column'}", input_df.columns.tolist())
                    new_value_main_table = st.selectbox(f"{'main_table'}", file_list_input_folder)
                with col3:
                    st.markdown("""#### Models training settings""")
                    new_value_t1df = st.text_input(
                        label=f"{'t1df'}. Latest observation periods on which to test the models,.",
                        value=json_object['t1df'])
                    new_value_t2df = st.text_input(
                        label=f"{'t2df'}. Latest observation periods on which to test the models,.",
                        value=json_object['t2df'])
                    new_value_t3df = st.text_input(
                        label=f"{'t3df'}. Latest observation periods on which to test the models,.",
                        value=json_object['t3df'])
                    new_value_cut_offs = st.text_input(
                        label=f"{'cut_offs'}..",
                        value=json_object['cut_offs'])

                    new_value_secondary_criterion_columns = st.text_input(
                        label=f"{'secondary_criterion_columns'}. In some graphs this will be "
                              f"visualized as well. Example - column to predict is in nb, and this can be the amount.",
                        value=json_object['secondary_criterion_columns'])
                    new_value_columns_to_exclude = st.text_input(
                        label=f"{'columns_to_exclude'}. Those columns will not be used for predicting.",
                        value=json_object['columns_to_exclude'])
                    new_value_lr_features = st.text_input(
                        label=f"{'lr_features'}. The exact features to be used for LR.",
                        value=json_object['lr_features'])
                    new_value_lr_features_to_include = st.text_input(
                        label=f"{'lr_features_to_include'}. .",
                        value=json_object['lr_features_to_include'])
                    new_value_trees_features_to_exclude = st.text_input(
                        label=f"{'trees_features_to_exclude'}. .",
                        value=json_object['trees_features_to_exclude'])

                json_object["criterion_column"] = new_value_criterion_column
                json_object["missing_treatment"] = new_value_missing_treatment
                json_object["main_table"] = new_value_main_table
                json_object["custom_calculations"] = new_value_custom_calculations
                json_object["additional_tables"] = new_value_additional_tables
                json_object["observation_date_column"] = new_value_observation_date_column
                json_object["secondary_criterion_columns"] = new_value_secondary_criterion_columns
                json_object["t1df"] = new_value_t1df
                json_object["t2df"] = new_value_t2df
                json_object["t3df"] = new_value_t3df
                json_object["periods_to_exclude"] = new_value_periods_to_exclude
                json_object["columns_to_exclude"] = new_value_columns_to_exclude
                json_object["lr_features"] = new_value_lr_features
                json_object["lr_features_to_include"] = new_value_lr_features_to_include
                json_object["trees_features_to_exclude"] = new_value_trees_features_to_exclude
                json_object["cut_offs"] = new_value_cut_offs
                json_object["under_sampling"] = new_value_under_sampling
                json_object["optimal_binning_columns"] = new_value_optimal_binning_columns
                json_object["columns_to_include"] = new_value_columns_to_include

                new_project_name = st.text_input(label=f"Enter the name of the new project")
                submitted_new = st.form_submit_button("Save as new project")
                submitted = st.form_submit_button("Update current project")
                if submitted:
                    with open(os.path.join(params_path + selected_param_file), 'w',
                              encoding='utf-8') as output_param_file:
                        json.dump(json_object, output_param_file)
                if submitted_new:
                    new_param_file_name = f"params_{new_project_name}.json"
                    with open(os.path.join(params_path + new_param_file_name), 'w',
                              encoding='utf-8') as output_param_file:
                        json.dump(json_object, output_param_file)
                definitions.params = json_object
    except Exception as e:
        st.write(f"ERROR: Issue with {file}: {e}")

with tab_log:
    if st.button("Run GIZMO data preparation"):
        with st.spinner("Running data preparation"):
            subprocess.call(
                ["python", "main.py", "--project", f"{selected_project}", "--data_prep_module", "standard"],
                stdout=open(f"{definitions.EXTERNAL_DIR}/logs/data_prep_{selected_project}.txt", "a"))
        st.success('Done! Check the log!')

        log_file = open(f"{definitions.EXTERNAL_DIR}/logs/data_prep_{selected_project}.txt", "r")
        st.code(log_file.read())
