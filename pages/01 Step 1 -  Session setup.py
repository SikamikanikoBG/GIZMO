import json
import os
import shutil
import subprocess

import streamlit as st

import definitions

st.set_page_config(
    page_title="GIZMO - Setup",
    page_icon="random",
    layout="wide",
)

st.session_state["my_notes"] = st.sidebar.text_area("My notes", value=st.session_state["my_notes"])

tab_settings, tab_log = st.tabs(["Settings - Step 1", "Settings - Step 2"])

with tab_settings:
    # load projects
    params_path = f"{definitions.EXTERNAL_DIR}/params/"

    file_list = os.listdir(params_path)
    for file in file_list[:]:
        if '.json' not in file:
            file_list.remove(file)

    files_flags_dict = {}
    col1, col2, col3 = st.columns(3)

    input_dir = './input_data/'
    dir_list = next(os.walk(input_dir))[1]

    selected_project = st.sidebar.selectbox("Select existing project:", sorted(dir_list))
    input_data_path = f"{definitions.ROOT_DIR}/input_data/{selected_project}"

    file_list_input_folder = os.listdir(input_data_path)

    st.header(f"Settings for: {selected_project}")
    selected_param_file = str()
    for file in file_list:
        if selected_project in file:
            selected_param_file = file
    if not selected_param_file:
        shutil.copyfile(f"{params_path}params_new_project.json", f"{params_path}params_{selected_project}.json")
    else:
        st.caption(selected_param_file)

    definitions.selected_project = selected_project
    definitions.selected_param_file = selected_param_file

    # Display and edit json param file
    try:
        with open(os.path.join(params_path + definitions.selected_param_file), 'r', encoding='utf-8') as param_file:
            json_object = json.load(param_file)
            col1, col2 = st.columns(2)

            with st.form("Settings"):
                with col1:
                    st.subheader("Data processing settings")
                    try:
                        default_ix = st.session_state['input_df'].columns.tolist().index(json_object['criterion_column'])
                    except:
                        default_ix = 0
                    new_value_criterion_column = st.selectbox(
                        f"[ criterion_column ] Current value: {json_object['criterion_column']}",
                        st.session_state['input_df'].columns.tolist(), index=default_ix)
                    new_value_custom_calculations = st.text_input(
                        label=f"[ custom_calculations ]. Current value: {json_object['custom_calculations']}",
                        value=json_object['custom_calculations'], disabled=True)
                    new_value_main_table = st.selectbox(
                        f"[ main_table ]. Well, the name of the main data file for the project. Current value: {json_object['main_table']}",
                        file_list_input_folder)
                    new_value_additional_tables = st.text_input(
                        label=f"[ additional_tables ]. Jizzmo will left join them to the main table based on keys that you are specifying here.",
                        value=json_object['additional_tables'], disabled=True)

                    try:
                        default_ix = st.session_state['input_df'].columns.tolist().index(json_object['observation_date_column'])
                    except:
                        default_ix = 0
                    new_value_observation_date_column = st.selectbox(label=f"[ observation_date_column ] Current value: {json_object['observation_date_column']}",
                                                                     options=st.session_state[
                                                                         'input_df'].columns.tolist(), index=default_ix)
                    new_value_periods_to_exclude = st.multiselect(
                        label=f"[ periods_to_exclude ]. Example - no full performance period, or bad nb of cases etc. Current value: {json_object['periods_to_exclude']}",
                        options=sorted(st.session_state['input_df'][new_value_observation_date_column].unique()))
                    new_value_under_sampling = st.slider(
                        label=f"[ under_sampling ] is weighting the target rate in the "
                              f"data before analysing it. Default seeting is 1. If you are not sure - leave it like that... "
                              f"Current value {json_object['under_sampling']}",
                        min_value=0.5,
                        max_value=1.0,
                        step=0.1,
                        value=1.0)
                    new_value_optimal_binning_columns = st.multiselect(
                        label=f"[ Optimal Binning ] Which NUMERICAL! columns you would like GIZMO to cut into bins for the analysis? Current value: {json_object['optimal_binning_columns']}",
                        options=st.session_state['input_df'].columns.tolist(), default=json_object['optimal_binning_columns'])
                    new_value_missing_treatment = st.text_input(label=f"{'missing_treatment'}",
                                                                value=json_object['missing_treatment'],
                                                                disabled=True)
                    new_value_columns_to_include = st.multiselect(
                        label=f"[ Columns to include ] If some columns are excluded by GIZMO due to low correlation or other reason - select them here if you want to force GIZMO to include them. {'columns_to_include'}. Current value: {json_object['columns_to_include']}",
                        options=st.session_state['input_df'].columns.tolist(),
                        default=json_object['columns_to_include'])

                    new_value_columns_to_exclude = st.multiselect(
                        label=f"[ Columns to EXCLUDE ] Which columns to be excluded from the analysis? Example - ID, Phone or others. {'columns_to_include'}. Current value: {json_object['columns_to_exclude']}",
                        options=st.session_state['input_df'].columns.tolist(),
                        default=json_object['columns_to_exclude'])
                with col2:
                    st.subheader("Models training settings")
                    new_value_t1df = st.selectbox(
                        label=f"[ Temporal validation 1 ]. Current value: {json_object['t1df']}",
                        options=sorted(st.session_state['input_df'][new_value_observation_date_column].unique(),
                                       reverse=True))
                    new_value_t2df = st.selectbox(
                        label=f"[ Temporal validation 2 ]. Current value: {json_object['t2df']}",
                        options=sorted(st.session_state['input_df'][new_value_observation_date_column].unique(),
                                       reverse=True))
                    new_value_t3df = st.selectbox(
                        label=f"[ Temporal validation 3 ]. Current value: {json_object['t3df']}",
                        options=sorted(st.session_state['input_df'][new_value_observation_date_column].unique(),
                                       reverse=True))
                    new_value_cut_offs = st.text_input(
                        label=f"[ Scoring bands ]",
                        value=json_object['cut_offs'], disabled=True)

                    new_value_secondary_criterion_columns = st.text_input(
                        label=f"[ secondary_criterion_columns ]. In some graphs this will be "
                              f"visualized as well. Example - column to predict is in nb, and this can be the amount.",
                        value=json_object['secondary_criterion_columns'], disabled=True)
                    new_value_lr_features = st.text_input(
                        label=f"[ lr_features ]. The exact features to be used for LR.",
                        value=json_object['lr_features'], disabled=True)
                    new_value_lr_features_to_include = st.text_input(
                        label=f"[ lr_features_to_include ]",
                        value=json_object['lr_features_to_include'], disabled=True)
                    new_value_trees_features_to_exclude = st.text_input(
                        label=f"[ trees_features_to_exclude ]",
                        value=json_object['trees_features_to_exclude'], disabled=True)

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

                submitted = st.form_submit_button(f"Update {selected_project} settings")
                if submitted:
                    with open(os.path.join(params_path + selected_param_file), 'w',
                              encoding='utf-8') as output_param_file:
                        json.dump(json_object, output_param_file)
                        st.success("Settings are updated successfully")

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
