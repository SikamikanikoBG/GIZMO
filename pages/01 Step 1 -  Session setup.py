import json
import os
import shutil
import subprocess

import pandas as pd
import streamlit as st

import definitions

st.set_page_config(
    page_title="GIZMO - Setup",
    page_icon="random",
    layout="wide",
)

if "my_notes" not in st.session_state:
    st.session_state["my_notes"] = ""
if "input_df" not in st.session_state:
    st.session_state['input_df'] = pd.DataFrame()
if "selected_param_file" not in st.session_state:
    st.session_state["selected_param_file"] = None
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
    dir_list = sorted(next(os.walk(input_dir))[1])
    dir_list.append("None")
    default_select = dir_list.index("None")

    project_selected_properly = False

    selected_project = st.selectbox(f"Select existing project:", dir_list, index=default_select)

    input_data_path = f"{definitions.ROOT_DIR}/input_data/{selected_project}"

    try:
        file_list_input_folder = os.listdir(input_data_path)
        project_selected_properly = True
    except Exception as e:
        st.warning(f"Dude, this project has no directory in our input folder... ERROR: {e}")
        st.stop()
    selected_project_input_data = st.selectbox("Select the input data:", file_list_input_folder)

    if st.button("Load settings and data"):

        if len(st.session_state["input_df"]) == 0:
            if ".csv" in selected_project_input_data:
                st.session_state['input_df'] = pd.read_csv(f"{input_data_path}/{selected_project_input_data}")
            elif ".parquet" in selected_project_input_data:
                st.session_state['input_df'] = pd.read_parquet(f"{input_data_path}/{selected_project_input_data}")

        st.header(f"Settings for: {selected_project}")
        selected_param_file = str()
        for file in file_list:
            if selected_project in file:
                selected_param_file = file
        if not selected_param_file:
            shutil.copyfile(f"{params_path}params_new_project.json", f"{params_path}params_{selected_project}.json")
        else:
            st.caption(selected_param_file)

        st.session_state["selected_project"] = selected_project
        st.session_state["selected_param_file"] = selected_param_file
        # Display and edit json param file
    try:
        with open(os.path.join(params_path + st.session_state.selected_param_file), 'r', encoding='utf-8') as param_file:
            json_object = json.load(param_file)
            col1, col2 = st.columns(2)

            with st.form("Settings"):
                with col1:
                    st.subheader("Data processing settings")
                    tolist = st.session_state['input_df'].columns.tolist()
                    tolist.append("")
                    try:
                        default_ix = tolist.index(json_object['criterion_column'])
                    except:
                        default_ix = 0
                    new_value_criterion_column = st.selectbox(
                        f"[ criterion_column ] Current value: {json_object['criterion_column']}",
                        tolist, index=default_ix)
                    new_value_custom_calculations = st.text_input(
                        label=f"[ custom_calculations ]. Current value: {json_object['custom_calculations']}",
                        value=json_object['custom_calculations'], disabled=True)
                    new_value_main_table = st.selectbox(
                        f"[ main_table ]. Well, the name of the main data file for the project. Current value: {json_object['main_table']}",
                        file_list_input_folder)
                    new_value_additional_tables = st.text_input(
                        label=f"[ additional_tables ]. Gizzmo will left join them to the main table based on keys that you are specifying here.",
                        value=json_object['additional_tables'], disabled=True)

                    try:
                        default_ix = tolist.index(json_object['observation_date_column'])
                    except:
                        default_ix = 0
                    new_value_observation_date_column = st.selectbox(label=f"[ observation_date_column ] Current value: {json_object['observation_date_column']}",
                                                                     options=tolist, index=default_ix)
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
                        options=tolist, default=json_object['optimal_binning_columns'])
                    new_value_missing_treatment = st.text_input(label=f"{'missing_treatment'}",
                                                                value=json_object['missing_treatment'],
                                                                disabled=True)
                    default_col_incl = json_object['columns_to_include']

                    label = f"[ Columns to include ] If some columns are excluded by GIZMO due to low correlation or other reason - select them here if you want to force GIZMO to include them. {'columns_to_include'}. Current value: {json_object['columns_to_include']}"
                    if len(json_object['columns_to_include']) == 0:
                        new_value_columns_to_include = st.multiselect(label=label, options=tolist)
                    else:
                        for el in json_object['columns_to_include']:
                            if el not in tolist:
                                json_object['columns_to_include'].remove(el)
                                st.warning(f"Warning: {el} was specified in the param, but was not found in the columns of the dataset.")
                        new_value_columns_to_include = st.multiselect(label=label, options=tolist, default=json_object['columns_to_include'])

                    label = f"[ Columns to EXCLUDE ] Which columns to be excluded from the analysis? Example - ID, Phone or others. {'columns_to_include'}. Current value: {json_object['columns_to_exclude']}"
                    if len(json_object['columns_to_exclude']) == 0:
                        st.warning("YES")
                        new_value_columns_to_exclude = st.multiselect(label=label, options=tolist)
                    else:
                        for el in json_object['columns_to_exclude']:
                            if el not in tolist:
                                json_object['columns_to_exclude'].remove(el)
                                st.warning(
                                    f"Warning: {el} was specified in the param, but was not found in the columns of the dataset.")
                        new_value_columns_to_exclude = st.multiselect(label=label, options=tolist, default=json_object['columns_to_exclude'])
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
                st.write(json_object)

                if submitted:
                    st.warning(st.session_state.selected_param_file)
                    with open(os.path.join(f"{params_path}{st.session_state.selected_param_file}"), 'w',
                              encoding='utf-8') as output_param_file:
                        json.dump(json_object, output_param_file)
                        st.success("Settings are updated successfully")

                # definitions.params = json_object
    except Exception as e:
        st.write(f"ERROR: {e}")

with tab_log:
    if st.button("Run GIZMO data preparation"):
        with st.spinner("Running data preparation"):
            subprocess.call(
                ["python", "main.py", "--project", f"{selected_project}", "--data_prep_module", "standard"],
                stdout=open(f"{definitions.EXTERNAL_DIR}/logs/data_prep_{selected_project}.txt", "a"))
        st.success('Done! Check the log!')

        log_file = open(f"{definitions.EXTERNAL_DIR}/logs/data_prep_{selected_project}.txt", "r")
        st.code(log_file.read())
