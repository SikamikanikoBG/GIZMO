import json
import os

import streamlit as st
import definitions

st.set_page_config(
    page_title="Jizzmo",
    page_icon="✅",
    layout="wide",
)

global selected_project

st.image("gizmo_logo.png")
st.header("Welcome to Jizzmo!")
st.write("The data driven business decisions made simple")


#for file in file_list:
#    if "ardi" in file:
#        try:
#            with open(os.path.join(params_path + file), 'r', encoding='utf-8') as param_file:
##                json_object = json.load(param_file)
#                files_flags_dict[file] = json_object['flag_trade']
#
#        except Exception as e:
#            st.write(f"ERROR: Issue with {file}: {e}")
#
#st.caption("Current statuses:")
#st.write(files_flags_dict)
