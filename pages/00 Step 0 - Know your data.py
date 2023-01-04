import codecs
import configparser
import os
import re

import pandas as pd
import pymssql
import streamlit as st
import streamlit.components.v1 as components
import sweetviz as sv

st.set_page_config(
    page_title="GIZMO - Setup",
    page_icon="random",
    layout="wide",
)

st.write("Load from Data and Analytics data server:")

if "input_df_org" not in st.session_state:
    st.session_state["input_df_org"] = pd.DataFrame()
if "my_notes" not in st.session_state:
    st.session_state["my_notes"] = ""

sample_size = st.sidebar.slider("Select the sample size you want to load:", 0, 100, 10, 10)
st.session_state["my_notes"] = st.sidebar.text_area("My notes", value=st.session_state["my_notes"])

st.header("Upload a file:")
data_file = st.file_uploader("Load 20000 rows sample data from the main table. Upload here the project dataset."
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

    st.session_state['input_df_org'] = input_df

st.header("Load from D&A SQL db:")
with st.form("SQL connection"):
    sql_db = st.selectbox("Database", ["DWH", "DWH_Stage", "DWH_temp"], index=1)
    sql_table = st.text_input("Table")
    sql_query = st.text_input(f"Custom query, default: select * from {sql_table}")
    # Load tables from SQL + additional modifications
    config = configparser.ConfigParser()
    # Define the connection using variables pulled from secret
    connection = pymssql.connect(
        server="10.128.11.98",
        user="pysqluser",
        password="CondaSnak3",
        database=sql_db)
    # Set up the cursor and execute an example query
    cur = connection.cursor()

    if sql_query:
        query = sql_query
    else:
        query = f"SELECT * from {sql_table}"

    submitted = st.form_submit_button("Submit")
    if submitted:
        try:
            input_df = pd.read_sql(query, connection)
            st.session_state['input_df_org'] = input_df
        except Exception as e:
            st.warning(e)
            st.warning("These db and table exist??? The Custom query is OK??? Gledai gi malko tiya raboti, de...")
        finally:
            cur.close()
            connection.close()

st.info(f"Data loaded with {len(st.session_state['input_df_org'])} records.")
#st.session_state['input_df_org'] = st.session_state['input_df'].copy()
st.session_state['input_df'] = st.session_state['input_df_org'].sample(frac=(sample_size/100)).copy()
st.info(f"Random sampled data {len(st.session_state['input_df'])} records based on the chosen sample size {sample_size}%.")

st.caption("Sanity check with top 5 rows:")
st.write(st.session_state['input_df'].head())


with st.form("Target variable"):
    target_feature = st.selectbox("Select which column is used for predictions: ", st.session_state['input_df'].columns)
    if st.form_submit_button(f"EDA with target"):
        st.info(target_feature)
        my_report = sv.analyze(st.session_state['input_df'], target_feat=target_feature)
        my_report.show_html(filepath="./pages/EDA.html", open_browser=False, layout="vertical", scale=1.0)

        report_file = codecs.open("./pages/EDA.html", 'r')
        page = report_file.read()
        components.html(page, width=1400, height=1000, scrolling=True)
    if st.form_submit_button("Standard EDA"):
        my_report = sv.analyze(st.session_state['input_df'])
        my_report.show_html(filepath="./pages/EDA.html", open_browser=False, layout="vertical", scale=1.0)

        report_file = codecs.open("./pages/EDA.html", 'r')
        page = report_file.read()
        components.html(page, width=1400, height=1000, scrolling=True)

with st.form("Create new project folder"):
    new_proj_dir = st.text_input("Name of new folder. Use ONLY lets and _, no special symbols or space.", help="Example: my_great_ai_project")
    if(bool(re.match('^[a-zA-Z0-9_]*$',new_proj_dir))==True):
        if len(new_proj_dir) > 1:
            st.success(f"You have chosen a great name! {new_proj_dir}, wow, that's a very fancy name for an AI project."
                       f"Now check the rest of the steps.")
        create_new_proj_dir = st.form_submit_button(f"Create {new_proj_dir} project folder")
        if create_new_proj_dir:
            os.mkdir(f'./input_data/{new_proj_dir}')
    else:
        st.error("Despite the fact that I am a super AI tool, I was created by a lazy and simple man that allowed the "
        f"name of the projects to contain only letters and '_' symbol, no spaces, no nothing. Boring, sad, but that's life..."
                   f"Please, correct the name and move on.")
        create_new_proj_dir_bad = st.form_submit_button(f"Click me harder")
        if create_new_proj_dir_bad:
            st.error("Please, just read the above error message once again and proceed accordingly.")



with st.form("Store data"):
    input_dir = './input_data/'
    dir_list = next(os.walk(input_dir))[1]
    selected_dir = st.selectbox("Select existing project folder", dir_list)
    name_file = st.text_input("How should I name the file?")

    if bool(re.match('^[a-zA-Z0-9_]*$', name_file)) == True:
        if len(new_proj_dir) > 5:
            st.success(f"Really?! {name_file}, that's all?!? Whatever, just click on the button below and lets move on...")
        save_orig_file = st.form_submit_button(f'Save the original file: {len(st.session_state["input_df_org"])} rows and name it {name_file}.parquet')
        if save_orig_file:
            st.session_state["input_df_org"].to_parquet(f"{input_dir}{selected_dir}/{name_file}.parquet")
    else:
        st.error("It is a shame, but I will repeat as for the project name - despite the fact that I am a super AI tool, I was created by a lazy and simple man that allowed the "
                 f"name of the projects to contain only letters and '_' symbol, no spaces, no nothing. Boring, sad, but that's life..."
                 f"Please, correct the name and move on.")
        create_new_proj_dir_bad = st.form_submit_button(f"Click me harder")
        if create_new_proj_dir_bad:
            st.error("Please, just read the above error message once again and proceed accordingly.")

