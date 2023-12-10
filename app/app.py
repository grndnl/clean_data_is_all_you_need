import streamlit as st
from pathlib import Path
import time
import shutil
import os
import base64
from PIL import Image
import json
import requests


def disable():
    st.session_state.disabled = True


def enable():
    st.session_state.disabled = False
    st.session_state.processed_files = False


def displayPDF(uploaded_files):
    pdf_displays = []
    for uploaded_file in uploaded_files:
        # Read file as bytes:
        bytes_data = uploaded_file.getvalue()

        # Convert to utf-8
        base64_pdf = base64.b64encode(bytes_data).decode('utf-8')

        # Embed PDF in HTML
        # pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="40%" height="350" ' \
        #               F'type="application/pdf"></iframe>'
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="40%" height="350" type="application/pdf">'
        pdf_displays.append(pdf_display)

    # Display file
    with st.expander("**Uploaded PDFs**"):
        st.markdown(" ".join(pdf_displays), unsafe_allow_html=True)


def display_mask(upload_files):
    with st.expander("**Preview identified sections**"):
        for uploaded_file in upload_files:
            name = uploaded_file.name

            # load image file
            image_file = Image.open(f"app/data/s3_outputs/model_outputs/{name[:-4]}_page_0001_dev_img_1_base_dla_result.jpg")

            st.markdown(f"**{name}**")
            st.image(image_file, width=200)  # , caption='Predicted masks for the first page'use_column_width=True)


def display_json(upload_files):
    with st.expander("**JSON Output**"):
        for uploaded_file in upload_files:
            name = uploaded_file.name
            # load json file
            with open(f"app/data/s4_json_text_output/{name[:-4]}/{name[:-4]}.json", "r") as f:
                json_file = json.load(f)

            st.divider()
            st.markdown(f"**{name[:-4]}.json**")
            st.json(json_file, expanded=False)


def display_markdown(upload_files):
    with st.expander("**Txt Output**"):
        for uploaded_file in upload_files:
            name = uploaded_file.name
            # load md file
            with open(f"app/data/s4_json_text_output/{name[:-4]}/{name[:-4]}.txt", "r", encoding="utf8") as f:
                txt_file = f.read()

            st.divider()
            st.markdown(f"**{name[:-4]}.txt**")
            st.code(txt_file, language='markdown', line_numbers=True)


def display_download_button():
    with open('app/data/s4_json_text_output.zip', 'rb') as f:
        download_btn = st.download_button(label=f'processed_files.zip', file_name=f'processed_files.zip', data=f)


def call_process_pdfs_api():
    # API endpoint
    api_url = "http://localhost:8100/dla?use_cpu=false&model_type=DIT"

    # Make the POST request to the FastAPI backend, and error out with timeout
    try:
        response = requests.get(api_url, timeout=120)
    except Exception as e:
        st.error(e)
        return None, None, None

    # Process the response
    if response.status_code == 200:
        # From response body get exit_msg
        exit_msg = response.json()['exit_msg']
        output_dir = response.json()['output_directory']
        json_txt_dir = response.json()['text_output_directory']
        return exit_msg, output_dir, json_txt_dir
    elif response.status_code == 500:
        st.error(response)
    else:
        # Handle errors or unsuccessful responses
        return None, None, None


def save_files(file_bytes):
    # create folder if it does not exist
    os.makedirs('app/data/s1_input_pdfs', exist_ok=True)

    for file_byte in file_bytes:
        with open(f"app/data/s1_input_pdfs/{file_byte.name}", "wb") as f:
            f.write(file_byte.getbuffer())


# ------------Page configs-----------------------------------------------------------------------
st.set_page_config(layout='wide')


# ------------Session State-----------------------------------------------------------------------
# st.write(st.session_state)
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = False
if "disabled" not in st.session_state:
    st.session_state.disabled = False
if "output_data" not in st.session_state:
    st.session_state.output_data = None, None, None

# ----------UI logic-------------------------------------------------------------------------
# Title of the page
st.title('📄 Clean Data is All You Need')

# Sidebar for navigation and control
with st.sidebar:
    st.header('About')
    # add information about what this page does
    st.markdown('Process PDFs of scientific papers into structured data.')
    st.markdown('[GitHub](https://github.com/grndnl/clean_data_is_all_you_need)')
    st.markdown('*Please do not upload sensitive information.*')
    # st.header(":red[This is a work in progress, and will not work. For a staged demo, please visit [this link](https://cleandataisallyouneed.streamlit.app/).]")

# Tabs
tab1, tab2 = st.tabs(["Demo", "Documentation"])

with tab1:
    # Columns
    # 4 columns of different sizes
    col1, col2, col3, col4 = st.columns([1, 0.3, 1, 1], gap='large')
    with col1:
        uploaded_files = st.file_uploader("**Upload PDFs**", accept_multiple_files=True, type='pdf', on_change=enable)
        save_files(uploaded_files)
    with col2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        process_button = st.button('**Process PDFs**', type='primary', on_click=disable,
                                   disabled=st.session_state.disabled)

    if st.session_state.processed_files:
        with col3:
            st.write(" ")
            st.write(" ")
            # st.write(" ")
            st.success(st.session_state.output_data[0])

    if uploaded_files:
        with col1:
            displayPDF(uploaded_files)

    # if st.session_state.processed_files:  # Saved state necessary for clean debugging when deployed locally
    #     with col3:
    #         display_mask(uploaded_files)
    #     with col4:
    #         st.write(" ")
    #         st.write(" ")
    #         display_download_button()
    #         display_json(uploaded_files, json_path)
    #         display_markdown(uploaded_files, text_path)

    if uploaded_files and process_button:
        with col3:
            with st.spinner('Processing PDFs...'):
                exit_msg, output_dir, json_txt_dir = call_process_pdfs_api()
            if None in [exit_msg, output_dir, json_txt_dir]:
                st.error("Something went wrong. Please try again.")
                st.stop()
            else:
                st.success(exit_msg)
                st.session_state.processed_files = True
                st.session_state.output_data = exit_msg, output_dir, json_txt_dir

            time.sleep(1)
            display_mask(uploaded_files)
            time.sleep(1)

        with col4:
            # zip processed files
            with st.spinner("Zipping processed files..."):
                shutil.make_archive('app/data/s4_json_text_output', 'zip', 'app/data/s4_json_text_output')

            st.write(st.session_state.processed_files)
            st.write(" ")
            st.write(" ")
            display_download_button()
            time.sleep(1)
            display_json(uploaded_files)
            time.sleep(1)
            display_markdown(uploaded_files)

    elif st.session_state.processed_files:  # Saved state necessary to resume once the download button is pressed
        with col3:
            display_mask(uploaded_files)
        with col4:
            st.write(" ")
            st.write(" ")
            display_download_button()
            display_json(uploaded_files)
            display_markdown(uploaded_files)

# ----------Documentation-----------------------------------------------------------------------------
with tab2:
    st.markdown("# Overview")
    st.markdown("# Method")
    st.markdown("## Document Layout Analysis")
    st.markdown("## Method Selector")
    st.markdown("## Text Extraction")
    st.markdown("## Equation Extraction")
    st.markdown("# Evaluation")
    st.markdown("## By the Numbers")
    st.markdown("## Downstream Task")
