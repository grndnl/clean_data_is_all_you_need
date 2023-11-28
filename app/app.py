import streamlit as st
from PyPDF2 import PdfReader
from pathlib import Path
import time
import shutil
import os
import base64
from PIL import Image
import json


def process_pdfs(files):
    processed_files = []

    # create tmp folder if it doesn't exist
    tmp_folder = Path('tmp')
    tmp_folder.mkdir(exist_ok=True)

    for file in files:
        # Read the PDF file from the uploaded file
        pdf_reader = PdfReader(file)

        # Extract text from the first page
        first_page = pdf_reader.pages[0]
        text = first_page.extract_text()

        # Write the text to a text file
        out_dir = tmp_folder / (file.name[:-4] + '.txt')
        text_file = out_dir
        text_file.write_text(text, encoding='utf-8')

        processed_files.append(out_dir)

        # wait 1 second
        time.sleep(1)
    return processed_files


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
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="40%" height="350" ' \
                      F'type="application/pdf"></iframe>'
        # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="40%" height="350" type="application/pdf">'
        pdf_displays.append(pdf_display)

    # Display file
    with st.expander("**Uploaded PDFs**"):
        st.markdown(" ".join(pdf_displays), unsafe_allow_html=True)


def display_mask(upload_files):
    with st.expander("**Preview identified sections**"):
        for uploaded_file in upload_files:
            name = uploaded_file.name

            # load image file
            image_file = Image.open("app/data/1603.09631_page_0001_dev_img_1_base_dla_result.jpg")

            st.markdown(f"**{name}**")
            st.image(image_file, width=200)  # , caption='Predicted masks for the first page'use_column_width=True)


def display_json(upload_files):
    with st.expander("**JSON Output**"):
        for uploaded_file in upload_files:
            name = uploaded_file.name
            # load json file
            with open("app/data/json from grobid.json", "r") as f:
                json_file = json.load(f)

            st.divider()
            st.markdown(f"**{name[:-4]}.json**")
            st.json(json_file, expanded=False)


def display_markdown(upload_files):
    with st.expander("**Txt Output**"):
        for uploaded_file in upload_files:
            name = uploaded_file.name
            # load md file
            with open("app/data/markdown from html.mmd", "r", encoding="utf8") as f:
                txt_file = f.read()

            st.divider()
            st.markdown(f"**{name[:-4]}.txt**")
            st.code(txt_file, language='markdown', line_numbers=True)


def display_download_button():
    with open('tmp/processed_files.zip', 'rb') as f:
        download_btn = st.download_button(label=f'processed_files.zip', file_name=f'processed_files.zip', data=f)


# ------------Page configs-----------------------------------------------------------------------
st.set_page_config(layout='wide')

# Clean local files  # TODO
# clean tmp folder and delete zip file if it exists
# shutil.rmtree('tmp', ignore_errors=True)
# os.remove('processed_files.zip') if os.path.exists('processed_files.zip') else None


# ------------Session State-----------------------------------------------------------------------
# st.write(st.session_state)
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = False
if "disabled" not in st.session_state:
    st.session_state.disabled = False

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

# Tabs
tab1, tab2 = st.tabs(["Demo", "Documentation"])

with tab1:
    # Columns
    # 4 columns of different sizes
    col1, col2, col3, col4 = st.columns([1, 0.3, 1, 1], gap='large')
    with col1:
        uploaded_files = st.file_uploader("**Upload PDFs**", accept_multiple_files=True, type='pdf', on_change=enable)
    with col2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        process_button = st.button('**Process PDFs**', type='primary', on_click=disable, disabled=st.session_state.disabled)

    if st.session_state.processed_files:
        with col3:
            st.write(" ")
            st.write(" ")
            # st.write(" ")
            st.success('Processing complete!')

    if uploaded_files:
        with col1:
            displayPDF(uploaded_files)

    if st.session_state.processed_files:  # Saved state necessary for clean debugging when deployed locally
        with col3:
            display_mask(uploaded_files)
        with col4:
            st.write(" ")
            st.write(" ")
            display_download_button()
            display_json(uploaded_files)
            display_markdown(uploaded_files)

    if uploaded_files and process_button:
        with col3:
            with st.spinner('Processing PDFs...'):
                processed_files = process_pdfs(uploaded_files)
            st.success('Processing complete!')
            st.session_state.processed_files = True

            time.sleep(1)
            display_mask(uploaded_files)

            time.sleep(0.5)

        with col4:
            # zip processed files
            shutil.make_archive('tmp/processed_files', 'zip', 'tmp')

            st.write(" ")
            st.write(" ")
            display_download_button()
            time.sleep(0.5)
            display_json(uploaded_files)
            time.sleep(0.5)
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
