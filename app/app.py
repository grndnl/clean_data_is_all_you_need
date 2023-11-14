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
    st.session_state['processed_files'] = False


def displayPDF(uploaded_file):
    uploaded_file = uploaded_file[0]

    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')

    # Embed PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" ' \
                  F'type="application/pdf"></iframe>'

    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


def display_mask():
    st.write("#### Predicted masks for the first page:")
    # load image file
    image_file = Image.open("app/data/1603.09631_page_0001_dev_img_1_base_dla_result.jpg")
    st.image(image_file, caption='Predicted masks for the first page:', use_column_width=True)


def display_json():
    st.write("#### JSON file for the first page:")

    # load json file
    with open("app/data/json from grobid.json", "r") as f:
        json_file = json.load(f)

    st.json(json_file)

# -----------------------------------------------------------------------------------
# Page configs
st.set_page_config(layout='wide')

# Clean local files  # TODO
# clean tmp folder and delete zip file if it exists
# shutil.rmtree('tmp', ignore_errors=True)
# os.remove('processed_files.zip') if os.path.exists('processed_files.zip') else None

# -----------------------------------------------------------------------------------
# Session State
# st.write(st.session_state)
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = False
if "disabled" not in st.session_state:
    st.session_state.disabled = False


# -----------------------------------------------------------------------------------
# UI logic
# Title of the page
st.title('📄 Clean Data is All You Need')

# Sidebar for navigation and control
# with st.sidebar:
#     st.header('Controls')
#     uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type='pdf')
#     process_button = st.button('Process PDFs')  # , disabled=True)

# Columns
# 4 columns of different sizes
col1, col2, col3, col4 = st.columns([1, 0.3, 1, 1], gap='large')
with col1:
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type='pdf', on_change=enable)
with col2:
    st.write(" ")
    st.write(" ")
    st.write(" ")
    process_button = st.button('Process PDFs', type='primary', on_click=disable, disabled=st.session_state.disabled)

if st.session_state.processed_files:
    with col3:
        st.write(" ")
        st.write(" ")
        # st.write(" ")
        st.success('Processing complete!')
        with open('processed_files.zip', 'rb') as f:
            st.download_button(label=f'processed_files.zip', file_name=f'processed_files.zip', data=f)

if uploaded_files:
    with col1:
        displayPDF(uploaded_files)

if uploaded_files and process_button:
    with col3:
        with st.spinner('Processing PDFs...'):
            processed_files = process_pdfs(uploaded_files)
        st.success('Processing complete!')
        st.session_state.processed_files = True

        display_mask()

    with col4:
        # zip processed files
        shutil.make_archive('tmp/processed_files', 'zip', 'tmp')

        with open('tmp/processed_files.zip', 'rb') as f:
            download_btn = st.download_button(label=f'processed_files.zip', file_name=f'processed_files.zip', data=f)

        display_json()
