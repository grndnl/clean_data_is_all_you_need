import streamlit as st
from PyPDF2 import PdfReader
from pathlib import Path
import time
import shutil
import os


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


# Page configs
st.set_page_config(layout='wide')

# Clean local files  # TODO
# clean tmp folder and delete zip file if it exists
# shutil.rmtree('tmp', ignore_errors=True)
# os.remove('processed_files.zip') if os.path.exists('processed_files.zip') else None

# Session State
# st.write(st.session_state)
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = False
if "disabled" not in st.session_state:
    st.session_state.disabled = False

# Title of the page
st.title('📄 Clean Data is All You Need')

# Sidebar for navigation and control
# with st.sidebar:
#     st.header('Controls')
#     uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type='pdf')
#     process_button = st.button('Process PDFs')  # , disabled=True)

# Columns
col1, col2, col3 = st.columns(3, gap='large')
with col1:
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type='pdf', on_change=enable)
with col2:
    st.write(" ")
    st.write(" ")
    st.write(" ")
    process_button = st.button('Process PDFs', type='primary', on_click=disable, disabled=st.session_state.disabled)

# Main area
# if uploaded_files:
#     # enable process button
#     if process_button.disabled:
#         process_button.disabled = False

if st.session_state.processed_files:
    with col3:
        st.success('Processing complete!')
        with open('processed_files.zip', 'rb') as f:
            st.download_button(label=f'processed_files.zip', file_name=f'processed_files.zip', data=f)

if uploaded_files and process_button:
    with col3:
        with st.spinner('Processing PDFs...'):
            processed_files = process_pdfs(uploaded_files)
        st.success('Processing complete!')
        st.session_state.processed_files = True
        # st.write('**Processed Files:**')

        # zip processed files
        shutil.make_archive('processed_files', 'zip', 'tmp')

        # for file in processed_files:
        #     # load file as binary
        #     data = file.read_bytes()
        #     file_name = file.stem[:-4]
        #     st.download_button(label=f'{file_name}.txt', file_name=f'{file_name}.txt', data=data)
        with open('processed_files.zip', 'rb') as f:
            download_btn = st.download_button(label=f'processed_files.zip', file_name=f'processed_files.zip', data=f)
        # if download_btn:
        #     st.write("**Files downloaded**")
