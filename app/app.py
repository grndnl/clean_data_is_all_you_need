import streamlit as st
import fitz
from pathlib import Path
import time
import shutil
import base64
from PIL import Image
import json
# import asyncio
# import threading
# from http.server import HTTPServer, SimpleHTTPRequestHandler



def disable():
    st.session_state.disabled = True


def enable():
    st.session_state.disabled = False
    st.session_state.processed_files = False


def displayPDF(uploaded_files):
    pdf_displays = []
    document = fitz.open("app/data/" + uploaded_files[0])
    page = document.load_page(0)  # number of page
    pix = page.get_pixmap()
    pix.save("image.png")

    # Display file
    with st.expander("**Uploaded PDFs**"):
        st.markdown(f"**{uploaded_files[0]}**")
        st.image("image.png", width=300)


def display_mask(upload_files):
    with st.expander("**Preview identified sections**"):
        for uploaded_file in upload_files:
            uploaded_file = Path("app/data/" + uploaded_file)
            name = uploaded_file.name

            # load image file
            image_file = Image.open("app/data/1603.09631_page_0001_dev_img_1_base_dla_result.jpg") # TODO This is actually the wrong image

            st.markdown(f"**{name}**")
            st.image(image_file, width=300)  # , caption='Predicted masks for the first page'use_column_width=True)


def display_json(upload_files):
    with st.expander("**JSON Output**"):
        for uploaded_file in upload_files:
            uploaded_file = Path("app/data/" + uploaded_file)
            name = uploaded_file.name
            # load json file
            with open("app/data/processed_files/1603.09631.json", "r") as f:
                json_file = json.load(f)

            st.divider()
            st.markdown(f"**{name[:-4]}.json**")
            st.json(json_file, expanded=False)


def display_markdown(upload_files):
    with st.expander("**Txt Output**"):
        for uploaded_file in upload_files:
            uploaded_file = Path("app/data/" + uploaded_file)
            name = uploaded_file.name
            # load md file
            with open("app/data/processed_files/1603.09631.txt", "r", encoding="utf8") as f:
                txt_file = f.read()

            st.divider()
            st.markdown(f"**{name[:-4]}.txt**")
            st.code(txt_file, language='markdown', line_numbers=True)


def display_download_button():
    with open('app/data/processed_files.zip', 'rb') as f:
        download_btn = st.download_button(label=f'processed_files.zip', file_name=f'processed_files.zip', data=f)


# ------------Page configs-----------------------------------------------------------------------
st.set_page_config(layout='wide')


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

# Tabs
tab1, tab2 = st.tabs(["Demo", "Documentation"])

with tab1:
    # Columns
    # 4 columns of different sizes
    col1, col2, col3, col4 = st.columns([1, 0.3, 1, 1], gap='large')
    with col1:
        # dropdown that allows to select one PDF file
        uploaded_files = st.selectbox("Select a PDF file", ["1603.09631.pdf"])
        uploaded_files = [uploaded_files]
        enable()

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
                # processed_files = process_pdfs(uploaded_files)
                time.sleep(1)
            st.success('Processing complete!')
            st.session_state.processed_files = True

            time.sleep(1)
            display_mask(uploaded_files)

            time.sleep(0.5)

        with col4:
            # zip processed files
            shutil.make_archive('app/data/processed_files', 'zip', 'app/data/processed_files')

            st.write(" ")
            st.write(" ")
            display_download_button()
            time.sleep(0.5)
            display_json(uploaded_files)
            time.sleep(0.5)
            display_markdown(uploaded_files)

# ----------Documentation-----------------------------------------------------------------------------
with tab2:
    selected_tab = st.radio("", ["**Overview**", "**Method**", "**Evaluation**"], horizontal=True)

    
    # Main content area
    if selected_tab == "**Overview**":
        st.markdown("# The Team")
        st.image("app/data/Team.png", caption="Team", width=800)
        st.write("This is an overview of the app's purpose and functionality.")
        st.write("You can add more details and content here.")
        st.title("User")
        st.write("Professional needing text extraction on diversity of PDFs for different downstream applications.")
        st.image("app/data/User_image.png", caption="Intended User", width=800)
        st.title("Problem")
        st.write("Parsing unstructured data has traditionally been difficult, time consuming, manually intensive and costly!")
        st.write("While OCR, parsing, and packages for unstructured data have been improving, there hasn’t been a one click solution to understand a document layout and extract all the components with labels, ready to enrich and pass on to downstream tasks.")
        st.image("app/data/Problem_example.png", caption="Problem Example", width=800)

        st.title("Solution")
        items = [
        "Combines two state-of-the-art vision transformer models to segment documents and OCR text and other elements.",
        "Easy to use.",
        "Accurate reproduction of data.",
        "Faster than parsing via OCR/regex.",
        "The structure can be used in downstream tasks."
    ]
    
        # Display the list as a bulleted list
        st.markdown("<ul>" + "".join([f"<li>{item}</li>" for item in items]) + "</ul>", unsafe_allow_html=True)
    
    elif selected_tab == "**Method**":
        st.markdown("# Method")
        st.write("This section provides details about the methods used in the app.")
        st.image("app/data/Pipeline.png", caption="Application Pipeline", width=1000)
        
        #Vision Transformer
        st.markdown("## Vision Transformer")
        st.write("Describe Vision Transfor method here.")
       
        st.image("app/data/Vision_transformer.png", caption="Vision Transformer Information", width=1000)
    
        st.markdown("## Document Layout Analysis - Example")
        st.write("Describe Vision Transfor method here.")    
        col1, col2 = st.columns(2)
        with col1:
            st.image("app/data/PDF_example.png", caption="Application Pipeline", width=700)
        
        with col2:
            st.image("app/data/Pdf_mask.jpg", caption="Application Pipeline", width=700)
        
        
        
    
        # Document Layout Analysis
        st.markdown("## Document Layout Analysis - Process Step by Step")
        st.write("Describe the document layout analysis method here.")
        st.image("app/data/DLA1.png", caption="Application Pipeline", width=1000)
        st.image("app/data/DLA2.png", caption="Application Pipeline", width=1000)
        
        # Nougat
        st.markdown("## Nougat - Visual Transformer")
        st.write("Describe Nougat method here.")
        st.image("app/data/NOUGAT.png", caption="Application Pipeline", width=700)
    
        # Method Selector
        st.markdown("## Method Selector")
        st.write("Explain the method selector feature and its functionality.")
        
        # Text Extraction
        st.markdown("## Text Extraction")
        st.write("Discuss the text extraction method and its capabilities.")
        
        # Equation Extraction
        st.markdown("## Equation Extraction")
        st.write("Explain how the equation extraction feature works.")
        
    elif selected_tab == "**Evaluation**":
        st.markdown("# Evaluation")
        st.write("This section covers the evaluation of the app's performance.")
        col1, col2 = st.columns(2)
        with col1:
            st.image("app/data/TEXT.png", caption="Text", width=500)
        with col2:
            st.image("app/data/JSON.png", caption="Json", width=500)
        st.markdown("## Metrics to evaluate parsers:")
        st.markdown("- Number of tokens")
        st.markdown("- Document cosine similarity")
        st.markdown("- Processing time (pending)")
    
        st.markdown("## Metrics to evaluate downstream tasks:")
        st.markdown("- Question/Answer - F1 score")
    
        # By the Numbers
        st.markdown("## By the Numbers")
        st.write("Provide numerical metrics and statistics related to the app's performance.")
        st.image("app/data/Lenght_Tokens_Result.png", caption="Amount of tokens after Text Extracion of different Methods", width=800)
        st.image("app/data/cosine_similary_Result.png", caption="Results of Cosine Similarity Againts Ground Truth", width=800)
   





    # Center the table using CSS styles
    st.markdown(
        """
        <style>
        div.stTable {
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
)
    

    st.markdown("# Evaluation with a Downstream Task")
    st.markdown("## Question Answering on Scientific Research Papers (Qasper)")
    col1, col2, col3 = st.columns(3)
    # Create a Streamlit DataFrame from the data
    
    with col1:
        
        st.image("app/data/Qasper.png", caption="Qasper", width=400)
      
    with col2:
    # Add the caption below the centered image
        
        st.markdown("Total # papers: 1,585")
        st.markdown("Total # questions: 5,692")
        st.markdown("Avg # words:")
        st.markdown("- Input: 3,629")
        st.markdown("- Output: 11.4")
    
    # Downstream Task
    st.markdown("## Downstream Task")
    st.write("Explain how the app's results can be used in a downstream task.")
    # Define the data
    st.markdown("## Results")
    with col1:
        data = [
        {'Dataset': 'Grobid', 'F1 (⇧)': 22.9},
        {'Dataset': 'Nougat', 'F1 (⇧)': 23.5},
        {'Dataset': 'Pypdf', 'F1 (⇧)': 22.1},
        {'Dataset': 'Our method (DLA+Text extraction)', 'F1 (⇧)': 23.4}
        ]
        st.table(data)
