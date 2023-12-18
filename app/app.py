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
        st.image("image.png", use_column_width=True) #, width=300)


def display_mask(upload_files):
    with st.expander("**Preview identified sections**"):
        for uploaded_file in upload_files:
            uploaded_file = Path("app/data/" + uploaded_file)
            name = uploaded_file.name

            # load image file
            image_file = Image.open(f"app/data/{name[:-4]}_page_0001_Phase_2_Final_DLA.jpg")

            st.markdown(f"**{name}**")
            st.image(image_file, use_column_width=True) # width=300)  # , caption='Predicted masks for the first page'use_column_width=True)


def display_json(upload_files):
    with st.expander("**JSON Output**"):
        for uploaded_file in upload_files:
            uploaded_file = Path("app/data/" + uploaded_file)
            name = uploaded_file.name
            # load json file
            with open(f"app/data/processed_files/{name[:-4]}_Phase_2.json", "r") as f:
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
            with open(f"app/data/processed_files/{name[:-4]}.txt", "r", encoding="utf8") as f:
                txt_file = f.read()

            st.divider()
            st.markdown(f"**{name[:-4]}.txt**")
            st.code(txt_file, language='markdown', line_numbers=True)


def display_download_button():
    with open('app/data/processed_files.zip', 'rb') as f:
        download_btn = st.download_button(label=f'processed_files.zip', file_name=f'processed_files.zip', data=f)


# ------------Page configs-----------------------------------------------------------------------
st.set_page_config(layout='wide', page_title="📄 Clean Data is All You Need")


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
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap='large')
    with col1:
        # dropdown that allows to select one PDF file
        uploaded_files = st.selectbox("Select a PDF file", ["1603.01514.pdf", "1604.00727.pdf"])
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
        st.write("In this data centric world we felt the need to move forward in our education. We are a group coming from different backgrounds.")
        st.write("This project was a multifaceted endeavor, encompassing software development, visual transformers, computer vision, and classification algorithms. We collaborated seamlessly, taking on various roles to overcome the unique challenges it presented.")
        st.image("app/data/Team.png", caption="Team", width=800)

        


        st.title("Problem")
        st.write("Parsing unstructured data has traditionally been difficult, time consuming, manually intensive and costly!")
        st.write("While OCR, parsing, and packages for unstructured data have been improving, there hasn’t been a one click solution to understand a document layout and extract all the components with labels, ready to enrich and pass on to downstream tasks.")
        st.image("app/data/Problem_example.png", caption="Problem Example", width=800)

        st.title("Solution")

        # Solution Description
        st.markdown("# Our Solution: Takes Unstructured data in PDF Documents and returns, Semi-structured Data")

        # Introduction
        st.write("In the data-centric world we live in, the need for efficient data extraction from unstructured PDF documents has become increasingly important. Our solution is a simple yet powerful application designed to address this challenge. It takes unstructured data in PDF documents and transforms it into semi-structured data, making it easier for you to work with your valuable information.")

        # Key Features
        st.markdown("## Key Features")
        st.write("Our solution offers a range of features to enhance your data extraction experience:")

        # Feature 1: Advanced Vision Transformer Models
        st.markdown("### 1. Advanced Vision Transformer Models")
        st.write("We've integrated two cutting-edge vision transformer models to tackle document element identification and text extraction integrated with Optical Character Recognition (OCR). These models work together seamlessly to provide accurate and reliable results.")

        # Feature 2: User-Friendly Experience
        st.markdown("### 2. User-Friendly Experience")
        st.write("Our application is designed with user-friendliness in mind. You don't need to be an expert in data extraction to use it. It simplifies the process, making it accessible to professionals from various backgrounds.")

        # Feature 3: Accurate Data Reproduction
        st.markdown("### 3. Accurate Data Reproduction")
        st.write("One of our primary goals is to ensure the accurate reproduction of your data. You can rely on our solution to faithfully capture the content from PDF documents, reducing the risk of information loss or distortion.")

        # Feature 4: Improved Processing Speed
        st.markdown("### 4. Improved Processing Speed")
        st.write("Compared to traditional methods like OCR and regex parsing, our solution offers comparable data extraction processing time but enrich with metadata. You'll save valuable time while getting the results you need.")

        # Feature 5: Structured Output
        st.markdown("### 5. Structured Output")
        st.write("Our application provides structured output in either JSON or plain text format. This structured data can be seamlessly integrated into downstream tasks, such as Natural Language Processing (NLP) models, fine-tuning, or custom text extraction applications.")

        # Conclusion
        st.write("Our solution combines state-of-the-art vision transformer models with user-friendliness, accuracy, and speed, making it a versatile tool for professionals across various domains. Experience the difference in PDF data extraction with our application today!")

           
        
        # Display the list as a bulleted list
        
        col1, col2 = st.columns(2)
        with col1:
            st.image("app/data/TEXT.png", caption="Problem Example", width=500)
        with col2:
            st.image("app/data/JSON.png", caption="Problem Example", width=500)


        st.title("User")
        st.write("Our application caters to a diverse user base, as it is designed to assist in various downstream tasks. We define our users as professionals who require text extraction from unstructured documents to support a range of applications. These applications encompass the creation of NLP models, LLMs, RAG models, fine-tuning of existing models, or the implementation of simple text extraction solutions.")
        st.write("Our users come from a wide array of domains and professions, including data analysts, scientists, business professionals, academic researchers, legal experts, and many others who can benefit from the versatile downstream applications our application supports.")
        st.image("app/data/User_image.png", caption="Intended User", width=800)
        st.markdown("## User Profiles")
        st.write("Our application caters to a diverse group of professionals who require text extraction from PDF documents for various purposes. Here are some of the key user profiles:")

        # User Descriptions
        st.markdown("# User Profiles")

        # Define user profiles and descriptions
        user_profiles = {
        "Data Analysts and Scientists": "Data analysts and scientists often need to extract data from PDF documents for analysis. Our tool, which semi-structures this data, can significantly streamline their workflows.",
        "Business Professionals": "Business professionals in fields such as finance, marketing, and management frequently encounter data in PDF format, such as reports, financial statements, and market research. Our tool, which semi-structures this data, makes it easier to integrate information into their decision-making processes.",
        "Academic Researchers": "Academic researchers across various fields deal with large volumes of data in PDF format, including scholarly articles, data reports, and archival documents. Our tool, which helps structure this data, aids in more efficient data analysis and research.",
        "Government and Public Sector Officials": "Many government documents, including legal documents, policy papers, and statistical reports, are in PDF format. Our tool, which semi-structures this data, facilitates better data management and accessibility.",
        "Healthcare Professionals": "In healthcare, patient records, research papers, and clinical trial data are often in PDF format. Our tool, which can extract and semi-structure this data, is beneficial for analysis and record-keeping.",
        "Legal Professionals": "Lawyers and legal researchers frequently work with a large number of legal documents, many of which are in PDF format. Our tool, which semi-structures this data, aids in legal research and case preparation.",
        "Archivists and Librarians": "These professionals handle a large amount of historical and archival documents, often stored in PDF format. Our tool, which semi-structures this information, makes cataloging and retrieval more efficient.",
        "IT Professionals and Developers": "IT professionals and developers often need to integrate data from various sources, including PDFs, into software applications or databases. Our semi-structured data extraction tool is invaluable in this process.",
        "Educational Professionals": "Teachers and educators may use our tool to organize and analyze educational material and research papers, which are often in PDF format.",
        "Marketing Professionals": "Marketing professionals can use our tool for analyzing market research reports, customer feedback, and other marketing materials that are frequently in PDF format."
        }

        # Display user profiles and descriptions
        for profile, description in user_profiles.items():
            st.markdown(f"### {profile}")
            st.write(description)

    elif selected_tab == "**Method**":
        st.markdown("# Method")
        # Title
        st.title("Document Processing Workflow Summary")

        # Introduction
        st.write("In our document processing workflow, we follow a series of steps to efficiently handle different types of documents.")
                
        st.image("app/data/Pipeline.png", caption="Application Pipeline", width=1000)
        
        # Step 1: Import and Convert
        st.write("## Import and Convert")
        st.write("We start by importing documents and converting them into images.")

            # Branching Decision
        st.write("Next, we make decisions based on the document's content and text extractability.")

            # Upper Branch
        st.write("## Upper Branch:")
        st.write("If the document contains extractable text, we utilize the Document Layout Analysis (DLA) model. ")
        st.write("The DLA model helps recognize regions, capture figures and tables as images, and directly extract text.")
        st.write("This upper branch provides detailed tags for each region, facilitating filtering and augmentation for downstream tasks.")

            # Lower Branch
        st.write("## Lower Branch:")
        st.write("If the document lacks extractable text, we opt for the lower branch.")
        st.write("In this case, we employ the Nougat OCR model to process the data.")
        st.write("The lower branch can handle documents without extractable text and can natively convert formulas and tables to LaTeX.")
        st.write("However, it does not offer detailed section tagging.")

        # Conclusion
        st.write("Our document processing workflow is designed to efficiently handle a variety of documents, providing flexibility and accuracy in data extraction.")
        
        
        
        # Technical Architecture Overview
        st.markdown("## Technical Architecture Overview: Vision Transformer")

        st.write("In this section, we will delve into the technical architecture of our solution. Our system employs two Vision Transformer models (ViTs). These ViTs, similar to traditional Transformers such as BERT and GPT, are designed to process sequential data using self-attention mechanisms, enabling them to grasp context and predict sequential output.")

        st.write("However, what sets ViTs apart is their ability to work with images as sequences of pixels or patches. This unique approach allows them to comprehend visual data spatially and recognize intricate relationships within images.")

        st.write("We will explore how our system utilizes these ViTs for Optical Character Recognition (OCR) tasks, effectively extracting text and other elements from various types of documents. Additionally, we will discuss how these models can be applied to gain a comprehensive understanding of a document's layout and structure.")

       
        st.image("app/data/Vision_transformer.png", caption="Vision Transformer Information", width=700)
    
                 
       
        
    
        # Document Layout Analysis
        st.markdown("## Document Layout Analysis - Process Step by Step")
                
        # Introduction
        st.write("In this section, we will delve into the Document Layout Analysis (DLA) component of our data extraction process. DLA plays a crucial role in understanding the structure and organization of documents, enabling us to extract valuable information effectively.")
        st.image("app/data/DLA1.png", caption="Application Pipeline", width=1000)
        st.image("app/data/DLA2.png", caption="Application Pipeline", width=1000)
        # Page-by-Page Import
        st.markdown("## Page-by-Page Import:")
        st.write("- Documents are imported into our system on a page-by-page basis.")
        st.write("- Each page is treated as an individual unit for analysis.")

        # Vision Transformer Utilization
        st.markdown("## Vision Transformer Utilization:")
        st.write("- We employ a powerful vision transformer model to recognize entities within each page.")
        st.write("- Unlike traditional OCR techniques, which process text sequentially, our vision transformer can process images as sequences of pixels or patches.")
        st.write("- This enables us to understand visual data spatially and capture complex relationships within images.")

        # Output: Bounding Boxes and Classification
        st.markdown("## Output: Bounding Boxes and Classification:")
        st.write("- After analysis, the DLA model generates a set of bounding boxes.")
        st.write("- Each bounding box is associated with a classification and a probability score.")
        st.write("- These bounding boxes represent regions of interest within the document.")

        # Addressing Overlaps
        st.markdown("## Addressing Overlaps:")
        st.write("- It's common to encounter multiple bounding boxes that overlap in the same region.")
        st.write("- To resolve this, we apply heuristic logic in the third step.")
        st.write("- The logic identifies regions with the highest probability and eliminates overlap from other boxes.")
        st.write("- This ensures that each region is uniquely claimed by only one bounding box.")

        # SVM Classifier for Unclassified Regions
        st.markdown("## SVM Classifier for Unclassified Regions:")
        st.write("- In some cases, certain regions may remain unclassified during the initial inference step.")
        st.write("- These unclassified regions often correspond to elements for which the model lacks training data.")
        st.write("- To address this, we employ a Support Vector Machine (SVM) classifier.")

        #Indexing for Structure
        st.markdown("## Indexing for Structure:")
        st.write("- Finally, all the bounding boxes are consolidated.")
        st.write("- They are organized based on their vertical order and column location, effectively structuring the document's content.")
        st.write("This document layout analysis is a critical step in our data extraction process. It empowers us to understand the document's structure and content, enabling accurate and meaningful data extraction for various downstream applications.")
        


        st.write("## An example with more lovel of detail")    
        col1, col2 = st.columns(2)
        with col1:
            st.image("app/data/PDF_example.png", caption="Application Pipeline", width=700)
        
        with col2:
            st.image("app/data/Pdf_mask.jpg", caption="Application Pipeline", width=700)
        
        
        # Neural Optical Understanding for Academic Documents (Nougat)
        st.markdown("## Neural Optical Understanding for Academic Documents (Nougat)")

        st.write("Nougat is an advanced document understanding tool that leverages state-of-the-art technology. It is powered by OCR-free Document Understanding Transformer (Donut), which demonstrated the ViT's capability to understand complex documents like receipts and business cards without the need for OCR (Optical Character Recognition).")

        st.write("The Nougat team at Meta extended this concept to scientific papers by training the model on LaTex versions of the documents as opposed to images. This innovative approach resulted in remarkably high extraction accuracy for plain text, as well as other challenging elements like tables and formulas, which are typically difficult for traditional OCR methods to handle.")

        st.write("Thanks to its Transformer-based architecture, Nougat offers highly parallelizable and cost-effective processing while maintaining exceptional accuracy. However, for our project, the output format wasn't the semi-structured JSON we aimed for. Achieving this required a prior understanding of the document's layout, followed by fine-tuning Nougat to interpret the data accordingly. We are actively working on implementing these enhancements for our final presentation. Now, for more insights into document layout, let's turn to Carlos.")

        st.image("app/data/NOUGAT.png", caption="Application Pipeline", width=700)
        

        #Nougat Fine tunning
        st.markdown("## Nougat Fine tunning")
        st.write("Images have been processed using MMDs and augmented with structural data obtained from DLA (Diffusion-Limited Aggregation). The fine-tuning process took place on 4 A100 GPUs for a total of 14 hours. The output data is now structured and ready to be parsed in JSON format.")
        col1, col2 = st.columns(2)
        with col1:
            st.image("app/data/nougat_finetunning.png", caption="Nougat Fine tuned Results", width=500)
        
        with col2:
            st.image("app/data/nougat_finetunning2.png", caption="Nougat Fine tuned Results", width=500)

        # Text and Json Extraction
        st.markdown("## Text and Json Extraction")
        st.write("Once we get the DLA outputs, we obtained and indexed dataset of masks, classification and positions. These are use to create text files and json files. We utilize PyMUPDF to extract the information and structured as the example below")
        col1, col2 = st.columns(2)
        with col1:
            st.image("app/data/TEXT.png", caption="Text", width=500)
        with col2:
            st.image("app/data/JSON.png", caption="Json", width=500)
       
        # Equation Extraction
        st.markdown("## Formula Extraction")
        st.write("We have fine-tuned the DLA model to recognize elements, particularly mathematical formulas, which is essential for determining the branch our system follows. As illustrated in the image below, there are multiple other categories our system is working on, and more can be added if required, as this solution is flexible.")
        st.image("app/data/DLAwithformula.png", caption="DLA mask visual results", width=700)

    elif selected_tab == "**Evaluation**":
        st.markdown("# Evaluation")
        st.write("This section covers we will discuss the evaluation of our approach and share some results.")
        
        st.markdown("## Metrics to evaluate parsers:")
        st.markdown("###  Number of tokens:")
        st.markdown(" This metric quantifies the amount of text retained or discarded during the extraction process.")
        st.markdown(" In terms of number of tokens, currently our method is the most similar to the Ground Truth Documentation. It has more tokens than Ground Truth which is ideal as shows that we are not loosing important information")
        st.image("app/data/Lenght_Tokens_Result.png", caption="Amount of tokens after Text Extracion of different Methods", width=800)
        st.markdown("###  Document cosine similarity:")
        st.markdown(" It measures the extent to which the extracted text preserves the document's overall meaning. This helps us ensure that we're not discarding crucial information.")
        st.markdown(" This is telling us that our method comes second, and that there are some differences between the text we extract and our ground truth, but just from this we can’t tell what those differences are.")
        st.image("app/data/cosine_similary_Result.png", caption="Results of Cosine Similarity Againts Ground Truth", width=800)
        st.markdown("###  Processing time: ")
        st.markdown("Efficiency matters, and our users value both time and cost savings. We analyze how quickly our method can process large volumes of documents")
        col1, col2 = st.columns(2)
        with col1:
            data = [
            {'Dataset': 'Grobid', 'Processing Time per page': 'Less than 1 second'},
            {'Dataset': 'Nougat', 'Processing Time per page': '4.93 seconds'},
            {'Dataset': 'Pypdf', 'Processing Time per page': 'Less than 1 second'},
            {'Dataset': 'Our method (DLA+Text extraction)', 'Processing Time per page': '4.47 seconds'}
            ]
            st.table(data)
        with col2:
            st.markdown("Our method exhibits faster performance compared to Nougat but lags behind the rest in terms of speed. It's worth highlighting that both Nougat and our method utilize Vision Transformers, providing versatility for various document types and formats. In contrast, Grobid has been exclusively trained on scientific papers and lacks an easy retraining mechanism. While Pypdf is swift, it falls short in offering the metadata enrichment capabilities we aim to provide.")
        st.markdown("## Metrics to evaluate downstream tasks:")
        st.markdown("As our users often employ the extracted data for downstream tasks, such as Question/Answering on a corpus of documents, we've included a specific metric to evaluate the performance of these downstream applications.")
        st.markdown("We can then compare each of the models against this Question / Answer benchmark on scientific research papers, called Qasper.")
        st.markdown("## Question Answering on Scientific Research Papers (Qasper)")
        col1, col2, col3 = st.columns(3)
        # Create a Streamlit DataFrame from the data
        
        with col2:
            
            st.image("app/data/Qasper.png", caption="Qasper", width=400)
          
        with col1:
        # Add the caption below the centered image
            
            st.markdown("Total # papers: 1,585")
            st.markdown("Total # questions: 5,692")
            st.markdown("Avg # words:")
            st.markdown("- Input: 3,629")
            st.markdown("- Output: 11.4")
        
        # Define the data
        st.markdown("## Results")
        col1, col2 = st.columns(2)
        with col1:
            data = [
            {'Dataset': 'Grobid', 'F1 (⇧)': 22.9},
            {'Dataset': 'Nougat', 'F1 (⇧)': 23.5},
            {'Dataset': 'Pypdf', 'F1 (⇧)': 22.1},
            {'Dataset': 'Our method (DLA+Text extraction)', 'F1 (⇧)': 23.4}
            ]
            st.table(data)
        with col2:
            st.markdown(" Our results are among the top performers, with an extremely narrow margin. It's important to note that Nougat was specifically trained on research paper formats. What this shows us is that our model is largely comparable to the state of the art (nougat). The text that we extract is sufficient for the model to answer questions about the paper, which means that we are not losing important details in our pipeline, which is very important.") 
