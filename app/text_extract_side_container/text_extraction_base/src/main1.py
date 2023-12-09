import glob
import pandas as pd
import os
import fitz  # PyMuPDF
import textwrap
import json
from typing import List
from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()

def extract_text_from_scaled_pdf(pdf_path, page_number, coords, new_dimensions):
    """Extract text from specified coordinates in a scaled PDF."""
    doc = fitz.open(pdf_path)
    scaled_doc = fitz.open()  # Create a new empty PDF for scaled pages

    # Scale the specific page
    page = doc.load_page(page_number)
    new_page = scaled_doc.new_page(width=int(new_dimensions[1]), height=int(new_dimensions[0]))
    new_page.show_pdf_page(new_page.rect, doc, page.number)

    # Extract text from the scaled page
    scaled_page = scaled_doc.load_page(0)  # As we have only one page in scaled_doc
    extracted_text = scaled_page.get_text("text", clip=fitz.Rect(coords))
    #extracted_text = ' '.join(extracted_text.split())



    # Clean up
    doc.close()
    scaled_doc.close()
    return extracted_text

def extract_page_number(filename):
    # Split the filename using "_page_" as the separator
    parts = os.path.basename(filename).split("_page_")

    if len(parts) >= 2:
        try:
            # Extract the next part as the page number and convert it to an integer
            page_number = int(parts[1].split("_")[0])
            return page_number
        except ValueError:
            # Handle the case where the extracted value is not a valid integer
            return None
    else:
        # Handle the case where "_page_" is not found in the filename
        return None

async def process_pdfs(files: List[UploadFile]):
    for upload_file in files:
        # Temporary storage of the uploaded file
        temp_file_path = f"{upload_file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            content = await upload_file.read()
            temp_file.write(content)

        # Process each PDF file
        # Set pdf_file_path to the path of the PDF file
        pdf_file_path = temp_file_path  # or your specific path
        file_name = upload_file.filename
        file_base = file_name.replace('.pdf', '')
        Mask_csv_path = '/app/src/text_extraction/output/page_masks/'
        #Mask_csv_path = os.path.dirname(__file__)
        file_pattern = os.path.join(Mask_csv_path, file_base + '_page_*_mask_summary.csv')
        #print(file_pattern)
        #print('working directory',os.getcwd())
        # Use glob to find all files that match the pattern
        print("File Pattern:", file_pattern)
        csv_files = glob.glob(file_pattern)
        print("Found CSV files:", csv_files)

        # Initialize an empty DataFrame to concatenate all the individual DataFrames
        combined_df = pd.DataFrame()

        # Loop through each file path and read the CSV file into a DataFrame
        for file in csv_files:
            df = pd.read_csv(file)

            # Extract specific columns
            coordinates_df = df[['x0f', 'x1f', 'y0f', 'y1f','mask_id','mask_shape','category_lbl']].copy()

            # Extract the page number from the file name
            # Assuming the format is always like '..._page_XXXX_...'
            page_number = os.path.basename(file).split('_')[2]

            # Add a new column for the page number
            coordinates_df['page_number'] = extract_page_number(file)
            print('page number', extract_page_number(file))

            # Concatenate the current DataFrame with the combined DataFrame
            combined_df = pd.concat([combined_df, coordinates_df], ignore_index=True)

        # Display the combined data
        print(combined_df)


        document_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
        df_sorted = combined_df.sort_values(by=[combined_df.columns[7], combined_df.columns[4]])

        output_dir = '/app/src/text_extraction/output_json_text'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        concatenated_text = ""
        json_structure = {"paper_id": document_name, "title": "", "paper_text": []}

        for index, row in df_sorted.iterrows():
            coords = (row['x0f'], row['y0f'], row['x1f'], row['y1f'])
            page_number = int(row['page_number']) - 1
            category = row['category_lbl']
            numbers = row['mask_shape'].strip("()").split(", ")

            # Ensure the function extract_text_from_scaled_pdf is defined or imported
            extracted_text = extract_text_from_scaled_pdf(pdf_file_path, page_number, coords, numbers)

            lines = extracted_text.split('\n')
            processed_lines = [line[:-1] if line.endswith('-') else line for line in lines]
            single_line_text = ''.join(processed_lines).strip()
            #single_line_text = textwrap.fill(single_line_text, width=80)

            if category == 'title':
                single_line_text = "\n## " + single_line_text + " ##\n"

            decoded_text = json.dumps(single_line_text, ensure_ascii=False)

            concatenated_text += "\n" + single_line_text

            section_dict = {
                "section_name": "",  # Set based on your data/logic
                "section_text": decoded_text,
                "section_annotation": category,
                "section_page": page_number + 1,
                "section_column": 0,  # Set based on your data/logic
                "section_location": [coords]
            }
            json_structure["paper_text"].append(section_dict)

        #json_output = json.dumps(json_structure, indent=4)
        json_output_file_path = f'{output_dir}/{document_name}.json'
        #with open(json_output_file_path, 'w') as json_file:
        #    json_file.write(json_output)

        output_file_path = f'{output_dir}/{document_name}.txt'
        with open(output_file_path, 'w') as file:
            file.write(concatenated_text)

    return json_output_file_path, output_file_path

@app.post("/process-pdfs")
async def create_upload_files(files: List[UploadFile] = File(...)):
    return await process_pdfs(files)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)