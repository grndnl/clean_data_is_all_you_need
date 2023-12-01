import glob
import pandas as pd
import os
import fitz  # PyMuPDF
import textwrap

from typing import Union

from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/")
def not_implemented():
    raise HTTPException(status_code=501, detail="This functionality is not implemented") # 501 Not Implemented, Is a server error as the functionality is not implemented.


@app.get("/hello") #localhotst/hello
def read_item(name: str = None):
    if name is None:
        raise HTTPException(status_code=400, detail="Name parameter is required") # 400 Bad Request, Is a client error as the parameter "name" was not included.
    return {"message": f"Hello {name}"}

#File Location and name

file_name = "1603.09631.pdf"
file_base = file_name.replace('.pdf', '')
pdf_file_path = '../../pdf_documents/'+file_name  #Where PDF is Located
Mask_csv_path = '../../output/page_masks/' #Where the csv mask output from the DLA is located

file_pattern = Mask_csv_path + file_base + '_page_*_mask_mask_summary.csv'
print(file_pattern)

# Use glob to find all files that match the pattern
#print("File Pattern:", file_pattern)
csv_files = glob.glob(file_pattern)
#print("Found CSV files:", csv_files)


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
    coordinates_df['page_number'] = int(page_number)

    # Concatenate the current DataFrame with the combined DataFrame
    combined_df = pd.concat([combined_df, coordinates_df], ignore_index=True)

# Display the combined data
print(combined_df)

#page_data = combined_df[combined_df['page_number'] == 1]
#print(page_data)




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

# Path to your CSV and PDF files

pdf_file_path = pdf_file_path
# Extract the base name of the PDF file (e.g., '1909.11687' from '1909.11687.pdf')
document_name = os.path.splitext(os.path.basename(pdf_file_path))[0]

# Reading the CSV file
df = combined_df
df_sorted = df.sort_values(by=[df.columns[7], df.columns[4]])
print(df_sorted)

# Initialize an empty string to concatenate text
concatenated_text = ""


# Extract text from each row's coordinates
for index, row in df_sorted.iterrows():
    coords = (row['x0f'], row['y0f'], row['x1f'], row['y1f'])
    page_number = int(row['page_number']) - 1  # Convert to zero-based index for PDF pages
    mask_shape = row['mask_shape']
    category = row['category_lbl']
    numbers = mask_shape.strip("()").split(", ")
    extracted_text = extract_text_from_scaled_pdf(pdf_file_path, page_number, coords, numbers)

    # Split the extracted text into lines
    lines = extracted_text.split('\n')

    # Process each line
    processed_lines = []
    for line in lines:
        # Check if the line ends with a hyphen
        #if line.endswith('-  '):
        #    processed_lines.append(line[:-3])
        #if line.endswith('- '):
            # Remove the hyphen and do not add a space after this line
        #    processed_lines.append(line[:-2])
        if line.endswith('-'):
            processed_lines.append(line[:-1])
        else:
            # Add a space at the end of the line for normal concatenation
            processed_lines.append(line)
    print(f"Extracted Text from Page {page_number+1}, Line {index}:", processed_lines)
    # Join all lines into a single line
    single_line_text = ''.join(processed_lines).strip()
    single_line_text = textwrap.fill(single_line_text, width=80)

    # Format the text for titles
    if category == 'title':
        single_line_text = "\n## " + single_line_text + " ##\n"

    concatenated_text += "\n" + single_line_text


# Print the concatenated text
print("\nConcatenated Text:\n", concatenated_text)

# Save the concatenated text into a .txt file
output_dir = '../../output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file_path = f'{output_dir}/{document_name}_extracted_text.txt'
with open(output_file_path, 'w') as file:
    file.write(concatenated_text)

print(f"Concatenated text saved to {output_file_path}")

#Json Creation
import json
import fitz  # PyMuPDF

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

    # Clean up
    doc.close()
    scaled_doc.close()
    return extracted_text

# Path to your PDF file
pdf_file_path = pdf_file_path
document_name = os.path.splitext(os.path.basename(pdf_file_path))[0]

# Reading the CSV file
# Make sure you define 'combined_df' before this line
df = combined_df
df_sorted = df.sort_values(by=[df.columns[7], df.columns[4]])

# Initialize the JSON structure outside the loop
json_structure = {
    "paper_id": document_name,
    "title": "",  # Set this based on your data/logic
    "paper_text": []
}



# Extract and process text
for index, row in df_sorted.iterrows():
    coords = (row['x0f'], row['y0f'], row['x1f'], row['y1f'])
    page_number = int(row['page_number']) - 1
    mask_shape = row['mask_shape']
    category = row['category_lbl']
    numbers = mask_shape.strip("()").split(", ")
    extracted_text = extract_text_from_scaled_pdf(pdf_file_path, page_number, coords, numbers)

    # Process the extracted text
    lines = extracted_text.split('\n')
    processed_lines = [line[:-1] if line.endswith('-') else line for line in lines]
    single_line_text = ''.join(processed_lines).strip()
    single_line_text = textwrap.fill(single_line_text, width=80)

    # Add to JSON structure
    section_dict = {
        "section_name": "",  # Set this based on your data/logic
        "section_text": single_line_text,
        "section_annotation": category,
        "section_page": page_number + 1,
        "section_column": 0,  # Set this based on your data/logic
        "section_location": [coords]
    }
    json_structure["paper_text"].append(section_dict)

# Convert the structured dictionary to JSON
json_output = json.dumps(json_structure, indent=4)

# Save the JSON output to a file
output_dir = '../../output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

json_output_file_path = f'{output_dir}/{document_name}_extracted_text.json'
with open(json_output_file_path, 'w') as json_file:
    json_file.write(json_output)

print(f"JSON output saved to {json_output_file_path}")