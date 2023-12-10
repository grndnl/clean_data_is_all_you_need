# %% IMPORTS

import json
import os
import shutil
from ast import literal_eval
from os.path import join

import fitz  # PyMuPDF
import numpy as np
import pandas as pd

from dla_pipeline_support_functions import (
    find_files_recursively,
    get_filename_without_extension,
    list_files_with_extensions,
    reset_directory,
)

pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 999)
pd.set_option("display.width", 999)

# %% Support Functions


# Debug, to ensure that the the contents on the individual csvs match the registry one
def valadiate_results_csvs(page_mask_directory):
    try:
        agg_csvs = pd.DataFrame()

        for file in list_files_with_extensions(page_mask_directory, ["csv"]):
            file_name = get_filename_without_extension(file)

            if file_name == "mask_registry":
                mask_registry = pd.read_csv(file)

            else:
                agg_csvs = pd.concat([agg_csvs, pd.read_csv(file)], axis=0)

        agg_csvs.drop("Unnamed: 0", axis=1, inplace=True)
        mask_registry.drop("Unnamed: 0", axis=1, inplace=True)

        agg_csvs.sort_values(["document", "page_no", "mask_id"], inplace=True)
        mask_registry.sort_values(["document", "page_no", "mask_id"], inplace=True)

        # Check that they are the same
        assert len(mask_registry) == len(
            agg_csvs
        ), "The aggregated content of all the individual page csvs and the mask_registry, do not have the same length"
        assert np.array_equal(
            agg_csvs, mask_registry
        ), "The aggregated content of all the individual page csvs and the mask_registry, do not match"

    except Exception as e:
        raise Exception(f"Match error: {e}")


def load_mask_registry(page_mask_directory, validate_csvs=False):
    valadiate_results_csvs(page_mask_directory)
    mask_registry = pd.read_csv(join(page_mask_directory, "mask_registry.csv"))

    mask_registry.sort_values(["document", "page_no", "mask_id"], inplace=True)

    mask_registry = mask_registry[
        [
            "document",
            "page_no",
            "mask_id",
            "category",
            "category_lbl",
            "score",
            "x0",
            "x1",
            "y0",
            "y1",
            "xcf",
            "ycf",
            "column",
            "mask_shape",
            "is_primary",
            "mask_img_file_names",
            "mask_file_names",
        ]
    ]

    mask_registry["mask_shape"] = mask_registry["mask_shape"].apply(
        lambda var: literal_eval(var)
    )

    return mask_registry


# %% Text Extraction


def extract_text_from_scaled_pdf(pdf_path, page_number, coords, new_dimensions):
    """Extract text from specified coordinates in a scaled PDF."""
    doc = fitz.open(pdf_path)
    scaled_doc = fitz.open()  # Create a new empty PDF for scaled pages

    # Scale the specific page
    page = doc.load_page(page_number)
    new_page = scaled_doc.new_page(
        width=int(new_dimensions[1]), height=int(new_dimensions[0])
    )
    new_page.show_pdf_page(new_page.rect, doc, page.number)

    # Extract text from the scaled page
    scaled_page = scaled_doc.load_page(0)  # As we have only one page in scaled_doc
    extracted_text = scaled_page.get_text("text", clip=fitz.Rect(coords))
    # extracted_text = ' '.join(extracted_text.split())

    # Clean up
    doc.close()
    scaled_doc.close()
    return extracted_text


def process_pdfs_local(data_directory: str, single_directory_output: bool = False):
    # Initial File Setup #################################################

    S1_INPUT_PDFS_DIR = join(data_directory, "s1_input_pdfs")
    S2_DLA_INPUTS_DIR = join(data_directory, "s2_dla_inputs")
    S3_OUTPUTS_DIR = join(data_directory, "s3_outputs")
    S4_JSON_TEXT_OUTPUTS_DIR = join(data_directory, "s4_json_text_output")
    PAGE_MASK_DIR = join(S3_OUTPUTS_DIR, "page_masks")

    reset_directory(S4_JSON_TEXT_OUTPUTS_DIR, erase_contents=True, verbose=True)

    # Look for results
    mask_registry = load_mask_registry(PAGE_MASK_DIR, validate_csvs=True)
    pdf_list = np.unique(mask_registry["document"].values)
    model_support_images = list_files_with_extensions(
        join(S3_OUTPUTS_DIR, "model_outputs"), ["jpg"]
    )

    process_log = []

    for pdf_file in pdf_list:
        process_dict = {
            "pdf_file": pdf_file,
            "pages_processed": -1,
            "output_directory": "",
            "process_complete": False,
            "error_message": "",
        }
        try:
            pdf_name = get_filename_without_extension(pdf_file)

            pdf_file_path = join(S1_INPUT_PDFS_DIR, pdf_file)
            assert os.path.exists(pdf_file_path), f"PDF File {pdf_file_path}, Not Found"

            doc_mask_registry = mask_registry.query(
                f"document=='{pdf_file}' & is_primary==True"
            )
            assert len(doc_mask_registry) != 0, f"No results found for {pdf_file}"

            doc_mask_registry.sort_values(by=["mask_id", "page_no"], inplace=True)

            # Used for section names
            doc_cat_dict = {cat:0 for cat in np.unique(doc_mask_registry['category_lbl'].values)}


            # SETUP OUTPUT DIR
            if single_directory_output:
                doc_output_dir = S4_JSON_TEXT_OUTPUTS_DIR
            else:
                doc_output_dir = join(S4_JSON_TEXT_OUTPUTS_DIR, pdf_name)
                reset_directory(doc_output_dir, erase_contents=True)

            ## Text Extraction Fabian base code ###############################
            ###################################################################

            concatenated_text = ""
            json_structure = {"paper_id": pdf_file, "title": "", "paper_text": []}

            for i, row in doc_mask_registry.iterrows():
                coords = (row["x0"], row["y0"], row["x1"], row["y1"])
                page_number = row["page_no"] - 1
                category = row["category_lbl"]
                numbers = row["mask_shape"]

                ## TEXT
                extracted_text = extract_text_from_scaled_pdf(
                    pdf_file_path, page_number, coords, numbers
                )

                lines = extracted_text.split("\n")
                processed_lines = [
                    line[:-1] if line.endswith("-") else line for line in lines
                ]
                single_line_text = " ".join(processed_lines).strip()
                # single_line_text = textwrap.fill(single_line_text, width=80)

                if category == "title":
                    single_line_text = "\n## " + single_line_text + " ##\n"

                concatenated_text += "\n" + single_line_text

                ## JSON
                doc_cat_dict[category] += 1
                section_name = f"{category}_{doc_cat_dict[category]:03}"

                decoded_text = json.dumps(single_line_text, ensure_ascii=False)

                section_dict = {
                    "section_name": section_name,
                    # "section_text": decoded_text,
                    "section_text": single_line_text,
                    "section_annotation": row["category_lbl"],
                    "section_page": row["page_no"],
                    "section_id": row['mask_id'],
                    "section_column": row['column'],
                    "section_im_bbox": (row["x0"], row["y0"], row["x1"], row["y1"]),                    
                }

                json_structure["paper_text"].append(section_dict)

                ## IMAGES
                mask_img_f_name = row["mask_img_file_names"]
                shutil.copy2(join(PAGE_MASK_DIR, mask_img_f_name), doc_output_dir)

            # Save TEXT FILE
            output_file_path = join(doc_output_dir, pdf_name + ".txt")
            with open(output_file_path, "w") as file:
                file.write(concatenated_text)

            # Save JSON FILE
            # json_output = json.dumps(json_structure, indent=4)
            json_output = json.dumps(json_structure, indent=4, ensure_ascii=False)
            json_output_file_path = join(doc_output_dir, pdf_name + ".json")
            with open(json_output_file_path, "w") as json_file:
                json_file.write(json_output)

            ###################################################################

            # COPY SUPPORTING FILES
            doc_mask_registry_path = join(
                doc_output_dir, f"{pdf_name}_mask_registry.csv"
            )
            doc_mask_registry.to_csv(doc_mask_registry_path, index=False)

            # Useful Images
            for im_path in model_support_images:
                im_name = get_filename_without_extension(im_path)
                if im_name.startswith(pdf_name) and im_name.endswith(
                    "_base_dla_result"
                ):
                    shutil.copy2(im_path, doc_output_dir)

            process_dict["process_complete"] = True

        except Exception as e:
            process_dict["process_complete"] = False
            process_dict["error_message"] = e

        finally:
            if i is not None:
                process_dict["pages_processed"] = i + 1

            if doc_output_dir is not None:
                process_dict["output_directory"] = doc_output_dir

            process_log.append(process_dict)

            pd.DataFrame(process_log).to_csv(
                join(S4_JSON_TEXT_OUTPUTS_DIR, "text_extract.csv"), index=False
            )


if __name__ == "__main__":
    data_directory = "/data"

    assert os.path.exists(data_directory)

    return_dict = process_pdfs_local(data_directory, False)

    print(json.dumps(return_dict, indent=4))
