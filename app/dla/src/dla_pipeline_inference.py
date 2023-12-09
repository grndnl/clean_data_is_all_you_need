# %% IMPORTS

import json
import os
import sys
from datetime import datetime
from os.path import join

import matplotlib.pyplot as plt

import argparse

# Pandas display options
import numpy as np
import pandas as pd
import yaml

from torch.cuda import is_available


from dla_pipeline_support_classes import (
    DLAModelDetection,
    DLAVisualizer,
    PDF_DocumentSet,
)
from dla_pipeline_support_functions import (
    display_images_with_titles,
    generate_batch_load_json,
    reset_directory,
)

from dla_text_extraction import process_pdfs_local

pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 999)
pd.set_option("display.width", 999)

# %% FUNCTION #################################################################
###############################################################################

available_models = ["DIT", "LAYOUTLMV3"]


def script_directory():
    return os.path.dirname(os.path.abspath(sys.argv[0]))


def process_documents(
    full_inference: bool = True,
    continue_from_previous: bool = False,
    model_type: str = "DIT",
    use_cpu: bool = False,
):
    """
    Function that runs the DLA pipeline

    Options:
        full_inference:
            TRUE: All pdf documents will be fully processed and run through the DLA model
            FALSE: Processing will be based on existing dla model results

        continue_from_previous:
            TRUE: Masks will be generated only for pages where no existing masks are found
            FALSE: New masks will be generated for all pages


        model_type:
            Model used for DLA inference.
            Available options “DIT” and “LAYOUTLMV3”
    """

    # INITIAL SETUP ###############################################################
    ###############################################################################

    RESET_DATA_DIRECTORIES = full_inference
    RUN_MODEL = full_inference
    if full_inference:
        continue_from_previous = False

    # LOAD settings.yaml
    settings_path = join(script_directory(), "dla_pipeline_settings.yaml")

    with open(settings_path, "r") as file:
        settings = yaml.safe_load(file)

    # DIRECTORIES AND CONFIGURATIONS ##############################################
    ###############################################################################

    # Data Directories
    DATA_DIRECTORY = settings["DIRECTORIES"]["DATA_DIRECTORY"]
    S1_INPUT_PDFS_DIR = join(DATA_DIRECTORY, "s1_input_pdfs")
    S2_DLA_INPUTS_DIR = join(DATA_DIRECTORY, "s2_dla_inputs")
    S3_OUTPUTS_DIR = join(DATA_DIRECTORY, "s3_outputs")
    S4_JSON_TEXT_OUTPUTS_DIR = join(DATA_DIRECTORY, "s4_json_text_output")

    reset_directory(
        directory_path=S2_DLA_INPUTS_DIR,
        erase_contents=RESET_DATA_DIRECTORIES,
        verbose=True,
    )
    reset_directory(
        directory_path=S3_OUTPUTS_DIR,
        erase_contents=RESET_DATA_DIRECTORIES,
        verbose=True,
    )

    # DLA Categories
    MODEL_CATEGORIES_JSON = join(DATA_DIRECTORY, "dla_categories.json")

    # Validate primary requirements
    for p in [
        S1_INPUT_PDFS_DIR,
        S2_DLA_INPUTS_DIR,
        S3_OUTPUTS_DIR,
        MODEL_CATEGORIES_JSON,
    ]:
        assert os.path.exists(p), f"DATA PATH NOT FOUND, F{p} does not exist."

    # Secondary requirements
    MODEL_INPUT_JSON = join(S3_OUTPUTS_DIR, "model_input.json")
    MODEL_OUTPUT_JSON = join(S3_OUTPUTS_DIR, "inference/coco_instances_results.json")

    # Model modules and settings
    if model_type not in available_models:
        raise Exception(f"MODEL_TYPE: {model_type}, not recognized")

    elif model_type == "DIT":
        from models.dit.object_detection.dla_pipeline_train_net import run_inference

        MODEL_CONFIG = join(
            script_directory(),
            "models/dit/object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml",
        )

    elif model_type == "LAYOUTLMV3":
        from models.layoutlmv3.object_detection.dla_pipeline_train_net import (
            run_inference,
        )

        MODEL_CONFIG = join(
            script_directory(),
            "models/layoutlmv3/object_detection/cascade_layoutlmv3.yaml",
        )

    MODEL_WEIGHTS = settings["MODEL"][model_type]["MODEL_WEIGHTS"]

    # FINAL DIRECTORY VALIDATION ##################################################
    ###############################################################################

    if full_inference:
        assert os.path.exists(
            MODEL_WEIGHTS
        ), f"MODEL Weights not found. {MODEL_WEIGHTS}, does not exist"
        assert os.path.exists(
            MODEL_CONFIG
        ), f"MODEL Config not found. {MODEL_CONFIG}, does not exist"
    else:
        assert os.path.exists(
            MODEL_OUTPUT_JSON
        ), f"MODEL Results not found, new inference required. {MODEL_OUTPUT_JSON}, does not exist"

    # LOAD AND SETUP CATEGORIES DICT ##############################################
    ###############################################################################

    try:
        with open(MODEL_CATEGORIES_JSON, "r") as json_file:
            categories_list = json.load(json_file)

        categories_dict = {cat["id"]: cat["name"] for cat in categories_list}

        categories_dict[0] = "unkown"

    except Exception as e:
        raise Exception(f"ERROR Opening {MODEL_CATEGORIES_JSON}: {e}")

    # PROCESS DOCUMENTS ###########################################################
    ###############################################################################

    document_set = PDF_DocumentSet(
        S1_INPUT_PDFS_DIR,
        S2_DLA_INPUTS_DIR,
        S3_OUTPUTS_DIR,
        use_existing_images=True,
        dla_categories=categories_dict,
    )

    try:
        ## Initialize PDFs
        ###############################################

        document_set.add_to_log_dict("Processing started")
        document_set.add_to_log_dict("Loading documents and generating page images")

        document_set.initialize_document_set()
        page_images_list = document_set.page_images_list_df

        document_set.add_to_log_dict(
            f"LOADING SUCCESSFUL: {len(document_set.documents)} documents and {len(document_set.pages)} pages loaded!!"
        )

        if RUN_MODEL:
            ## Generating DiT loader JSON
            ###############################################

            document_set.add_to_log_dict("Generating DiT loader JSON")

            generate_batch_load_json(
                images_dir=S2_DLA_INPUTS_DIR,
                categories_json=MODEL_CATEGORIES_JSON,
                output_path=MODEL_INPUT_JSON,
            )

            assert os.path.exists(
                MODEL_INPUT_JSON
            ), f"MODEL Input json not found. {MODEL_INPUT_JSON}, does not exist"

            ## Inference
            ###############################################

            document_set.add_to_log_dict(
                f"Inference started with MODEL_TYPE: {model_type}"
            )

            document_set.add_to_log_dict(f"OPTIONS 'use_cpu': {use_cpu}")

            document_set.add_to_log_dict(f"Is GPU AVAILABLE: {is_available()}")

            document_set.add_to_log_dict(f"Processing: {len(page_images_list)}: pages")

            start = datetime.now()
            run_inference(
                config_file=MODEL_CONFIG,
                model_weights=MODEL_WEIGHTS,
                output_dir=S3_OUTPUTS_DIR,
                model_input_json=MODEL_INPUT_JSON,
                images_dir=S2_DLA_INPUTS_DIR,
                use_cpu=use_cpu,
            )
            duration = datetime.now() - start

            document_set.add_to_log_dict("Inference completed")
            document_set.add_to_log_dict(
                f"Total Execution Time: {duration.total_seconds():0.2f} seconds"
            )
            document_set.save_log("up_to_inference")

        ## Process Model Results
        ###############################################

        document_set.add_to_log_dict("Loading and processing results")

        document_set.add_to_log_dict("Removing pages with no DLA Model Results")
        document_set.remove_empty_pages(results_json_path=MODEL_OUTPUT_JSON)

        if continue_from_previous:
            document_set.add_to_log_dict("Loading previous results")
            document_set.remove_pre_processed_pages()
        else:
            document_set.add_to_log_dict("Resetting the output directories")
            document_set.reset_output_directories()

        document_set.add_to_log_dict("Processing DLA results")
        document_set.batch_process_dit_results(
            batch_size=50,
            results_json_path=MODEL_OUTPUT_JSON,
            use_rectangular_masks=False,
            margin_thickness=100,
            v_pad=-1,
            h_pad=3,
            clear_data_after=True,
        )

        ## Closeout
        ###############################################
        document_set.save_document_list()
        document_set.save_page_images_list()
        document_set.save_mask_registry()

        document_set.add_to_log_dict("Extracting Text")
        process_pdfs_local(DATA_DIRECTORY,single_directory_output=False)
        document_set.add_to_log_dict("Extracting Text: Complete")

        document_set.add_to_log_dict("Full process completed")
        exit_code = 0  # 0 for no errors
        exit_msg = f"SUCCESFULLY PROCESSED {len(document_set.document_list)} documents and {len(document_set.page_images_list)} pages"

    except Exception as e:
        document_set.add_to_log_dict("PROCESS ERROR")
        document_set.add_to_log_dict(e)
        exit_code = 1  # 1 for errors
        exit_msg = f"PROCESS ERROR: {e}. SEE execution_log for details"

        document_set.save_document_list()
        document_set.save_page_images_list()
        document_set.save_mask_registry()

    finally:

        document_set.save_log()

        return exit_code, exit_msg, S3_OUTPUTS_DIR, S4_JSON_TEXT_OUTPUTS_DIR


if __name__ == "__main__":
    # Manual values
    # If no arguments are passed, these values will be used
    full_inference = "store_true"
    continue_last = "store_true"
    model_type = "DIT"
    # model_type = "LAYOUTLMV3"
    use_cpu = "store_true"
    ###########################################################################

    # Parse arguments
    parser = argparse.ArgumentParser("Function that runs the DLA pipeline.")

    parser.add_argument("--full_inference", action=full_inference)
    parser.add_argument("--continue_last", action=continue_last)
    parser.add_argument("--model_type", type=str, default=model_type)
    parser.add_argument("--use_cpu", action=use_cpu)

    args = parser.parse_args()

    # Execute code
    print("\nArguments passed:\n")
    print(f" -> {args}\n")
    print("*" * 100 + "\n")

    exit_code = process_documents(
        full_inference=args.full_inference,
        continue_from_previous=args.continue_last,
        model_type=args.model_type,
        use_cpu=args.use_cpu,
    )

    print(f"EXIT CODE: {exit_code}")
