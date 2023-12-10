# %% IMPORTS

import json
import os
# from ast import literal_eval
from os.path import join

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from PIL.JpegImagePlugin import JpegImageFile
from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    LayoutLMv3Tokenizer,
)
import itertools
from dla_pipeline_support_functions import load_mask_registry

pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 999)
pd.set_option("display.width", 999)

# NOTE: Fix the underlaying issue
pd.options.mode.chained_assignment = None  # default='warn'

# %% Functions


# Normalize box diamentions to range 0 to 1000
def normalized_box(box, image_width, image_height):
    return [
        int(1000 * (box[0] / image_width)),
        int(1000 * (box[1] / image_height)),
        int(1000 * (box[2] / image_width)),
        int(1000 * (box[3] / image_height)),
    ]

def get_logit_scores_from_predictions(scores, predictions):
    return scores[np.arange(len(predictions)), predictions]


def logit_to_prob(logit):
    odds = np.exp(logit)
    prob = odds / (1 + odds)
    return prob

logit_to_prob_vect = np.vectorize(logit_to_prob)

def generate_doclaylet_dataset(
    page_image_registry: pd.DataFrame,
    doc_text_registry: pd.DataFrame,
    mask_registry: pd.DataFrame,
    max_text_length: int,
    s2_dla_inputs_dir: str,
):
    normalized_bbox_page_dict_list = []

    dataset_dict = {
        "document_id": [],
        "page_no": [],
        "images": [],
        "original_img_shape": [],
        "words": [],
        "bboxes": [],
        "normalized_bboxes": [],
        "umask_id": [],
        "dummy_label": [],
    }
    for i, row in doc_text_registry.iterrows():
        # DOCUMENT SPECIFIC VALUES ########################################
        ###################################################################

        doc_id = row["pdf_file"]

        doc_json_path = join(row["output_directory"], row["json_path"])
        with open(doc_json_path, "r") as json_file:
            doc_json = json.load(json_file)

        doc_json_df = pd.DataFrame(doc_json["paper_text"])

        # Ensure the box is read as numbers
        # doc_json_df["section_im_bbox"] = doc_json_df["section_im_bbox"].apply(
        #     lambda var: literal_eval(str(var))
        # )


        doc_json_df.sort_values(by=["section_page", "section_id"], inplace=True)

        # PAGE SPECIFIC VALUES ############################################
        ###################################################################
        doc_image_df = page_image_registry.query(f"document=='{doc_id}'")

        for ii, im_row in doc_image_df.iterrows():
            page_no = im_row["page_no"]

            # Dataset Doc Info
            dataset_dict["document_id"].append(doc_id)
            dataset_dict["page_no"].append(page_no)

            # Dataset Images
            page_img_path = join(s2_dla_inputs_dir, im_row["file_name"])
            page_img = JpegImageFile(page_img_path)
            dataset_dict["original_img_shape"].append(page_img.size)

            image_width, image_height = page_img.size
            dataset_dict["images"].append(page_img)

            # MASK SPECIFIC VALUES ########################################
            ###############################################################

            doc_page_json_df = doc_json_df.query(f"section_page=={page_no}")
            doc_page_json_df["mask_id"] = doc_page_json_df["section_id"]

            # Dataset Words
            #   NOTE: We need to ensure that per page there are less tokens
            #   generated than the maximum number of tokens allowed. The code
            #   below, ensures that each section gets tokens.

            page_text = doc_page_json_df["section_text"].to_list()

            no_masks = len(page_text)
            words_per_mask = int((max_text_length / no_masks) * 0.60)

            short_page_text = []
            for section_text in page_text:
                short_txt = section_text.split(" ")[0:words_per_mask]
                short_txt = " ".join(short_txt)
                short_page_text.append(short_txt)

            dataset_dict["words"].append(short_page_text)

            # Dataset bboxes
            bboxes = doc_page_json_df["section_im_bbox"].to_list()

            normalized_bboxes = [
                normalized_box(bboxs, image_width, image_height) for bboxs in bboxes
            ]

            dataset_dict["bboxes"].append(bboxes)

            dataset_dict["normalized_bboxes"].append(normalized_bboxes)

            # Unique Mask ID:
            page_mask_df = mask_registry.query(
                f"document=='{doc_id}' & page_no=={page_no}"
            )
            page_mask_df = pd.merge(
                doc_page_json_df, page_mask_df, on="mask_id", how="inner"
            )
            umask_ids = page_mask_df["umask_id"].to_list()

            assert len(short_page_text) == len(bboxes) == len(umask_ids)

            dataset_dict["umask_id"].append(umask_ids)
            dataset_dict["dummy_label"].append(np.zeros(len(umask_ids)))

            normalized_bbox_page_dict = {}

            for umask_id, n_bbox in zip(umask_ids, normalized_bboxes):
                normalized_bbox_page_dict[tuple(n_bbox)] = umask_id

            normalized_bbox_page_dict_list.append(normalized_bbox_page_dict)

    return Dataset.from_dict(dataset_dict), dataset_dict, normalized_bbox_page_dict_list


def process_single_page(
    idx: int,
    dataset_dict: dict,
    model: LayoutLMv3ForTokenClassification,
    processor: LayoutLMv3Processor,
    max_length: int,
    normalized_bbox_page_dict_list: list,
    mask_registry: pd.DataFrame,
    id2label: dict,
):
    img = dataset_dict["images"][idx]
    texts_list = dataset_dict["words"][idx]
    normalized_bboxes = dataset_dict["normalized_bboxes"][idx]
    normalized_bbox_page_dict = normalized_bbox_page_dict_list[idx]

    encoding = processor(
        img,
        texts_list,
        boxes=normalized_bboxes,
        truncation=True,
        stride=128,
        padding="max_length",
        max_length=max_length,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    offset_mapping = encoding.pop("offset_mapping")

    overflow_to_sample_mapping = encoding.pop("overflow_to_sample_mapping")

    # change the shape of input_ids ############################################
    x = []
    for i in range(0, len(encoding["input_ids"])):
        x.append(torch.tensor(encoding["input_ids"][i]))
    x = torch.stack(x)
    encoding["input_ids"] = x

    # change the shape of pixel values

    x = []
    for i in range(0, len(encoding["pixel_values"])):
        x.append(torch.from_numpy(encoding["pixel_values"][i]))
    x = torch.stack(x)
    encoding["pixel_values"] = x

    x = []
    for i in range(0, len(encoding["attention_mask"])):
        x.append(torch.tensor(encoding["attention_mask"][i]))
    x = torch.stack(x)
    encoding["attention_mask"] = x

    # change the shape of bbox
    x = []
    for i in range(0, len(encoding["bbox"])):
        x.append(torch.tensor(encoding["bbox"][i]))
    x = torch.stack(x)
    encoding["bbox"] = x

    ## RUN MODEL
    outputs = model(**encoding)

    logits = outputs.logits

    predictions = logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    if len(token_boxes) == 512:
        predictions = [predictions]
        token_boxes = [token_boxes]

    predictions = list(itertools.chain(*predictions))
    token_boxes = list(itertools.chain(*token_boxes))


    # GET Prediction Probability Scores from the logits

    logits_all_cats = logits.squeeze()

    prediction_logits = get_logit_scores_from_predictions(logits_all_cats, predictions)
    prediction_scores = logit_to_prob_vect(prediction_logits.detach().numpy())


    # AGREGATE ALL THE PREDICTIONS FOR THE SAME MASK TYPE

    prediction_dict = {}
    prediction_score_dict = {}

    for prediction, prediction_scr, token_bx in zip(
        predictions, prediction_scores, token_boxes
    ):
        token_bx_tup = tuple(token_bx)

        if token_bx_tup not in prediction_dict.keys():
            prediction_dict[token_bx_tup] = []
            prediction_score_dict[token_bx_tup] = []

        prediction_dict[token_bx_tup].append(prediction)
        prediction_score_dict[token_bx_tup].append(prediction_scr)

    # Add the scores of all categories and select the one with the highest average
    for token_bx in prediction_dict.keys():
        pred_array = prediction_dict[token_bx]
        pred_score_array = prediction_score_dict[token_bx]

        cats = np.zeros(len(id2label)).tolist()

        for i in range(len(pred_array)):
            i_c = pred_array[i]
            cats[i_c] += pred_score_array[i]

        cats = np.array(cats)

        cats = cats / (i + 1)

        most_common_value = np.argmax(cats)

        prediction_dict[token_bx] = most_common_value
        prediction_score_dict[token_bx] = cats[most_common_value]


    # UPDATE MASK REGISTRY
    for box_tup, pred in prediction_dict.items():
        if box_tup in normalized_bbox_page_dict.keys():
            umask_id = normalized_bbox_page_dict[box_tup]

            matching_index = mask_registry.index[
                mask_registry["umask_id"] == umask_id
            ].tolist()[0]

            mask_registry.at[matching_index, "new_category"] = pred
            mask_registry.at[matching_index, "new_category_lbl"] = id2label[pred]

            pred_score = prediction_score_dict[box_tup]
            mask_registry.at[matching_index, "new_category_score"] = pred_score

def process_documents_phase_2(data_directory: str):
    exit_code = 1
    error_message = "Not Executed"
    try:
        S1_INPUT_PDFS_DIR = join(data_directory, "s1_input_pdfs")
        S2_DLA_INPUTS_DIR = join(data_directory, "s2_dla_inputs")
        S3_OUTPUTS_DIR = join(data_directory, "s3_outputs")
        S4_JSON_TEXT_OUTPUTS_DIR = join(data_directory, "s4_json_text_output")
        PAGE_MASK_DIR = join(S3_OUTPUTS_DIR, "page_masks")

        PRETRAINED_MODEL_DIR = "/user/w210/large_file_repo/models_pretrained"
        MODEL_TAG = (
            "layoutlmv3-finetuned-DocLayNet_large_sci_23_12_02-15_50_34/checkpoint-5946"
        )
        MODEL_WEIGHTS = join(PRETRAINED_MODEL_DIR, MODEL_TAG)
        MODEL_PROCESSOR = join(PRETRAINED_MODEL_DIR, "microsoft-layoutlmv3-base-processor")
        MODEL_CATEGORIES_JSON = join(data_directory, "dla_categories_doclaynet.json")

        GLOBAL_BATCH_SIZE = 1
        MAX_LENGTH = 512

        for pth in [
            data_directory,
            S1_INPUT_PDFS_DIR,
            S2_DLA_INPUTS_DIR,
            S3_OUTPUTS_DIR,
            S4_JSON_TEXT_OUTPUTS_DIR,
            PAGE_MASK_DIR,
            MODEL_WEIGHTS,
            MODEL_PROCESSOR,
            MODEL_CATEGORIES_JSON,
        ]:
            assert os.path.exists(pth), f"PATH NOT FOUND: {pth}"

        mask_registry = load_mask_registry(PAGE_MASK_DIR, validate_csvs=False)
        mask_registry["umask_id"] = np.arange(len(mask_registry))
        mask_registry["new_category"] = np.full(len(mask_registry), -1)
        mask_registry["new_category_lbl"] = "Unkown"
        mask_registry["new_category_score"] = np.zeros(len(mask_registry))

        page_image_registry = pd.read_csv(join(S3_OUTPUTS_DIR, "page_images_list.csv"))

        doc_text_registry = pd.read_csv(join(S4_JSON_TEXT_OUTPUTS_DIR, "text_extract.csv"))

        doc_text_registry["json_path"] = doc_text_registry.apply(
            lambda var: var["pdf_file"].replace(".pdf", ".json"),
            axis=1,
        )

        ####################################################
        dataset, dataset_dict, normalized_bbox_page_dict_list = generate_doclaylet_dataset(
            page_image_registry,
            doc_text_registry,
            mask_registry,
            max_text_length=MAX_LENGTH,
            s2_dla_inputs_dir=S2_DLA_INPUTS_DIR,
        )

        # Create 1d2lbl
        with open(MODEL_CATEGORIES_JSON, "r") as json_file:
            categories_dict = json.load(json_file)

        id2label = {int(k): v for k, v in categories_dict.items()}
        label2id = {v: int(k) for k, v in categories_dict.items()}

        ## LOAD MODEL
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = LayoutLMv3Processor.from_pretrained(MODEL_PROCESSOR, apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            MODEL_WEIGHTS, id2label=id2label, label2id=label2id
        )

        for i in range(len(dataset)):

            try:
                torch.cuda.empty_cache()
                process_single_page(
                    idx=i,
                    dataset_dict=dataset_dict,
                    model=model,
                    processor=processor,
                    max_length=MAX_LENGTH,
                    normalized_bbox_page_dict_list=normalized_bbox_page_dict_list,
                    mask_registry=mask_registry,
                    id2label=id2label,
                )
            except Exception as e:
                pass

        exit_code = 0
        error_message = f"DLA Phase 2: SUCCESSFULLY PROCESSED {i+1} pages"
    except Exception as e:
        exit_code = 1
        error_message = f"DLA Phase 2, PROCESSING ERROR: {e}"


    return mask_registry, exit_code, error_message


if __name__ == "__main__":
    if not os.path.exists("/data"):
        os.symlink("/user/w210/clean_data_is_all_you_need/app/data", "/data")

    DATA_DIRECTORY = "/data"

    mask_registry = process_documents_phase_2(DATA_DIRECTORY)

    # print(mask_registry.query("is_primary=='True"))

    
