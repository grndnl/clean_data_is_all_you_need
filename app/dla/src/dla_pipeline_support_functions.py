# %% Imports

import json
import os
import shutil
from ast import literal_eval
from datetime import datetime
from os.path import join

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import sepfir2d


# %% FILE IO Support Functions
def find_files_recursively(directory):
    file_list = []
    dir_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))

        for dir in dirs:
            dir_list.append(os.path.join(root, dir))

    return file_list, dir_list


def delete_contents_in_directory(directory_path, verbose=False):
    try:
        # Delete all contents in the directory recursively
        # shutil.rmtree(directory_path)
        # os.makedirs(directory_path)

        # NOTE: Just using rmtree sometimes produces an error where
        # sub-directories cannot be removed if stuff is in them.
        # Deleting the files 1st seems to help.

        files, dirs = find_files_recursively(directory_path)

        for f in files:
            os.remove(f)

        for d in dirs:
            shutil.rmtree(d)

        if verbose:
            print(f'All contents in "{directory_path}" have been deleted successfully.')
    except Exception as e:
        print(f"An error occurred: {e}")
        raise Exception(e)


def reset_directory(directory_path, erase_contents=False, verbose=False):
    """
    Checks if a directory exists.
        - If it does it can erase all condents.
        - If it does not, it will create it.
    """
    try:
        if os.path.exists(directory_path):
            if verbose:
                print(f"Directory: {directory_path}, was found.")
            if erase_contents:
                delete_contents_in_directory(directory_path, verbose)
        else:
            os.makedirs(directory_path)
            if verbose:
                print(f"Directory: {directory_path}, was created.")
    except Exception as e:
        print(f"Unable to clear directory: {directory_path}")
        raise Exception(e)


def get_filename_without_extension(file_path):
    # Get the base name of the file without extension
    base_name = os.path.basename(file_path)
    # Get the filename without extension
    filename_without_extension = os.path.splitext(base_name)[0]
    return filename_without_extension


def list_files_with_extensions(folder_path, extensions):
    file_list = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(extension.lower()) for extension in extensions):
            file_list.append(join(folder_path, file))

    return np.sort(np.array(file_list))


# %% Debug Functions


def print_log_title(message, no_empty_lines: int = 2):
    print("\n" * no_empty_lines)
    print("*" * 80)
    print(message)


def display_images_with_titles(images_to_display, titles=None, figsize=(10, 30)):
    rows = 1
    columns = len(images_to_display)

    if titles == None:
        titles = [str(val) for val in range(columns)]

    plt.figure(figsize=figsize)

    # Loop through the images and display them in subplots
    for i, im in enumerate(images_to_display[0:columns]):
        # Add a subplot at the i-th position
        plt.subplot(rows, columns, i + 1)
        plt.imshow(im)
        plt.axis("off")  # Hide axis labels and ticks
        plt.title(titles[i])  # Set the title for each subplot

    # Adjust layout to prevent overlapping of titles
    plt.tight_layout()

    # Show the combined image
    plt.show()


# %% Array Manipulation Functions
def assert_mask_properties(mask_list):
    for i, mask in enumerate(mask_list):
        assert len(mask.shape) == 2, f"ERROR Mask {i}: Mask must have 2 dimensions"

        assert (
            np.array(mask.shape).min() > 1
        ), f"ERROR Mask {i}: All dimensions must have at least 2 elements"

        assert mask.dtype == "uint8", f"ERROR Mask {i}: Mask dtype must be uint8"

        assert np.array_equal(
            np.sort(np.unique(mask)), np.array([0, 1]).astype("uint8")
        ), f"ERROR Mask {i}: Mask must only contain 0 and 1"


def convert_non_zero_values(arr, to=1):
    zero_array = arr.copy()
    zero_array[zero_array != 0] = to

    return zero_array


def get_array_filled_ratio(arr):
    return np.count_nonzero(arr) / arr.size


def get_smallest_array_overlap(arr1, arr2):
    arr1 = convert_non_zero_values(arr1, to=1)
    arr2 = convert_non_zero_values(arr2, to=2)
    smallest_count = np.array([arr1[arr1 != 0].size, arr2[arr2 != 0].size]).min()
    arr_sum = arr1 + arr2
    overlap_count = arr_sum[arr_sum == 3].size

    return overlap_count / smallest_count, overlap_count


def get_array_overlap(arr1, arr2):
    """
    Rturns how much arr1 overlaps with arr2 as number elements and fraction of the arr1 total
    """
    arr1 = convert_non_zero_values(arr1, to=1)
    arr2 = convert_non_zero_values(arr2, to=2)

    arr1_count = arr1[arr1 != 0].size
    arr_sum = arr1 + arr2
    overlap_count = arr_sum[arr_sum == 3].size

    overlap_fraction = overlap_count / arr1_count

    return overlap_count, overlap_fraction


def stack_2_arrays(bottom, top):
    assert np.array_equal(
        bottom.shape, top.shape
    ), "ASSERT ERROR, stack_2_arrays, arrays not of equal shape"

    positions = np.argwhere(top != 0)
    stacked = bottom.copy()
    stacked[positions[:, 0], positions[:, 1]] = top[positions[:, 0], positions[:, 1]]

    return stacked


def crop_stacks_2(array_stack_list, invert_return=True):
    arr_count = len(array_stack_list)
    assert arr_count > 1, "ASSERT ERROR, crop_stacks_2, arr_count > 1"

    arr_tags = [i + 100 for i in range(arr_count)]

    # Stack
    for i, arr in enumerate(array_stack_list):
        tag = arr_tags[i]
        top = convert_non_zero_values(arr, tag)

        if i == 0:
            agregate_arr = top
            arr_shape = arr.shape
        else:
            agregate_arr = stack_2_arrays(agregate_arr, top)

    # Generate
    cropped_arrays = []

    for tag in arr_tags:
        positions = np.argwhere(agregate_arr == tag)
        cropped_array = np.zeros(arr_shape, dtype="uint8")
        cropped_array[positions[:, 0], positions[:, 1]] = 1
        cropped_arrays.append(cropped_array)

    if invert_return:
        return cropped_arrays[::-1]
    else:
        return cropped_arrays


def crop_stacks(array_stack_list):
    assert (
        len(array_stack_list) > 1
    ), "ASSERT ERROR, crop_stacks, len(array_stack_list) > 1"

    agregate_arr = convert_non_zero_values(array_stack_list[0], 1)

    # Stack
    for i, arr in enumerate(array_stack_list[1:]):
        top = convert_non_zero_values(arr, i + 2)
        agregate_arr = stack_2_arrays(agregate_arr, top)

    # Generate
    cropped_arrays = []
    for i, arr in enumerate(array_stack_list):
        positions = np.argwhere(agregate_arr == i + 1)
        cropped_array = np.zeros(arr.shape, dtype="uint8")
        cropped_array[positions[:, 0], positions[:, 1]] = 1
        cropped_arrays.append(cropped_array)

    return cropped_arrays


def generate_rectangular_mask(x0, x1, y0, y1, shape):
    mask = np.zeros(shape, dtype="uint8")
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    return cv2.fillPoly(mask, [pts], 1)


def aggregate_masks(mask_list):
    aggregate_mask = np.zeros(mask_list[0].shape, dtype="uint8")

    for mask in mask_list:
        aggregate_mask = stack_2_arrays(bottom=aggregate_mask, top=mask)

    return aggregate_mask


def aggregate_masks_tagged(mask_list, tag_multiplier=2):
    no_items = len(mask_list)
    tag_list = np.arange(
        start=tag_multiplier, stop=tag_multiplier * (no_items + 1), step=tag_multiplier
    )

    aggregate_mask = np.zeros(mask_list[0].shape, dtype="uint8")

    for i, mask in enumerate(mask_list):
        tag = tag_list[i]
        mask = mask.copy()
        mask[mask != 0] = tag
        aggregate_mask = stack_2_arrays(bottom=aggregate_mask, top=mask)

    return aggregate_mask, tag_list


def get_batch(input_list: list, batch_size: int):
    assert batch_size > 0, "ASSERT ERROR, get_batch, batch_size > 0"
    for i in range(0, len(input_list), batch_size):
        yield input_list[i : i + batch_size]


# %% Mask Post-Processing Functions
def get_mask_bounding_box(mask: np.ndarray, box_buffer: int = 0):
    assert len(mask.shape) == 2, "Mask must have 2 dimensions"
    assert (
        np.array(mask.shape).min() > 1
    ), "All dimensions must have at least 2 elements"
    assert mask.dtype == "uint8", "Mask dtype must be uint8"
    assert np.array_equal(
        np.sort(np.unique(mask)), np.array([0, 1]).astype("uint8")
    ), "Mask must only contain 0 and 1"

    positions = np.argwhere(mask != 0)

    if len(positions) > 0:
        x0 = positions[:, 1].min() - box_buffer
        x1 = positions[:, 1].max() + box_buffer

        y0 = positions[:, 0].min() - box_buffer
        y1 = positions[:, 0].max() + box_buffer
    else:
        x0 = 0
        x1 = mask.shape[0] - 1

        y0 = 0
        y1 = mask.shape[1] - 1

    return (x0, x1, y0, y1)


def apply_mask_to_image(img, mask, box=[]):
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    if len(box) == 4:
        x0 = box[0]
        x1 = box[1]
        y0 = box[2]
        y1 = box[3]

        cropped_img = masked_img[y0:y1, x0:x1, :]
        cropped_mask = mask[y0:y1, x0:x1]
    else:
        cropped_img = masked_img
        cropped_mask = mask

    return masked_img, cropped_img, mask, cropped_mask


def invert_mask_polarity(mask):
    out_mask = mask.copy()
    out_mask[out_mask == 0] = -1
    out_mask[out_mask == 1] = 0
    out_mask[out_mask == -1] = 1

    return out_mask


def apply_border_to_mask(mask, border_thickness=50, value=0):
    out_mask = mask.copy()
    x1, y1 = 0, 0  # Top-left corner
    x2, y2 = out_mask.shape[1], out_mask.shape[0]  # Bottom-right corner

    cv2.rectangle(out_mask, (x1, y1), (x2, y2), value, border_thickness)

    return out_mask


def pad_mask_perimeter(mask, padding: int = 0, padding_value: int = 1):
    if padding == 0:
        return mask
    else:
        padded_contours = []
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for i, cnt in enumerate(contours):
            msk = np.zeros(mask.shape, dtype="uint8")
            cv2.drawContours(msk, [cnt], -1, padding_value, padding * 2)
            cv2.drawContours(msk, [cnt], -1, padding_value, -1)

            padded_contours.append(msk)

        return aggregate_masks(padded_contours)


def binary_bin_channel(channel, thresh):
    mod_channel = channel.copy()

    mod_channel[mod_channel <= thresh] = 0
    mod_channel[mod_channel > thresh] = 255

    return mod_channel


# edge detection kernels
p = [0.30320, 0.249724, 0.439911, 0.249724, 0.30320]
d = [-0.104550, -0.292315, 0.0, 0.292315, 0.104550]


def extract_edge(im_g, d, p):
    im_x = sepfir2d(im_g, d, p)  # spatial (x) derivative
    im_y = sepfir2d(im_g, p, d)  # spatial (y) derivative
    return np.sqrt(im_y**2 + im_x**2)


def get_content_cluster_masks_bar_scan(
    masked_img,
    initial_mask=None,
    filled_threshold=0.1,
    bar_width=10,
    scan_step=5,
    binary_threshold=200,
):
    """
    This function identifies clusters of content on a white page and
    generates a mask for each cluster, allowing it to be isolated.
    If an initial mask is given, it is applied before searching for
    new clusters.
    """

    mask_groups = []

    # Convert image to pure black and white
    masked_img_bw = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    masked_img_bw[masked_img_bw <= binary_threshold] = 0
    masked_img_bw[masked_img_bw > binary_threshold] = 255

    ##############################################################################
    ## Apply initial mask
    ##############################################################################

    if initial_mask is None:
        initial_mask = np.ones(masked_img_bw.shape)

    # This tag will be used to designtate pixes to be masked out.
    # The assumption is white=empty space and black is content.
    mask_tag_value = 50

    # Apply initial mask
    positions = np.argwhere(initial_mask == 0)
    masked_img_bw[positions[:, 0], positions[:, 1]] = mask_tag_value

    ##############################################################################
    ## Mask out slices that only contain white spaces
    ##############################################################################

    # vertical scan
    run = True
    i0 = 0
    while run:
        i1 = i0 + bar_width

        if i1 > masked_img_bw.shape[0]:
            i1 = masked_img_bw.shape[0]
            run = False

        bar = masked_img_bw[i0:i1, :]

        if 0 not in bar:
            masked_img_bw[i0:i1, :] = mask_tag_value

        i0 += scan_step

    # horizontal scan
    run = True
    i0 = 0
    while run:
        i1 = i0 + bar_width

        if i1 > masked_img_bw.shape[1]:
            i1 = masked_img_bw.shape[1]
            run = False

        bar = masked_img_bw[:, i0:i1]

        if 0 not in bar:
            masked_img_bw[:, i0:i1] = mask_tag_value

        i0 += scan_step

    cleaned_mask = np.zeros(initial_mask.shape, dtype="uint8")
    positions = np.argwhere(masked_img_bw != mask_tag_value)
    cleaned_mask[positions[:, 0], positions[:, 1]] = 1

    ##############################################################################
    ## Find groups of content
    ##############################################################################

    # Find edges
    result_edges = extract_edge(cleaned_mask, d=d, p=p)
    cv2.normalize(result_edges, result_edges, 0, 255, cv2.NORM_MINMAX)
    result_edges = result_edges.astype("uint8")

    # Find contours from the edges
    contours, hierarchy = cv2.findContours(
        result_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    ln_thickness = -1
    v = 1

    # Filter out contours that are too small
    for i, cnt in enumerate(contours):
        msk = np.zeros(initial_mask.shape, dtype="uint8")
        cv2.drawContours(msk, [cnt], -1, v, ln_thickness)

        filled_ratio = get_array_filled_ratio(msk)
        if filled_ratio >= filled_threshold:
            mask_groups.append(msk)
        # else:
        #     print(f"rejected_filled_ratio: {filled_ratio}")

    return mask_groups


def get_content_cluster_masks_contours(
    masked_img,
    initial_mask=None,
    filled_threshold=0.1,
    binary_threshold=200,
):
    """
    This function identifies clusters of content on a white page and
    generates a mask for each cluster, allowing it to be isolated.
    If an initial mask is given, it is applied before searching for
    new clusters.
    """

    mask_groups = []

    # Convert image to grayscale
    masked_img_bw = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    # Binary bin
    masked_img_bw[masked_img_bw <= binary_threshold] = 0
    masked_img_bw[masked_img_bw > binary_threshold] = 255

    ##############################################################################
    ## Apply initial mask and fill with white space
    ##############################################################################

    if initial_mask is None:
        initial_mask = np.ones(masked_img_bw.shape)

    # Apply initial mask
    positions = np.argwhere(initial_mask == 0)
    masked_img_bw[positions[:, 0], positions[:, 1]] = 255

    ##############################################################################
    ## Blur and bin
    ##############################################################################

    # Apply a blur
    ks = 25  # kernel size
    masked_img_bw = cv2.GaussianBlur(masked_img_bw, (ks, ks), 0)

    # Binary bin
    masked_img_bw[masked_img_bw <= 250] = 0
    masked_img_bw[masked_img_bw > 250] = 255

    ##############################################################################
    ## Find groups of content
    ##############################################################################

    # Find edges
    result_edges = extract_edge(masked_img_bw, d=d, p=p)
    cv2.normalize(result_edges, result_edges, 0, 255, cv2.NORM_MINMAX)
    result_edges = result_edges.astype("uint8")

    # Find contours from the edges
    contours, hierarchy = cv2.findContours(
        result_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    ln_thickness = -1
    v = 1

    # Filter out contours that are too small
    for i, cnt in enumerate(contours):
        msk = np.zeros(initial_mask.shape, dtype="uint8")
        cv2.drawContours(msk, [cnt], -1, v, ln_thickness)

        filled_ratio = get_array_filled_ratio(msk)
        if filled_ratio >= filled_threshold:
            mask_groups.append(msk)
        else:
            print(f"rejected_filled_ratio: {filled_ratio}")

    return mask_groups


def find_mask_parent(target_mask, mask_list):
    assert_mask_properties(mask_list)
    assert_mask_properties([target_mask])

    agg_mask, agg_tags = aggregate_masks_tagged(mask_list)

    offset_tags = agg_tags + 1

    sum_mask = agg_mask + target_mask

    unique_values = np.unique(sum_mask)

    mask_overlap = []

    for i, tag in enumerate(offset_tags):
        if tag in unique_values:
            _, overlap_fraction = get_array_overlap(target_mask, mask_list[i])
            mask_overlap.append(overlap_fraction)
        else:
            mask_overlap.append(0.0)

    mask_overlap = np.array(mask_overlap)

    if mask_overlap.max() == 0:
        idx_of_max_overlap = -1
    else:
        idx_of_max_overlap = np.argwhere(mask_overlap == mask_overlap.max())[0][0]

    return idx_of_max_overlap, mask_overlap


def find_mask_list_parents(target_mask_list, parent_mask_list):
    assert_mask_properties(parent_mask_list)
    assert_mask_properties(target_mask_list)

    agg_mask, agg_tags = aggregate_masks_tagged(parent_mask_list)
    offset_tags = agg_tags + 1

    target_parents = []
    for target_mask in target_mask_list:
        sum_mask = agg_mask + target_mask

        unique_values = np.unique(sum_mask)

        mask_overlap = []

        for i, tag in enumerate(offset_tags):
            if tag in unique_values:
                _, overlap_fraction = get_array_overlap(
                    target_mask, parent_mask_list[i]
                )
                mask_overlap.append(overlap_fraction)
            else:
                mask_overlap.append(0.0)

        mask_overlap = np.array(mask_overlap)

        if mask_overlap.max() == 0:
            idx_of_max_overlap = -1
        else:
            idx_of_max_overlap = np.argwhere(mask_overlap == mask_overlap.max())[0][0]

        target_parents.append(idx_of_max_overlap)

    return target_parents


# %% Model Support Functions


def generate_batch_load_json(images_dir, categories_json, output_path):
    # build images dict
    images = []
    for page in list_files_with_extensions(images_dir, ["jpg", "jpeg"]):
        img = cv2.imread(page)
        file_name = os.path.basename(page)

        images.append(
            {
                "file_name": file_name,
                "height": img.shape[0],
                "id": os.path.splitext(file_name)[0],
                "width": img.shape[1],
            }
        )

    # load categories dict
    with open(categories_json, "r") as json_file:
        categories = json.load(json_file)

    metadata_dict = {"images": images, "annotations": [], "categories": categories}

    # Save json file
    with open(output_path, "w") as json_file:
        json.dump(metadata_dict, json_file)

    return metadata_dict


# %% COCO EDA FUNCTIONS


def load_coco_annotations_json(json_path, plot_details=True, plot_title="Data"):
    try:
        with open(json_path, "r") as json_file:
            json_dir = json.load(json_file)

        return generate_coco_annotations_df(json_dir, plot_details, plot_title)

    except Exception as e:
        print(f"Error loading {json_path}")
        raise Exception(f"Error loading {json_path}: {e}")


def generate_coco_annotations_df(json_dict, plot_details=True, plot_title="Data"):
    df_categories = pd.DataFrame(json_dict["categories"])
    df_images = pd.DataFrame(json_dict["images"])
    df_annotations = pd.DataFrame(json_dict["annotations"])

    if plot_details:
        print("Categories")
        print(df_categories)
        print("*" * 100, end="\n\n")

        print("Images")
        print(f"Tatal number of images: {len(df_images)}")
        print("Columns")
        print(df_images.columns)
        print("*" * 100, end="\n\n")

        print("Annotations")
        print(f"Tatal number of annotations: {len(df_annotations)}")
        print("Columns")
        print(df_annotations.columns)
        print("*" * 100, end="\n\n")

        counts, edges = np.histogram(
            df_annotations["category_id"],
            bins=np.arange(df_categories["id"].min(), df_categories["id"].max() + 2),
        )

        xticks = edges[0:-1]
        xlabels = df_categories["name"].values
        plt.bar(x=edges[0:-1], height=counts)
        plt.xticks(xticks, xlabels, rotation=90)
        plt.title(plot_title)

        plt.show()

    return json_dict, df_categories, df_images, df_annotations


# %% MASK REGISTRY


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
