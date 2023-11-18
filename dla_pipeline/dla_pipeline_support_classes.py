# %% IMPORTS

import json
import os
import pickle
from datetime import datetime
from os.path import join

import cv2
import numpy as np
import pandas as pd
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, GenericMask, Visualizer

## PDF Processing
from pdf2image import convert_from_path
from PIL import Image

## Mask Processing
from pycocotools import _mask as coco_mask
from PyPDF2 import PdfReader

## Support functions
from dla_pipeline_support_functions import (
    aggregate_masks,
    apply_border_to_mask,
    apply_mask_to_image,
    convert_non_zero_values,
    crop_stacks,
    crop_stacks_2,
    delete_contents_in_directory,
    generate_rectangular_mask,
    get_array_filled_ratio,
    get_batch,
    get_content_cluster_masks_bar_scan,
    get_content_cluster_masks_contours,
    invert_mask_polarity,
    list_files_with_extensions,
    pad_mask_perimeter,
    reset_directory,
    stack_2_arrays,
)

# %% DLA CLASSES


class DLAModelDetection:
    def __init__(self) -> None:
        self.page_id = None

        self.categories = None

        self.category = None
        self.category_lbl = None

        self.score = None

        self.x0f = None
        self.x1f = None

        self.y0f = None
        self.y1f = None

        self.x0 = None
        self.x1 = None

        self.y0 = None
        self.y1 = None

        self.mask = np.array([])
        self.rect_mask = np.array([])

        self.column = None
        self.cat_group = None
        self.score_group = None

        # Used to tag the detection for the final set
        self.is_primary = False

    @property
    def summary_dict(self):
        summary = {}

        summary["category"] = self.category
        summary["category_lbl"] = self.category_lbl

        summary["score"] = self.score

        summary["x0f"] = self.x0f
        summary["x1f"] = self.x1f
        summary["y0f"] = self.y0f
        summary["y1f"] = self.y1f

        summary["x0"] = self.x0
        summary["x1"] = self.x1
        summary["y0"] = self.y0
        summary["y1"] = self.y1

        summary["xcf"] = self.xcf
        summary["ycf"] = self.ycf

        summary["column"] = self.column
        summary["cat_group"] = self.cat_group
        summary["score_group"] = self.score_group

        summary["mask_shape"] = self.mask.shape

        summary["is_primary"] = self.is_primary

        return summary

    @property
    def is_initialized(self):
        is_initialized = True
        for k in self.summary_dict:
            if self.summary_dict[k] is None:
                is_initialized = False

        if len(self.mask) == 0 or len(self.rect_mask) == 0:
            is_initialized = False

        return is_initialized

    def initialize_from_dit_model_output_old(self, model_output, categories):

        self.categories = categories

        out_dict = model_output.get_fields()

        self.mask = out_dict["pred_masks"][0].cpu().detach().numpy().astype("uint8")

        box = out_dict["pred_boxes"][0].tensor.cpu().detach().numpy()[0]

        self.category = out_dict["pred_classes"].cpu().detach().numpy()[0]
        self.category_lbl = self.categories[self.category]

        self.score = out_dict["scores"].cpu().detach().numpy()[0]

        self.x0f = box[0]
        self.y0f = box[1]
        self.x1f = box[2]
        self.y1f = box[3]

        self.x0 = np.floor(box[0]).astype(int)
        self.y0 = np.floor(box[1]).astype(int)
        self.x1 = np.ceil(box[2]).astype(int)
        self.y1 = np.ceil(box[3]).astype(int)

        self.rect_mask = generate_rectangular_mask(
            x0=self.x0, x1=self.x1, y0=self.y0, y1=self.y1, shape=self.mask.shape
        )

    def initialize_from_dit_model_detection(self, model_detection, categories):
        self.page_id = model_detection["image_id"]

        self.categories = categories
        self.category = model_detection["category_id"]
        self.category_lbl = self.categories[self.category]

        self.score = model_detection["score"]

        self.bbox = model_detection["bbox"]

        self.x0f = self.bbox[0]
        self.y0f = self.bbox[1]

        self.dxf = self.bbox[2]
        self.dyf = self.bbox[3]

        self.x1f = self.x0f + self.dxf
        self.y1f = self.y0f + self.dyf

        self.xcf = self.x0f + (self.x1f - self.x0f) / 2
        self.ycf = self.y0f + (self.y1f - self.y0f) / 2

        self.x0 = np.floor(self.x0f).astype(int)
        self.y0 = np.floor(self.y0f).astype(int)
        self.x1 = np.ceil(self.x1f).astype(int)
        self.y1 = np.ceil(self.y1f).astype(int)

        self.box = [self.x0f, self.y0f, self.x1f, self.y1f]

        self.segmentation = model_detection["segmentation"]

        self.mask = coco_mask.decode([self.segmentation])[:, :, 0]

        self.rect_mask = generate_rectangular_mask(
            x0=self.x0, x1=self.x1, y0=self.y0, y1=self.y1, shape=self.mask.shape
        )

    # Initializer
    def initialize_from_mask(
        self,
        page_id,
        mask,
        categories,
        category,
        score=0.0,
        x0f=None,
        x1f=None,
        y0f=None,
        y1f=None,
        box_buffer=2.0,
    ):

        self.page_id = page_id

        self.categories = categories
        self.category = category
        self.category_lbl = self.categories[self.category]
        self.score = score

        self.mask = mask

        if x0f is None or x1f is None or y0f is None or y1f is None:
            positions = np.argwhere(mask != 0)

            if len(positions) > 0:
                self.x0f = positions[:, 1].min() - box_buffer
                self.x1f = positions[:, 1].max() + box_buffer

                self.y0f = positions[:, 0].min() - box_buffer
                self.y1f = positions[:, 0].max() + box_buffer
            else:
                self.x0f = 0
                self.x1f = 0

                self.y0f = 0
                self.y1f = 0

        else:
            self.x0f = x0f
            self.x1f = x1f

            self.y0f = y0f
            self.y1f = y1f

        self.xcf = self.x0f + (self.x1f - self.x0f) / 2
        self.ycf = self.y0f + (self.y1f - self.y0f) / 2

        self.x0 = np.floor(self.x0f).astype(int)
        self.x1 = np.ceil(self.x1f).astype(int)

        self.y0 = np.floor(self.y0f).astype(int)
        self.y1 = np.ceil(self.y1f).astype(int)

        self.rect_mask = generate_rectangular_mask(
            x0=self.x0, x1=self.x1, y0=self.y0, y1=self.y1, shape=self.mask.shape
        )

        self.segmentation = None

        self.box = [self.x0f, self.y0f, self.x1f, self.y1f]
        self.dxf = self.x1f - self.x0f
        self.dyf = self.y1f - self.y0f

        self.bbox = [self.x0f, self.y0f, self.dxf, self.dyf]

    def apply_mask_to_image(self, img, use_rectangular_masks=False):
        mask = self.rect_mask if use_rectangular_masks else self.mask
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        cropped_img = masked_img[self.y0 : self.y1, self.x0 : self.x1, :]

        return masked_img, cropped_img


class DLAVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)

    def draw_instance_predictions(self, predictions):

        boxes = [detect.box for detect in predictions]
        scores = [detect.score for detect in predictions]
        classes = [detect.category for detect in predictions]
        labels = [detect.category_lbl for detect in predictions]
        masks = [
            GenericMask(detect.mask, detect.mask.shape[0], detect.mask.shape[1])
            for detect in predictions
        ]
        keypoints = None

        colors = None
        alpha = 0.5

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


# %% PDF CLASSES


class PDF_DocumentGroup:
    def __init__(
        self,
        document_directory,
        page_images_directory,
        output_directory,
        use_existing_images,
        dla_categories,
        dla_score_groups=[0.8, 0.5, 0.20],
    ) -> None:
        # Log
        self.log_list = []

        # Mask Registry
        self.mask_registry = pd.DataFrame()

        # Misc
        self.use_existing_images = use_existing_images

        # Directories
        self.document_directory = document_directory
        self.page_images_directory = page_images_directory
        self.output_directory = output_directory

        self.page_masks_directory = join(self.output_directory, "page_masks")
        self.model_outputs_directory = join(self.output_directory, "model_outputs")

        # DLA
        self.dla_categories = dla_categories
        self.dla_score_groups = dla_score_groups

    def initialize_document_set(self):
        # Documents
        self.add_to_log_dict("Building document list")
        self.documents = {}
        self.__build_document_list()

        # Pages
        self.add_to_log_dict("Generating images")
        self.pages = {}
        self.__generate_pdf_page_images(self.use_existing_images)

    def __repr__(self):
        return self.document_directory

    def __build_document_list(self):
        for path in list_files_with_extensions(self.document_directory, ["pdf"]):
            try:
                pdf = PDF_Document(path)
                self.documents[pdf.title] = pdf
            except Exception as e:
                self.add_to_log_dict(f"Error loading {path}")
                self.add_to_log_dict(e)

    @property
    def document_list(self):
        doc_list = []
        for doc in self.documents.values():
            doc_list.append({"title": doc.title, "path": doc.path})

        return doc_list

    @property
    def document_list_df(self):
        return pd.DataFrame(self.document_list)

    def __generate_pdf_page_images(self, use_existing_images: bool, dpi: int = 70):
        try:
            self.pages.clear()
            for ii, doc in enumerate(self.documents.values()):
                self.add_to_log_dict(
                    f"Generating images for {doc.title}", verbose=False
                )
                if ii % 10 == 0:
                    ratio = ((ii + 1) / len(self.documents)) * 100
                    self.add_to_log_dict(
                        f"Processed {ii+1} / {len(self.documents)} documents ({ratio:0.2f}%)"
                    )

                images_exist = True
                image_paths = []

                for i in range(doc.num_pages):
                    file_name = f"{doc.title}_page_{i+1:04}"
                    file_path = join(self.page_images_directory, file_name + ".jpg")
                    image_paths.append(file_path)

                    if not os.path.exists(file_path):
                        images_exist = False

                if use_existing_images and images_exist:
                    from_existing = True
                else:
                    from_existing = False
                    page_images = convert_from_path(doc.path, dpi=dpi)

                for i, file_path in enumerate(image_paths):
                    page = PDF_Page(
                        image_path=file_path,
                        page_number=i + 1,
                        document=doc,
                        dla_categories=self.dla_categories,
                        dla_score_groups=self.dla_score_groups,
                    )

                    self.pages[page.title] = page

                    if not from_existing:
                        image = page_images[i]
                        image.save(file_path)

        except Exception as e:
            self.add_to_log_dict(f"Error generating images for {doc.path}")
            self.add_to_log_dict(e)

    @property
    def page_images_list(self):
        im_list = []
        for im in self.pages.values():
            im_list.append(
                {
                    "title": im.title,
                    "path": im.image_path,
                }
            )

        return im_list

    @property
    def page_images_list_df(self):
        return pd.DataFrame(self.page_images_list)

    def save_development_images(self, page):
        for i, title in enumerate(page.development_images.keys()):
            image = page.development_images[title]

            file_name = f"{page.title}_dev_img_{i}_{title}.jpg"
            # cv2.imwrite(join(self.model_outputs_directory, file_name), image)
            self.save_array_as_image(
                join(self.model_outputs_directory, file_name), image
            )

    def save_all_development_images(self):
        for page in self.pages.values():
            self.save_development_images(page)

    def save_masks(self, page, use_rectangular_masks):
        root_file_name = join(self.page_masks_directory, f"{page.title}_mask_")

        if page.is_initialized:
            # df = page.summary_df.reset_index()
            # df.rename(columns={"index": "mask_id"}, inplace=True)

            df = page.summary_df
            df.sort_values(by=["column", "y0"], ascending=[True, True], inplace=True)

            ii = 0
            masked_img_file_name_list = []
            mask_file_name_list = []
            new_index = []
            for i in df.index:
                dla = page.page_detections[i]

                _, masked_img = dla.apply_mask_to_image(
                    page.image, use_rectangular_masks=use_rectangular_masks
                )
                masked_img_file_name = f"{root_file_name}img_{ii}.jpg"
                masked_img_file_name_list.append(os.path.basename(masked_img_file_name))

                # cv2.imwrite(masked_img_file_name, masked_img)
                self.save_array_as_image(masked_img_file_name, masked_img)

                mask = dla.rect_mask if use_rectangular_masks else dla.mask
                mask_file_name = f"{root_file_name}{ii}.pkl"
                mask_file_name_list.append(os.path.basename(mask_file_name))

                # with open(mask_file_name, "wb") as file:
                #     pickle.dump(mask, file)

                self.save_mask_as_pkl(mask_file_name, mask)

                new_index.append(ii)
                ii += 1
            df["document"] = page.document.file_name
            df["page_no"] = page.page_number
            df["mask_id"] = new_index
            df["mask_img_file_names"] = masked_img_file_name_list
            df["mask_file_names"] = mask_file_name_list
            df.to_csv(root_file_name + "mask_summary.csv", index=True)

            return df
        else:
            raise Exception("Trying to save masks on a page that is not initialized")

    def save_all_masks(self, use_rectangular_masks):
        for page in self.pages.values():
            new_registry = self.save_masks(
                page=page, use_rectangular_masks=use_rectangular_masks
            )

            self.mask_registry = pd.concat(
                [self.mask_registry, new_registry], axis=0, ignore_index=True
            )

    def reset_output_directories(self):
        reset_directory(self.model_outputs_directory, True, False)
        reset_directory(self.page_masks_directory, True, False)

    def add_dit_model_results(self, results_json_path, categories_dict):
        try:
            with open(results_json_path, "r") as json_file:
                results_list = json.load(json_file)

            number_of_detections = len(results_list)

            for i, detection in enumerate(results_list):
                # if i % 1000 == 0:
                #     current = i + 1
                #     ratio = current / number_of_detections
                #     self.add_to_log_dict(
                #         f"Loaded {current} / {number_of_detections} detections ({ratio*100:0.2f}%)"
                #     )

                image_id = detection["image_id"]

                if image_id in self.pages:
                    dla_detection = DLAModelDetection()
                    dla_detection.initialize_from_dit_model_detection(
                        detection, categories_dict
                    )

                    self.pages[image_id].add_to_detections(dla_detection)

            for page in self.pages.values():
                page.complete_setup()

            self.add_to_log_dict(f"Loaded {number_of_detections} detections")

        except Exception as e:
            self.add_to_log_dict("Error loading DiT results")
            self.add_to_log_dict(e)
            raise Exception(e)

    def process_page_masks(
        self,
        use_rectangular_masks: bool,
        margin_thickness: int = 100,
        v_pad: int = 0,
        h_pad: int = 0,
        pad_limit_categories: list = None,
        clear_data_after: bool = False,
    ):
        # self.add_to_log_dict(f"Processing batch of {len(self.pages)} pages.")

        start = datetime.now()
        for i, page in enumerate(self.pages.values()):
            if i % 10 == 0:
                ratio = ((i + 1) / len(self.pages)) * 100
                self.add_to_log_dict(
                    f"Processed {i+1} / {len(self.pages)} pages ({ratio:0.2f}%)"
                )

            self.add_to_log_dict(f"processing: {page.title}", verbose=False)

            page.process_masks(
                use_rectangular_masks,
                margin_thickness=margin_thickness,
                v_pad=v_pad,
                h_pad=h_pad,
                pad_limit_categories=pad_limit_categories,
            )
            self.save_development_images(page=page)
            new_registry = self.save_masks(
                page=page, use_rectangular_masks=use_rectangular_masks
            )
            self.mask_registry = pd.concat(
                [self.mask_registry, new_registry], axis=0, ignore_index=True
            )

            if clear_data_after:
                page.clear_data()

        duration = datetime.now() - start

        self.add_to_log_dict(
            f"Batch processing completed:  {len(self.pages)} pages in {duration.total_seconds(): 0.2f} seconds"
        )

    def add_to_log_dict(self, message, verbose=True):
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {"time": current_timestamp, "message": message}

        self.log_list.append(entry)

        if verbose:
            print("*" * 80)
            print(entry)
            print("")

    def save_log(self, extra_tag=None):
        try:
            time_stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

            if extra_tag is None:
                file_name = f"execution_log_{time_stamp}.csv"
            else:
                file_name = f"execution_log_{time_stamp}_{extra_tag}.csv"

            path = join(self.output_directory, file_name)

            pd.DataFrame(self.log_list).to_csv(path)

            self.add_to_log_dict(f"LOG: {path}, saved")

        except Exception as e:
            self.add_to_log_dict(f"ERROR saving {path}.")
            self.add_to_log_dict(e)
            raise Exception(e)

    def save_mask_registry(self):
        try:
            file_path = join(self.page_masks_directory, "mask_registry.csv")
            self.mask_registry.to_csv(file_path, index=True)

            self.add_to_log_dict(f"MASK Registry: {file_path}, saved")
        except Exception as e:
            self.add_to_log_dict(f"ERROR saving {file_path}.")
            self.add_to_log_dict(e)
            raise Exception(e)

    def process_dit_results(
        self,
        results_json_path,
        use_rectangular_masks: bool,
        margin_thickness: int = 100,
        v_pad: int = 0,
        h_pad: int = 0,
        pad_limit_categories: list = None,
        clear_data_after: bool = False,
    ):

        self.add_to_log_dict("Loading page images")
        self.load_page_images()

        self.add_to_log_dict("Adding DIT Model Results")
        self.add_dit_model_results(
            results_json_path=results_json_path, categories_dict=self.dla_categories
        )

        self.add_to_log_dict("Processing page masks")
        self.process_page_masks(
            use_rectangular_masks=use_rectangular_masks,
            margin_thickness=margin_thickness,
            v_pad=v_pad,
            h_pad=h_pad,
            pad_limit_categories=pad_limit_categories,
            clear_data_after=clear_data_after,
        )

    def load_page_images(self):
        try:
            for page in self.pages.values():
                image_path = page.image_path
                page.image = np.array(Image.open(image_path))

        except Exception as e:
            self.add_to_log_dict(f"Error loading {image_path}")
            self.add_to_log_dict(e)

            raise Exception(e)

    def remove_empty_pages(self, results_json_path):
        try:
            with open(results_json_path, "r") as json_file:
                results_list = json.load(json_file)

            pages_list_set = set(k for k in self.pages.keys())

            detection_set = set()

            for detection in results_list:
                image_id = detection["image_id"]
                detection_set.add(image_id)

            empty_pages_set = pages_list_set - detection_set

            for k in empty_pages_set:
                pg = self.pages.pop(k)
                self.add_to_log_dict(
                    f"NO DLA Detections found in {pg.title}, removed from set"
                )

            assert len(self.pages) == len(
                detection_set
            ), "ASSERT ERROR, Unequal number of detections and pages"

        except Exception as e:
            self.add_to_log_dict("Error removing empty pages")
            self.add_to_log_dict(e)
            raise Exception(e)

    def save_array_as_image(self, file_path, image_array):
        try:
            cv2.imwrite(file_path, image_array)
            # self.add_to_log_dict(f"Saved {file_path}")

        except Exception as e:
            self.add_to_log_dict(f"ERROR Saving {file_path}")
            self.add_to_log_dict(e)

    def save_mask_as_pkl(self, file_path, mask_array):
        try:
            with open(file_path, "wb") as file:
                pickle.dump(mask_array, file)

            # self.add_to_log_dict(f"Saved {file_path}")

        except Exception as e:
            self.add_to_log_dict(f"ERROR Saving {file_path}")
            self.add_to_log_dict(e)


class PDF_DocumentSet(PDF_DocumentGroup):
    def __init__(
        self,
        document_directory,
        page_images_directory,
        output_directory,
        use_existing_images,
        dla_categories,
        dla_score_groups=[0.8, 0.5, 0.2],
    ) -> None:
        super().__init__(
            document_directory,
            page_images_directory,
            output_directory,
            use_existing_images,
            dla_categories,
            dla_score_groups,
        )

    def batch_process_dit_results(
        self,
        batch_size: int,
        results_json_path,
        use_rectangular_masks: bool,
        margin_thickness: int = 100,
        v_pad: int = 0,
        h_pad: int = 0,
        pad_limit_categories: list = None,
        clear_data_after: bool = False,
    ):

        full_pages_list = list(self.pages.values())

        batch_count = int(np.ceil(len(full_pages_list) / batch_size))

        for i, pages_list in enumerate(get_batch(full_pages_list, batch_size)):
            self.add_to_log_dict(f"=========> Processing batch {i+1} / {batch_count}")
            batch = PDF_DocumentBatch(
                document_directory=self.document_directory,
                page_images_directory=self.page_images_directory,
                output_directory=self.output_directory,
                use_existing_images=self.use_existing_images,
                dla_categories=self.dla_categories,
                dla_score_groups=self.dla_score_groups,
                pages_list=pages_list,
                log_list=self.log_list,
                mask_registry=self.mask_registry,
                batch_no=i,
            )

            batch.process_dit_results(
                results_json_path=results_json_path,
                use_rectangular_masks=use_rectangular_masks,
                margin_thickness=margin_thickness,
                v_pad=v_pad,
                h_pad=h_pad,
                pad_limit_categories=pad_limit_categories,
                clear_data_after=clear_data_after,
            )

            self.mask_registry = batch.mask_registry


class PDF_DocumentBatch(PDF_DocumentGroup):
    def __init__(
        self,
        document_directory,
        page_images_directory,
        output_directory,
        use_existing_images,
        dla_categories,
        dla_score_groups,
        pages_list,
        log_list,
        mask_registry,
        batch_no,
    ) -> None:
        super().__init__(
            document_directory,
            page_images_directory,
            output_directory,
            use_existing_images,
            dla_categories,
            dla_score_groups,
        )

        self.log_list = log_list
        self.mask_registry = mask_registry
        self.batch_no = batch_no

        self.documents = {}
        self.pages = {}

        for page in pages_list:
            self.pages[page.title] = page
            self.documents[page.document.title] = page.document


class PDF_Document:
    def __init__(self, path: str) -> None:
        assert path.lower().endswith("pdf"), f"file {path} does not end with pdf"

        self.path = path
        self.file_name = os.path.basename(path)
        self.directory = path.split(self.file_name)[0]
        self.title = os.path.splitext(self.file_name)[0]

        pdf = PdfReader(path)

        self.num_pages = len(pdf.pages)
        self.metadata = pdf.metadata

    def __repr__(self) -> str:
        return self.file_name


class PDF_Page:
    def __init__(
        self,
        image_path,
        page_number,
        document: PDF_Document,
        dla_categories,
        dla_score_groups=[0.8, 0.5, 0.20],
    ) -> None:
        # Mask processeing
        self.dla_score_groups = dla_score_groups

        # Page Data
        self.image_path = image_path
        self.page_number = page_number
        self.file_name = os.path.basename(image_path)
        self.title = os.path.splitext(self.file_name)[0]
        self.document = document

        self.image = None
        self.development_images = {}

        # DLA Detections
        self.dla_categories = dla_categories
        self.__page_detections = []
        self.is_initialized = False

    def __repr__(self) -> str:
        return self.title

    def add_to_detections(self, dla_detection):
        self.is_initialized = False
        self.__page_detections.append(dla_detection)

    def clear_detections(self):
        self.is_initialized = False
        self.__page_detections.clear()

    def pad_masks(
        self, padding: int = 0, limit_categories=None, use_rectangular_masks=False
    ):

        if padding > 0:
            padded_detections = []

            if limit_categories is None:
                limit_categories = self.dla_categories

            for dla in self.page_detections:

                if dla.category in limit_categories:
                    mask = dla.rect_mask if use_rectangular_masks else dla.mask

                    dla_new = DLAModelDetection()

                    dla_new.initialize_from_mask(
                        page_id=self.title,
                        mask=pad_mask_perimeter(mask=mask, padding=padding),
                        categories=dla.categories,
                        category=dla.category,
                        score=dla.score,
                    )

                    padded_detections.append(dla_new)

                else:
                    padded_detections.append(dla)

            self.clear_detections()
            for dla in padded_detections:
                self.add_to_detections(dla)

            self.complete_setup()

    def pad_rect_masks(self, v_pad: int = 0, h_pad: int = 0, pad_limit_categories=None):
        padded_detections = []

        if pad_limit_categories is None:
            pad_limit_categories = self.dla_categories

        for dla in self.page_detections:
            if dla.category in pad_limit_categories:
                mask = generate_rectangular_mask(
                    x0=dla.x0 - h_pad,
                    x1=dla.x1 + h_pad,
                    y0=dla.y0 - v_pad,
                    y1=dla.y1 + v_pad,
                    shape=dla.mask.shape,
                )

                dla_new = DLAModelDetection()

                dla_new.initialize_from_mask(
                    page_id=self.title,
                    mask=mask,
                    categories=dla.categories,
                    category=dla.category,
                    score=dla.score,
                )

                padded_detections.append(dla_new)

            else:
                padded_detections.append(dla)

        self.clear_detections()
        for dla in padded_detections:
            self.add_to_detections(dla)

        self.complete_setup()

    def filter_page_detections_based_on_overlap_old(
        self, quality_tresholds, filled_threshold=1e-4, use_rectangular_masks=False
    ):
        # A key assumption is that masks are sorted by priority
        inverted_mask_list = self.get_mask_list(
            by="score", ascending=True, use_rectangular_masks=use_rectangular_masks
        )

        overlapped_mask_list = crop_stacks_2(inverted_mask_list)[::-1]

        # high_q_threshold = 0.50
        # mid_q_threshold = 0.20

        high_q_threshold = quality_tresholds[0]
        mid_q_threshold = quality_tresholds[1]

        processed_page_detections = []

        for i, dla in enumerate(self.page_detections):
            add_to_detections = False

            score = dla.score

            overlapped_mask = overlapped_mask_list[i]

            if score > high_q_threshold:
                if not (np.all(overlapped_mask == 0)):
                    add_to_detections = True

            elif score > mid_q_threshold:
                # self.development_images[
                #     f"overlapped_mask_{i}_PRE-ADJ"
                # ] = convert_non_zero_values(overlapped_mask, to=250)

                masked_img, cropped_img, _, cropped_mask = apply_mask_to_image(
                    img=self.image,
                    mask=overlapped_mask,
                    box=[dla.x0, dla.x1, dla.y0, dla.y1],
                )

                remainder_masks = get_content_cluster_masks_bar_scan(
                    masked_img=masked_img,
                    initial_mask=overlapped_mask,
                    filled_threshold=filled_threshold,
                    bar_width=10,
                    scan_step=5,
                    binary_threshold=200,
                )

                if len(remainder_masks) > 0:
                    overlapped_mask = aggregate_masks(remainder_masks)
                    add_to_detections = True

                    # self.development_images[
                    #     f"overlapped_mask_{i}_POST-ADJ"
                    # ] = convert_non_zero_values(overlapped_mask, to=250)

                # else:
                #     self.development_images[f"overlapped_mask_{i}POST-ADJ"] = np.full(
                #         fill_value=255, shape=overlapped_mask.shape
                #     )

            if add_to_detections:
                new_dla = DLAModelDetection()

                new_dla.initialize_from_mask(
                    page_id=dla.page_id,
                    mask=overlapped_mask,
                    categories=dla.categories,
                    category=dla.category,
                    score=dla.score,
                )

                processed_page_detections.append(new_dla)

        self.clear_detections()

        for dla in processed_page_detections:
            self.add_to_detections(dla)

    def filter_page_detections_based_on_overlap_old2(
        self, quality_tresholds, filled_threshold=1e-4, use_rectangular_masks=False
    ):
        # A key assumption is that masks are sorted by priority
        inverted_mask_list = self.get_mask_list(
            by=["score_group", "score", "cat_group"],
            ascending=[False, True, False],
            use_rectangular_masks=use_rectangular_masks,
        )

        overlapped_mask_list = crop_stacks_2(inverted_mask_list)[::-1]

        high_q_threshold = 0.80

        processed_page_detections = []

        for i, dla in enumerate(self.page_detections):
            add_to_detections = False

            score = dla.score

            overlapped_mask = overlapped_mask_list[i]

            if score > high_q_threshold:
                if not (np.all(overlapped_mask == 0)):
                    add_to_detections = True

            else:
                # self.development_images[
                #     f"overlapped_mask_{i}_PRE-ADJ"
                # ] = convert_non_zero_values(overlapped_mask, to=250)

                masked_img, cropped_img, _, cropped_mask = apply_mask_to_image(
                    img=self.image,
                    mask=overlapped_mask,
                    box=[dla.x0, dla.x1, dla.y0, dla.y1],
                )

                remainder_masks = get_content_cluster_masks_bar_scan(
                    masked_img=masked_img,
                    initial_mask=overlapped_mask,
                    filled_threshold=filled_threshold,
                    bar_width=10,
                    scan_step=5,
                    binary_threshold=200,
                )

                if len(remainder_masks) > 0:
                    overlapped_mask = aggregate_masks(remainder_masks)
                    add_to_detections = True

                    # self.development_images[
                    #     f"overlapped_mask_{i}_POST-ADJ"
                    # ] = convert_non_zero_values(overlapped_mask, to=250)

                # else:
                #     self.development_images[f"overlapped_mask_{i}POST-ADJ"] = np.full(
                #         fill_value=255, shape=overlapped_mask.shape
                #     )

            if add_to_detections:
                new_dla = DLAModelDetection()

                new_dla.initialize_from_mask(
                    page_id=dla.page_id,
                    mask=overlapped_mask,
                    categories=dla.categories,
                    category=dla.category,
                    score=dla.score,
                )

                processed_page_detections.append(new_dla)

        self.clear_detections()

        for dla in processed_page_detections:
            self.add_to_detections(dla)

    def filter_page_detections_based_on_overlap(
        self, filled_threshold: float = 1e-4, use_rectangular_masks: bool = False
    ):
        mask_list, summary_df = self.get_mask_list(
            by="score", ascending=True, use_rectangular_masks=use_rectangular_masks
        )

        if len(mask_list) > 1:
            overlapped_mask_list = crop_stacks_2(mask_list)
        else:
            overlapped_mask_list = mask_list

        processed_page_detections = []

        for i in summary_df.index:
            dla = self.page_detections[i]
            overlapped_mask = overlapped_mask_list[i]

            masked_img, cropped_img, _, cropped_mask = apply_mask_to_image(
                img=self.image,
                mask=overlapped_mask,
                box=[dla.x0, dla.x1, dla.y0, dla.y1],
            )

            remainder_masks = get_content_cluster_masks_bar_scan(
                masked_img=masked_img,
                initial_mask=overlapped_mask,
                filled_threshold=filled_threshold,
                bar_width=10,
                scan_step=5,
                binary_threshold=200,
            )

            if len(remainder_masks) > 0:
                overlapped_mask = aggregate_masks(remainder_masks)

                new_dla = DLAModelDetection()

                new_dla.initialize_from_mask(
                    page_id=dla.page_id,
                    mask=overlapped_mask,
                    categories=dla.categories,
                    category=dla.category,
                    score=dla.score,
                )

                new_dla.is_primary = True

                processed_page_detections.append(new_dla)

            else:
                dla.is_primary = False
                processed_page_detections.append(dla)

        self.clear_detections()

        for dla in processed_page_detections:
            self.add_to_detections(dla)

        self.complete_setup()

    def filter_page_detections_based_score(
        self, score_threshold: float = 0.20, use_rectangular_masks: bool = False
    ):
        processed_page_detections = []

        for dla in self.page_detections:
            if dla.score >= score_threshold:
                mask = dla.rect_mask if use_rectangular_masks else dla.mask

                new_dla = DLAModelDetection()

                new_dla.initialize_from_mask(
                    page_id=dla.page_id,
                    mask=mask,
                    categories=dla.categories,
                    category=dla.category,
                    score=dla.score,
                )

                processed_page_detections.append(new_dla)

        self.clear_detections()

        for dla in processed_page_detections:
            self.add_to_detections(dla)

        self.complete_setup()

    def process_masks(
        self,
        use_rectangular_masks: bool,
        margin_thickness: int = 100,
        v_pad: int = 0,
        h_pad: int = 0,
        pad_limit_categories: list = None,
    ):

        # Constants

        ##############################################################################
        ## Generate visualization image
        ##############################################################################

        v = DLAVisualizer(
            img_rgb=self.image, metadata=MetadataCatalog.get("publaynet_val"), scale=1.0
        )
        result = v.draw_instance_predictions(self.page_detections)
        result_image = result.get_image()[:, :, ::-1]

        self.development_images["base_page_image"] = self.image
        self.development_images["base_dla_result"] = result_image

        ##############################################################################
        ## Filtering masks based on score
        ##############################################################################

        self.filter_page_detections_based_score(
            score_threshold=self.dla_score_groups[-1], use_rectangular_masks=True
        )

        ##############################################################################
        ## Pad Masks
        ##############################################################################

        self.pad_rect_masks(
            v_pad=v_pad, h_pad=h_pad, pad_limit_categories=pad_limit_categories
        )

        ##############################################################################
        ## Separating the high quality from low-quality masks
        ##############################################################################

        high_q_image, high_q_mask = self.apply_all_masks_to_image(
            img=self.image, use_rectangular_masks=False
        )

        low_q_mask = invert_mask_polarity(high_q_mask)
        low_q_mask = apply_border_to_mask(
            low_q_mask, border_thickness=margin_thickness, value=0
        )

        low_q_image = cv2.bitwise_and(self.image, self.image, mask=low_q_mask)

        self.development_images["img_seg_hq_mask_initial"] = high_q_image
        self.development_images["img_seg_remainder_initial"] = low_q_image

        ##############################################################################
        ## Collect remainder masks
        ##############################################################################

        remainder_masks = get_content_cluster_masks_bar_scan(
            masked_img=low_q_image,
            initial_mask=low_q_mask,
            filled_threshold=0.001,
            bar_width=10,
            scan_step=5,
            binary_threshold=200,
        )

        for mask in remainder_masks:
            dla_new = DLAModelDetection()
            dla_new.initialize_from_mask(
                page_id=self.title,
                mask=mask,
                categories=self.dla_categories,
                category=0,
                score=0,
            )

            self.add_to_detections(dla_new)

        self.complete_setup()

        ##############################################################################
        ## Filtering masks based on overlap
        ##############################################################################

        self.filter_page_detections_based_on_overlap(
            filled_threshold=1e-4,
            use_rectangular_masks=use_rectangular_masks,
        )

        ##############################################################################
        ## Save final images
        ##############################################################################

        final_masked_image, _ = self.apply_all_masks_to_image(
            img=self.image, use_rectangular_masks=use_rectangular_masks
        )

        self.development_images["img_seg_final"] = final_masked_image

    def complete_setup_old(self, page_edges_based_on_detections=False, center_offset=0):
        summary_list = [dla.summary_dict for dla in self.__page_detections]
        summary_df = pd.DataFrame(summary_list)

        if page_edges_based_on_detections:
            left_edge = summary_df["x0f"].values.min()
            right_edge = summary_df["x1f"].values.max()

        else:
            left_edge = 0
            right_edge = self.image_shape[1]

        mid_edge = left_edge + (right_edge - left_edge) * 0.50 + center_offset
        self.__edges = (left_edge, mid_edge, right_edge)

        def det_column(row):
            delta = np.abs(mid_edge - row["xcf"])

            if delta <= 10:
                marker = row["x0f"]
            else:
                marker = row["xcf"]

            return 0 if marker < mid_edge else 1

        summary_df["col"] = summary_df.apply(det_column, axis=1)

        for i in range(len(summary_df)):
            self.__page_detections[i].column = summary_df["col"][i]

        self.is_initialized = True

    def complete_setup(
        self,
        center_offset: float = 0,
    ):
        self.is_initialized = False

        left_edge = 0
        right_edge = self.image_shape[1]
        mid_edge = left_edge + (right_edge - left_edge) * 0.50 + center_offset
        self.__edges = (left_edge, mid_edge, right_edge)

        for dla in self.__page_detections:
            # Set Column
            score = dla.score
            category = dla.category
            category_lbl = dla.category_lbl

            delta = np.abs(mid_edge - dla.xcf)

            if delta <= 10:
                marker = dla.x0f
            else:
                marker = dla.xcf

            dla.column = 0 if marker < mid_edge else 1

            # Set Cat Group
            if category_lbl in ["table", "figure", "list"]:
                group = 1
            elif category_lbl == "title":
                group = 2
            elif category_lbl == "text":
                group = 3
            else:
                group = 4

            dla.cat_group = group

            # Set Score Group
            score_group = len(self.dla_score_groups)

            for i, sg in enumerate(self.dla_score_groups):
                if score > sg:
                    score_group = i
                    break

            dla.score_group = score_group

        self.is_initialized = True

    @property
    def image_shape(self):
        if self.image is None:
            return np.array([0, 0, 0])
        else:
            return self.image.shape

    @property
    def page_detections(self):
        assert (
            self.is_initialized
        ), f"ASSERTION ERROR: {self.title}, page_detections, not initialized"
        return self.__page_detections

    @property
    def summary_list(self):
        assert (
            self.is_initialized
        ), f"ASSERTION ERROR: {self.title}, summary_list, not initialized"
        return [dla.summary_dict for dla in self.__page_detections]

    @property
    def summary_df(self):
        assert (
            self.is_initialized
        ), f"ASSERTION ERROR: {self.title}, summary_df, not initialized"
        return pd.DataFrame(self.summary_list)

    @property
    def edges(self):
        assert (
            self.is_initialized
        ), f"ASSERTION ERROR: {self.title}, edges, not initialized"
        return self.__edges

    def get_mask_list(self, use_rectangular_masks=False, by="score", ascending=False):
        assert (
            self.is_initialized
        ), f"ASSERTION ERROR: {self.title}, get_mask_list, not initialized"

        summary_df = self.summary_df.sort_values(by=by, ascending=ascending)

        mask_list = []

        dla_index = np.array(summary_df.index, dtype=int)

        for i in dla_index:
            if use_rectangular_masks:
                mask_list.append(self.page_detections[i].rect_mask)
            else:
                mask_list.append(self.page_detections[i].mask)

        return mask_list, summary_df

    def get_aggregate_mask(self, mask_list=None, use_rectangular_masks=False):

        if mask_list is None:
            mask_list, _ = self.get_mask_list(use_rectangular_masks)

        aggregate_mask = np.zeros(mask_list[0].shape, dtype="uint8")

        for mask in mask_list:
            aggregate_mask = stack_2_arrays(bottom=aggregate_mask, top=mask)

        return aggregate_mask

    def apply_all_masks_to_image(self, img, use_rectangular_masks=False):

        aggregate_mask = self.get_aggregate_mask(
            use_rectangular_masks=use_rectangular_masks
        )

        return cv2.bitwise_and(img, img, mask=aggregate_mask), aggregate_mask

    def clear_data(self):
        del self.development_images
        del self.image

        for dla in self.page_detections:
            dla.mask = None
            dla.rect_mask = None

        self.development_images = {}
        self.image = None
