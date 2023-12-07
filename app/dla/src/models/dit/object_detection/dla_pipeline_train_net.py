#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------

"""
Detection Training Script for MPViT.
"""

import itertools

# Extra Imports
import json
import logging
import os
import weakref
from typing import Any, Dict, List, Set

import cv2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from .ditod import (
    DetrDatasetMapper,
    ICDAREvaluator,
    MyDetectionCheckpointer,
    MyTrainer,
    add_vit_config,
)
# from dla_pipeline_support_functions import list_files_with_extensions


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


# Base "main" function from script
def main(args):
    """
    register publaynet first
    """
    register_coco_instances(
        "publaynet_train", {}, "./publaynet_data/train.json", "./publaynet_data/train"
    )

    register_coco_instances(
        "publaynet_val", {}, "./data/s2_dla_inputs/val.json", "./data/s2_dla_inputs"
    )

    # Test to see if I can remove
    register_coco_instances("icdar2019_train", {}, "data/train.json", "data/train")

    # Test to see if I can remove
    register_coco_instances("icdar2019_test", {}, "data/test.json", "data/test")

    cfg = setup(args)

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def inference_main(args, model_input_json, images_dir):
    print(model_input_json)
    print(images_dir)
    """
    register publaynet first
    """
    register_coco_instances(
        "publaynet_train", {}, "./publaynet_data/train.json", "./publaynet_data/train"
    )

    register_coco_instances("publaynet_val", {}, model_input_json, images_dir)

    # Test to see if I can remove
    register_coco_instances("icdar2019_train", {}, "data/train.json", "data/train")

    # Test to see if I can remove
    register_coco_instances("icdar2019_test", {}, "data/test.json", "data/test")

    cfg = setup(args)

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def run_inference(
    config_file, model_weights, output_dir, model_input_json, images_dir, num_gpus=1
):
    OPTS = [
        "--config-file",
        config_file,
        "--eval-only",
        "--num-gpus",
        str(num_gpus),
        "MODEL.WEIGHTS",
        model_weights,
        "OUTPUT_DIR",
        output_dir,
    ]

    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args(OPTS)
    print("")
    print("Command Line Args:\n", args)
    print("")

    launch(
        inference_main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, model_input_json, images_dir),
    )


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    print("")
    print("Command Line Args:\n", args)
    print("")

    if args.debug:
        import debugpy

        print("Enabling attach starts.")
        debugpy.listen(address=("0.0.0.0", 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
