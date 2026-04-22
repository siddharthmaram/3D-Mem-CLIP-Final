from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
from src.const import *
from src.model_utils import *

logger = logging.getLogger(__name__)

def format_question(step):
    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")

    return question, image_goal

def get_step_info(step, verbose=False):
    question, image_goal = format_question(step)
    egocentric_imgs = step.get("egocentric_views", [])
    frontier_imgs = step["frontier_imgs"]

    snapshot_classes = {}  # rgb_id -> list of classes
    snapshot_full_imgs = {}  # rgb_id -> full img
    snapshot_crops = {}  # rgb_id -> list of crops
    snapshot_clusters = {}  # rgb_id -> list of clusters
    obj_map = step["obj_map"]
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]["full_img"]
        snapshot_full_imgs[rgb_id] = snapshot_img
        snapshot_crops[rgb_id] = [
            crop_data["crop"]
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        snapshot_class = [
            crop_data["obj_class"]
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        cluster_class = [
            obj_map[int(obj_id)] for obj_id in step["snapshot_objects"][rgb_id]
        ]
        seen_classes.update(sorted(list(set(snapshot_class))))
        snapshot_classes[rgb_id] = snapshot_class
        snapshot_clusters[rgb_id] = cluster_class

    keep_index_snapshot = {
        rgb_id: list(range(len(snapshot_crops[rgb_id]))) for rgb_id in snapshot_crops
    }

    top_k = step.get("top_k_categories", 5)
    
    ranked_results = get_top_k_similar_objects(question, list(seen_classes), k=top_k, image_goal=image_goal)
    selected_classes = [res["label"] for res in ranked_results]

    keep_index = [
        i
        for i, k in enumerate(snapshot_clusters.keys())
        if len(set(snapshot_clusters[k]) & set(selected_classes)) > 0
    ]
    keep_snapshot_id = [list(snapshot_classes.keys())[i] for i in keep_index]
    snapshot_classes = {rgb_id: snapshot_classes[rgb_id] for rgb_id in keep_snapshot_id}

    keep_index_snapshot = {}
    for rgb_id in keep_snapshot_id:
        keep_index_snapshot[rgb_id] = [
            i
            for i in range(len(snapshot_classes[rgb_id]))
            if snapshot_classes[rgb_id][i] in selected_classes
        ]
        snapshot_classes[rgb_id] = [
            snapshot_classes[rgb_id][i] for i in keep_index_snapshot[rgb_id]
        ]

    snapshot_full_imgs = {
        rgb_id: snapshot_full_imgs[rgb_id] for rgb_id in keep_index_snapshot.keys()
    }

    for rgb_id in snapshot_classes.keys():
        snapshot_crops[rgb_id] = [
            snapshot_crops[rgb_id][i] for i in keep_index_snapshot[rgb_id]
        ]

    return (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        keep_index,
        keep_index_snapshot,
    )


def explore_step(step, cfg, verbose=False):
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        snapshot_id_mapping,
        snapshot_crop_mapping,
    ) = get_step_info(step, verbose)

    snapshot_images = []
    snapshot_crops_idx = {}

    for i, rgb_id in enumerate(snapshot_full_imgs.keys()):
        snapshot_images.append((i, snapshot_full_imgs[rgb_id]))
        snapshot_crops_idx[i] = snapshot_crops[rgb_id]

    target_type, snapshot_idx, object_idx = get_frontier(
        question, 
        frontier_imgs, 
        snapshot_images,
        snapshot_crops_idx,
        image_goal=image_goal
    )

    if target_type == "snapshot":
        final_response = f"snapshot {snapshot_idx}, object {object_idx}"
    else:
        final_response = f"frontier {snapshot_idx}"

    return (
        final_response,
        snapshot_id_mapping,
        snapshot_crop_mapping,
        "",
        len(snapshot_full_imgs),
    )